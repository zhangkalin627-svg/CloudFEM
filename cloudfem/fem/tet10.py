"""Second-order (Tet10) finite elements: scalar problems (heat).

Solves the same class of problems as :mod:`cloudfem.fem.tet_poisson`
and :mod:`cloudfem.fem.tet_heat`, ::

    -div(k(x) grad u) = f(x)          (steady)
    rho_cp * dT/dt = div(k grad T) + f   (transient, theta method)

but with quadratic (10-node) tetrahedral elements instead of P1.
With barycentric coordinates ``L = (L1, L2, L3, L4)`` attached to the
element corners, the shape functions are

    corner i:        N_i  = L_i (2 L_i - 1)
    edge (i, j):     N_ij = 4 L_i L_j

in the local node ordering of :mod:`cloudfem.mesh.tet10_mesh`.

Integration
-----------
Stiffness and load integrals use the 5-point Keast quadrature rule
for tetrahedra (exact for polynomials of degree <= 3). On affine
elements the Tet10 stiffness integrand ``grad N_i . grad N_j`` is
degree 2, so the rule integrates it *exactly* for constant ``k``; for
space-varying ``k``/``f`` they are sampled at the quadrature points
(the P1 code samples at the centroid).

The consistent mass matrix is integrated *exactly* in closed form:
every shape function is a quadratic polynomial in ``L``, products are
degree 4 (beyond the quadrature rule), and the barycentric moments
``int L^a dV = V * 6 * a! / (|a| + 3)!`` give each entry analytically.

Element maps are affine (edge midpoints at geometric midpoints, see
:mod:`cloudfem.mesh.tet10_mesh`), so gradients use the constant
corner Jacobian, exactly as in the P1 code.

Scope limits (v1.5): Dirichlet boundary conditions only for the
transient solver (no Neumann/Robin face terms yet); steady solves
accept the same ``bc`` dict style as tet_poisson.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh.tet10_mesh import EDGES
from .tet_poisson import apply_dirichlet

__all__ = [
    "shape_functions", "quadrature", "element_operators",
    "assemble_tet10", "assemble_mass_tet10", "solve_tet10_poisson",
    "transient_heat_tet10",
]


# ------------------------------------------------------------------
# Shape functions and quadrature
# ------------------------------------------------------------------

def shape_functions(L):
    """Tet10 shape functions and barycentric derivatives.

    L : (..., 4) barycentric coordinates (rows sum to 1).
    Returns (N, dN_dL) with shapes (..., 10) and (..., 10, 4):
    N[..., a] is the shape function of local node a and
    dN_dL[..., a, i] = dN_a / dL_i.
    """
    L = np.asarray(L, dtype=float)
    N = np.empty(L.shape[:-1] + (10,))
    dN = np.zeros(L.shape[:-1] + (10, 4))
    for i in range(4):
        N[..., i] = L[..., i] * (2.0 * L[..., i] - 1.0)
        dN[..., i, i] = 4.0 * L[..., i] - 1.0
    for k, (i, j) in enumerate(EDGES):
        N[..., 4 + k] = 4.0 * L[..., i] * L[..., j]
        dN[..., 4 + k, i] = 4.0 * L[..., j]
        dN[..., 4 + k, j] = 4.0 * L[..., i]
    return N, dN


def quadrature():
    """5-point Keast rule for the tetrahedron (degree-3 exact).

    Returns (Lq, wq): barycentric points (5, 4) and weights (5,)
    normalised so ``sum(wq) = 1`` and, for any polynomial g,
    ``sum_q wq * g(Lq) = (1/V) * int g dV`` on any tetrahedron.
    """
    a, b = 0.5, 1.0 / 6.0
    Lq = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a],
    ])
    wq = np.array([-4.0 / 5.0] + [9.0 / 20.0] * 4)
    return Lq, wq


def element_operators(mesh, el):
    """Quadrature data for one Tet10 element.

    Returns (G, N, xq, vol):
      G  : (Q, 10, 3) physical gradients grad(N_a) at the Q=5
           quadrature points,
      N  : (Q, 10) shape-function values,
      xq : (Q, 3) physical quadrature-point coordinates,
      vol: element volume (from the corner Jacobian).
    """
    el = np.asarray(el, dtype=int)
    p = mesh.nodes[el[:4]]                       # (4, 3) corners
    J = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)
    det = float(np.linalg.det(J))
    vol = abs(det) / 6.0
    Lq, _ = quadrature()
    N, dN_dL = shape_functions(Lq)               # (Q,10), (Q,10,4)
    # reference coords (xi, eta, zeta) = (L2, L3, L4), L1 = 1-xi-eta-zeta
    dN_dxi = np.stack([dN_dL[..., 1] - dN_dL[..., 0],
                       dN_dL[..., 2] - dN_dL[..., 0],
                       dN_dL[..., 3] - dN_dL[..., 0]], axis=-1)  # (Q,10,3)
    G = dN_dxi @ np.linalg.inv(J)                # rows: grad_x = grad_xi J^-1
    xq = Lq @ p
    return G, N, xq, vol


def _scalar_fn(v):
    """Normalise a float-or-callable(x, y, z) spec to a callable."""
    return v if callable(v) else (lambda x, y, z, v=v: v)


# ------------------------------------------------------------------
# Steady scalar problem (heat conduction / Poisson)
# ------------------------------------------------------------------

def assemble_tet10(mesh, k=1.0, f=0.0):
    """Assemble Tet10 stiffness matrix A and load vector rhs.

    k, f : floats or callables(x, y, z), sampled at the quadrature
    points. Returns (A csr_matrix (N, N), rhs (N,)).
    """
    n = mesh.n_nodes
    k_fn = _scalar_fn(k)
    f_fn = _scalar_fn(f)
    _, wq = quadrature()
    rows, cols, data = [], [], []
    rhs = np.zeros(n)
    for el in mesh.elements:
        G, N, xq, vol = element_operators(mesh, el)
        kv = np.array([k_fn(x, y, z) for x, y, z in xq])
        fv = np.array([f_fn(x, y, z) for x, y, z in xq])
        Ke = vol * np.einsum("q,qji,qki->jk", wq * kv, G, G)
        fe = vol * (N * (wq * fv)[:, None]).sum(axis=0)
        ii = np.repeat(el, 10)
        jj = np.tile(el, 10)
        rows.append(ii)
        cols.append(jj)
        data.append(Ke.ravel())
        rhs[el] += fe
    A = sp.coo_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return A, rhs


def solve_tet10_poisson(mesh, k=1.0, f=0.0, bc=None):
    """Solve -div(k grad u) = f on a Tet10Mesh with Dirichlet BCs.

    bc : dict as in tet_poisson (boundary name -> value or
    callable(x, y, z)). Returns the nodal solution array u.
    """
    A, rhs = assemble_tet10(mesh, k=k, f=f)
    if bc:
        A, rhs = apply_dirichlet(A, rhs, mesh, bc)
    u = spla.spsolve(A, rhs)
    return np.asarray(u).ravel()


# ------------------------------------------------------------------
# Consistent mass matrix (exact closed-form integrals)
# ------------------------------------------------------------------

def _shape_polys():
    """Each Tet10 shape function as [(coef, (a, b, c, d)), ...] in L."""
    polys = []
    for i in range(4):
        e2 = [0, 0, 0, 0]
        e2[i] = 2
        e1 = [0, 0, 0, 0]
        e1[i] = 1
        polys.append([(2.0, tuple(e2)), (-1.0, tuple(e1))])
    for (i, j) in EDGES:
        e = [0, 0, 0, 0]
        e[i] += 1
        e[j] += 1
        polys.append([(4.0, tuple(e))])
    return polys


def _bary_moment(alpha):
    """(1/V) * int L^alpha dV = 6 * alpha! / (|alpha| + 3)!."""
    from math import factorial
    num = 1
    for a in alpha:
        num *= factorial(a)
    return 6.0 * num / factorial(sum(alpha) + 3)


def _mass_reference():
    """Unit-volume Tet10 mass matrix int N_a N_b dV / V (10, 10)."""
    polys = _shape_polys()
    M = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            s = 0.0
            for c1, a1 in polys[a]:
                for c2, a2 in polys[b]:
                    alpha = tuple(x + y for x, y in zip(a1, a2))
                    s += c1 * c2 * _bary_moment(alpha)
            M[a, b] = s
    return M


_MASS_REF = _mass_reference()


def assemble_mass_tet10(mesh, rho_cp=1.0):
    """Consistent Tet10 capacity/mass matrix int rho_cp N_i N_j dV.

    Exact closed-form element integrals (see module docstring).
    rho_cp : float or callable(x, y, z) (element centroid value).
    Returns csr_matrix (N, N). Row sums are rho_cp * int N_i dV, with
    int N_i dV = -V/20 for corner nodes and +V/5 for midpoint nodes.
    """
    n = mesh.n_nodes
    rc_fn = _scalar_fn(rho_cp)
    vols = mesh.element_volumes()
    rows, cols, data = [], [], []
    for e, el in enumerate(mesh.elements):
        cen = mesh.nodes[el[:4]].mean(axis=0)
        me = rc_fn(*cen) * vols[e] * _MASS_REF
        rows.append(np.repeat(el, 10))
        cols.append(np.tile(el, 10))
        data.append(me.ravel())
    return sp.coo_matrix((np.concatenate(data),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


# ------------------------------------------------------------------
# Transient heat conduction (theta method, Dirichlet BCs)
# ------------------------------------------------------------------

_THETA = {"be": 1.0, "implicit_euler": 1.0,
          "cn": 0.5, "crank_nicolson": 0.5}


def _dirichlet_values(mesh, bc):
    """Resolve a tet_poisson-style bc dict to {node_index: value}."""
    fixed = {}
    if not bc:
        return fixed
    for name, value in bc.items():
        if name == "all":
            ids = np.arange(mesh.n_nodes)
        else:
            ids = mesh.boundaries[name]
        v_fn = _scalar_fn(value)
        for i in ids:
            i = int(i)
            x, y, z = mesh.nodes[i]
            fixed[i] = v_fn(x, y, z)
    return fixed


def transient_heat_tet10(mesh, k=1.0, rho_cp=1.0, f=0.0, bc=None,
                         t_end=1.0, dt=0.1, T0=0.0, method="cn"):
    """Transient heat conduction on a Tet10Mesh (theta method).

    Same conventions as tet_heat.transient_heat_tet, except that only
    Dirichlet boundary conditions are supported (v1.5 scope). The
    step actually used is t_end / ceil(t_end / dt).

    Returns (times, T_history) with times[0] = 0.
    """
    if method not in _THETA:
        raise ValueError(f"unknown method {method!r}; "
                         f"use one of {sorted(_THETA)}")
    theta = _THETA[method]
    n = mesh.n_nodes

    K, F = assemble_tet10(mesh, k=k, f=f)
    C = assemble_mass_tet10(mesh, rho_cp=rho_cp)

    n_steps = int(np.ceil(t_end / dt - 1e-12))
    if n_steps < 1:
        raise ValueError("t_end must be positive")
    h = t_end / n_steps

    A_sys = C + theta * h * K
    R_sys = C - (1.0 - theta) * h * K

    fixed = _dirichlet_values(mesh, bc)
    if fixed:
        A_bc, _ = apply_dirichlet(A_sys, np.zeros(n), mesh, bc)
        fixed_idx = np.array(sorted(fixed), dtype=int)
        fixed_val = np.array([fixed[i] for i in fixed_idx])
        cols = A_sys.tocsr()[:, fixed_idx]
    else:
        A_bc = A_sys
        fixed_idx = fixed_val = cols = None
    solve = spla.factorized(A_bc.tocsc())

    T = (np.full(n, T0, dtype=float) if np.isscalar(T0)
         else np.asarray(T0, dtype=float).copy())
    if fixed_idx is not None:
        T[fixed_idx] = fixed_val

    times = [0.0]
    hist = [T.copy()]
    for _ in range(n_steps):
        rhs = R_sys @ T + h * F
        if fixed_idx is not None:
            rhs = rhs - cols @ fixed_val
            rhs[fixed_idx] = fixed_val
        T = np.asarray(solve(rhs)).ravel()
        if fixed_idx is not None:
            T[fixed_idx] = fixed_val
        times.append(times[-1] + h)
        hist.append(T.copy())
    times[-1] = t_end
    return np.array(times), np.array(hist)
