"""Linear elasticity (structural field): P1 vector finite elements.

Solves::

    -div sigma(u) = b            in Omega
    sigma(u) = C : (eps(u) - eps_th)
    eps(u) = sym(grad u)
    eps_th = alpha * (T - T_ref) * I      (thermal strain)

Plane stress and plane strain are supported. Two displacement DOFs per
node (ux, uy). The thermal-strain load turns a temperature field into a
mechanical load, which is the sequential EM -> thermal -> structural
chain used for induction-heating thermal-stress analysis.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fem.poisson import _gradients


def constitutive(E, nu, plane="strain"):
    """3x3 constitutive matrix D in Voigt notation [xx, yy, xy]."""
    if plane == "strain":
        fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = fac * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])
    elif plane == "stress":
        fac = E / (1.0 - nu ** 2)
        D = fac * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
    else:
        raise ValueError(f"plane must be 'strain' or 'stress', got {plane!r}")
    return D


def _strain_displacement(p):
    """Strain-displacement matrix B (3, 6) and area for one P1 triangle."""
    b, c, area = _gradients(p)
    B = np.array([
        [b[0], 0.0, b[1], 0.0, b[2], 0.0],
        [0.0, c[0], 0.0, c[1], 0.0, c[2]],
        [c[0], b[0], c[1], b[1], c[2], b[2]],
    ])
    return B, area


def _dof_map(el):
    dofs = np.empty(6, dtype=int)
    dofs[0::2] = 2 * el
    dofs[1::2] = 2 * el + 1
    return dofs


def assemble_elasticity(mesh, E, nu, plane="strain", body=None,
                        T=None, alpha=0.0, T_ref=0.0):
    """Assemble the 2N x 2N stiffness matrix and load vector.

    Parameters
    ----------
    mesh : TriMesh
    E, nu : Young's modulus, Poisson's ratio (floats).
    plane : 'strain' or 'stress'.
    body : None, (bx, by) floats, or (bx_fn, by_fn) callables f(x, y).
    T : None, nodal temperature array, or callable f(x, y).
    alpha : thermal expansion coefficient (1/K).
    T_ref : stress-free reference temperature.

    Returns
    -------
    A : csr_matrix (2N, 2N), rhs : (2N,) array, dof order ux0, uy0, ux1, ...
    """
    D = constitutive(E, nu, plane)
    n2 = 2 * mesh.n_nodes
    rows, cols, data = [], [], []
    rhs = np.zeros(n2)

    if body is None:
        bx_fn = by_fn = lambda x, y: 0.0
    elif callable(body[0]):
        bx_fn, by_fn = body
    else:
        bx, by = body
        bx_fn = lambda x, y: bx
        by_fn = lambda x, y: by

    T_is_callable = callable(T)
    if T is not None and not T_is_callable:
        T = np.asarray(T, dtype=float)

    eye2 = np.array([1.0, 1.0, 0.0])  # Voigt identity

    for el in mesh.elements:
        p = mesh.nodes[el]
        B, area = _strain_displacement(p)
        ke = (B.T @ D @ B) * area
        dofs = _dof_map(el)
        for i in range(6):
            for j in range(6):
                rows.append(dofs[i])
                cols.append(dofs[j])
                data.append(ke[i, j])

        # body force, constant per element evaluated at the centroid
        xc, yc = p[:, 0].mean(), p[:, 1].mean()
        bxv, byv = bx_fn(xc, yc), by_fn(xc, yc)
        for k in range(3):
            rhs[dofs[2 * k]] += bxv * area / 3.0
            rhs[dofs[2 * k + 1]] += byv * area / 3.0

        # thermal load: int B^T D eps_th dA, eps_th constant per element
        if T is not None and alpha != 0.0:
            Tc = T(xc, yc) if T_is_callable else T[el].mean()
            eth = alpha * (Tc - T_ref) * eye2
            fth = (B.T @ D @ eth) * area
            rhs[dofs] += fth

    A = sp.coo_matrix((data, (rows, cols)), shape=(n2, n2)).tocsr()
    return A, rhs


def apply_dirichlet_vec(A, rhs, mesh, bc):
    """Apply displacement Dirichlet BCs to the vector system.

    bc : dict mapping boundary name (or 'all') -> (ux, uy). Each component
         may be a float, a callable f(x, y), or None (leave free).

    Returns the full (2N,) solution vector with constrained DOFs filled
    from the BC values.
    """
    rhs = np.asarray(rhs, dtype=float).copy()
    n2 = 2 * mesh.n_nodes
    fixed = {}

    for name, vals in bc.items():
        if name == "all":
            ids = np.arange(mesh.n_nodes)
        else:
            ids = np.asarray(mesh.boundaries[name])
        ux_v, uy_v = vals
        for comp, v in ((0, ux_v), (1, uy_v)):
            if v is None:
                continue
            v_fn = v if callable(v) else (lambda x, y, v=v: v)
            for i in ids:
                i = int(i)
                x, y = mesh.nodes[i]
                fixed[2 * i + comp] = v_fn(x, y)

    u = np.zeros(n2)
    fixed_idx = np.array(sorted(fixed), dtype=int)
    if len(fixed_idx):
        v = np.array([fixed[i] for i in fixed_idx])
        u[fixed_idx] = v
        free = np.ones(n2, dtype=bool)
        free[fixed_idx] = False
        A = A.tocsr()
        rhs_free = rhs[free] - A[free][:, fixed_idx] @ v
        A_free = A[free][:, free].tocsr()
        u[free] = spla.spsolve(A_free, rhs_free)
    else:
        u = np.asarray(spla.spsolve(A.tocsr(), rhs)).ravel()
    return u


def solve_elasticity(mesh, E, nu, plane="strain", bc=None, body=None,
                     T=None, alpha=0.0, T_ref=0.0):
    """Solve the linear elasticity problem.

    Returns (ux, uy), the nodal displacement components.
    """
    A, rhs = assemble_elasticity(mesh, E, nu, plane=plane, body=body,
                                 T=T, alpha=alpha, T_ref=T_ref)
    if bc:
        u = apply_dirichlet_vec(A, rhs, mesh, bc)
    else:
        u = np.asarray(spla.spsolve(A.tocsr(), rhs)).ravel()
    return u[0::2].copy(), u[1::2].copy()


def element_stress_strain(mesh, u, E, nu, plane="strain",
                          T=None, alpha=0.0, T_ref=0.0):
    """Per-element strain and stress in Voigt notation [xx, yy, xy].

    u : (2N,) displacement vector, or (ux, uy) tuple of nodal arrays.
    T : None, nodal temperature array, or callable f(x, y).
    Returns (sig, eps), each (M, 3).
    """
    if isinstance(u, tuple):
        ux, uy = u
        uv = np.empty(2 * mesh.n_nodes)
        uv[0::2] = ux
        uv[1::2] = uy
    else:
        uv = np.asarray(u, dtype=float).ravel()

    D = constitutive(E, nu, plane)
    T_is_callable = callable(T)
    if T is not None and not T_is_callable:
        T = np.asarray(T, dtype=float)
    eye2 = np.array([1.0, 1.0, 0.0])

    sig = np.zeros((mesh.n_elements, 3))
    eps = np.zeros((mesh.n_elements, 3))
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        B, _ = _strain_displacement(p)
        dofs = _dof_map(el)
        ee = B @ uv[dofs]
        eps[e] = ee
        eth = np.zeros(3)
        if T is not None and alpha != 0.0:
            xc, yc = p[:, 0].mean(), p[:, 1].mean()
            Tc = T(xc, yc) if T_is_callable else T[el].mean()
            eth = alpha * (Tc - T_ref) * eye2
        sig[e] = D @ (ee - eth)
    return sig, eps


def von_mises(sig):
    """Von Mises equivalent stress for Voigt stress [xx, yy, xy].

    Accepts (M, 3) or (3,) arrays.
    """
    s = np.asarray(sig, dtype=float)
    sx, sy, txy = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2)
