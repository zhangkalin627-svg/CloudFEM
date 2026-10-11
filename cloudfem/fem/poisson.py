"""P1 (linear triangular) finite element assembly.

Solves scalar elliptic problems of the form::

    -div(k(x) grad u) + c(x) u = f(x)    in Omega

with Dirichlet boundary conditions. Supports real and complex
arithmetic (complex is used by the frequency-domain EM solver).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _gradients(p):
    """Gradients of P1 basis functions on one triangle.

    p : (3, 2) node coordinates.
    Returns b, c with grad(phi_i) = (b_i, c_i), and area.
    """
    x = p[:, 0]
    y = p[:, 1]
    area2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    area = 0.5 * area2
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / area2
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / area2
    return b, c, abs(area)


def assemble(mesh, k=1.0, c=0.0, f=0.0, dtype=float, neumann=None):
    """Assemble stiffness matrix A and load vector rhs.

    Parameters
    ----------
    mesh : TriMesh
    k : float or callable(x, y) -> float, diffusion coefficient.
    c : float or callable, reaction coefficient.
    f : float or callable(x, y), source term.
    dtype : float or complex.
    neumann : dict {boundary name: g} for Neumann BCs, adding
        int_Gamma g*v ds to the load vector. g is the *fully weighted*
        boundary integrand factor (e.g. heat flux k*dT/dn in planar
        problems), float or callable(x, y).

    Returns
    -------
    A : csr_matrix (N, N), rhs : (N,) array.
    """
    n = mesh.n_nodes
    rows, cols, data = [], [], []
    rhs = np.zeros(n, dtype=dtype)

    k_fn = k if callable(k) else lambda x, y: k
    c_fn = c if callable(c) else lambda x, y: c
    f_fn = f if callable(f) else lambda x, y: f

    for el in mesh.elements:
        p = mesh.nodes[el]
        b, cc, area = _gradients(p)
        xc, yc = p[:, 0].mean(), p[:, 1].mean()
        ke = k_fn(xc, yc)
        ce = c_fn(xc, yc)
        fe = f_fn(xc, yc)

        # element stiffness: k * area * (b_i b_j + c_i c_j)
        grad = np.outer(b, b) + np.outer(cc, cc)
        ae = ke * area * grad
        if np.any(np.asarray(ce) != 0.0):
            # lumped-ish consistent mass: ce * area/12 * [[2,1,1],[1,2,1],[1,1,2]]
            me = ce * area / 12.0 * (np.ones((3, 3)) + np.eye(3))
            ae = ae + me

        for i in range(3):
            rhs[el[i]] += fe * area / 3.0  # f constant per element
            for j in range(3):
                rows.append(el[i])
                cols.append(el[j])
                data.append(ae[i, j])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=dtype).tocsr()

    if neumann:
        bedges = mesh.boundary_edges()
        for name, g in neumann.items():
            g_fn = g if callable(g) else (lambda x, y, g=g: g)
            ids = set(int(i) for i in mesh.boundaries[name])
            for i, j in bedges:
                if i in ids and j in ids:
                    pi, pj = mesh.nodes[i], mesh.nodes[j]
                    length = float(np.linalg.norm(pj - pi))
                    xm, ym = (pi[0] + pj[0]) / 2.0, (pi[1] + pj[1]) / 2.0
                    gv = g_fn(xm, ym)
                    rhs[i] += gv * length / 2.0
                    rhs[j] += gv * length / 2.0

    return A, rhs


def apply_dirichlet(A, rhs, mesh, bc):
    """Apply Dirichlet BCs.

    bc : dict mapping boundary name (or 'all') -> value or callable(x, y).
         A boundary name must exist in mesh.boundaries.
    Returns modified (A, rhs); constrained nodes get identity rows.
    """
    A = A.tolil()
    rhs = rhs.copy()
    n = mesh.n_nodes
    fixed = {}

    for name, value in bc.items():
        if name == "all":
            ids = np.arange(n)
        else:
            ids = mesh.boundaries[name]
        v_fn = value if callable(value) else (lambda x, y, v=value: v)
        for i in ids:
            x, y = mesh.nodes[int(i)]
            fixed[int(i)] = v_fn(x, y)

    for i, v in fixed.items():
        A.rows[i] = [i]
        A.data[i] = [1.0]
        rhs[i] = v
    # zero out column entries for fixed dofs (symmetric handling not needed
    # for correctness of the direct solve on remaining dofs)
    A = A.tocsr()
    fixed_idx = np.array(sorted(fixed), dtype=int)
    if len(fixed_idx):
        # subtract known contributions: rhs -= A[:, fixed] * v_fixed
        # (do before zeroing columns)
        v = np.array([fixed[i] for i in fixed_idx])
        rhs = rhs - A[:, fixed_idx] @ v
        # re-set rhs at fixed nodes (the matvec above touched them)
        for i, vv in fixed.items():
            rhs[i] = vv
        A[:, fixed_idx] = 0
        A = A.tocsr()
        for i, vv in fixed.items():
            A[i, i] = 1.0
    return A, rhs


def solve_poisson(mesh, k=1.0, c=0.0, f=0.0, bc=None, dtype=float,
                  neumann=None):
    """Solve -div(k grad u) + c u = f with Dirichlet BCs.

    Returns the nodal solution array u.
    """
    A, rhs = assemble(mesh, k=k, c=c, f=f, dtype=dtype, neumann=neumann)
    if bc:
        A, rhs = apply_dirichlet(A, rhs, mesh, bc)
    u = spla.spsolve(A, rhs)
    return np.asarray(u).ravel()
