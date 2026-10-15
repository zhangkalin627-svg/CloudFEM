"""P1 (linear tetrahedral) finite element assembly.

Solves scalar elliptic problems of the form::

    -div(k(x) grad u) = f(x)    in Omega

with Dirichlet boundary conditions. v0.7 scope is steady heat
conduction on 3D box meshes; transient stepping reuses the same
element matrices via the 2D timestep machinery pattern.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _tet_gradients(p):
    """Gradients of P1 basis functions on one tetrahedron.

    p : (4, 3) node coordinates.
    Returns G (4, 3) with G[i] = grad(phi_i), and volume.
    """
    m = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)  # (3, 3)
    det = np.linalg.det(m)
    vol = abs(det) / 6.0
    # reference gradients: phi0 = 1-x-y-z, phi1 = x, phi2 = y, phi3 = z
    g_ref = np.array([[-1.0, -1.0, -1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    g_phys = np.linalg.solve(m.T, g_ref.T).T  # (4, 3)
    return g_phys, vol


def assemble(mesh, k=1.0, f=0.0):
    """Assemble stiffness matrix A and load vector rhs for a TetMesh.

    k : float or callable(x, y, z) -> float, diffusion coefficient
        (evaluated at the element centroid).
    f : float or callable(x, y, z), source term (element centroid).

    Returns A : csr_matrix (N, N), rhs : (N,) array.
    """
    n = mesh.n_nodes
    rows, cols, data = [], [], []
    rhs = np.zeros(n)

    k_fn = k if callable(k) else (lambda x, y, z, k=k: k)
    f_fn = f if callable(f) else (lambda x, y, z, f=f: f)

    for el in mesh.elements:
        p = mesh.nodes[el]
        g, vol = _tet_gradients(p)
        xc, yc, zc = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()
        ke = k_fn(xc, yc, zc)
        fe = f_fn(xc, yc, zc)

        # element stiffness: k * V * (g_i . g_j)
        ae = ke * vol * (g @ g.T)

        for i in range(4):
            rhs[el[i]] += fe * vol / 4.0
            for j in range(4):
                rows.append(el[i])
                cols.append(el[j])
                data.append(ae[i, j])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, rhs


def apply_dirichlet(A, rhs, mesh, bc):
    """Apply Dirichlet BCs to a tetrahedral system.

    bc : dict mapping boundary name (or 'all') -> value or
         callable(x, y, z). Boundary names must exist in
         mesh.boundaries.
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
        v_fn = value if callable(value) else (lambda x, y, z, v=value: v)
        for i in ids:
            x, y, z = mesh.nodes[int(i)]
            fixed[int(i)] = v_fn(x, y, z)

    for i, v in fixed.items():
        A.rows[i] = [i]
        A.data[i] = [1.0]
        rhs[i] = v
    A = A.tocsr()
    fixed_idx = np.array(sorted(fixed), dtype=int)
    if len(fixed_idx):
        v = np.array([fixed[i] for i in fixed_idx])
        rhs = rhs - A[:, fixed_idx] @ v
        for i, vv in fixed.items():
            rhs[i] = vv
        A[:, fixed_idx] = 0
        A = A.tocsr()
        for i, vv in fixed.items():
            A[i, i] = 1.0
    return A, rhs


def solve_tet_poisson(mesh, k=1.0, f=0.0, bc=None):
    """Solve -div(k grad u) = f on a TetMesh with Dirichlet BCs.

    Returns the nodal solution array u.
    """
    A, rhs = assemble(mesh, k=k, f=f)
    if bc:
        A, rhs = apply_dirichlet(A, rhs, mesh, bc)
    u = spla.spsolve(A, rhs)
    return np.asarray(u).ravel()
