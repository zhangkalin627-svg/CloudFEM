"""Linear elasticity in 3D (structural field): P1 vector finite elements.

Solves::

    -div sigma(u) = b            in Omega
    sigma(u) = C : (eps(u) - eps_th)
    eps(u) = sym(grad u)
    eps_th = alpha * (T - T_ref) * I      (thermal strain)

Voigt notation uses the order [xx, yy, zz, yz, xz, xy] with
engineering shear strains (gamma_yz = du_y/dz + du_z/dy, etc.).
Three displacement DOFs per node (ux, uy, uz), interleaved so that
the DOF of component c of node i is 3*i + c.

Assumptions: small strains, isotropic linear elastic material, and
P1 (linear tetrahedral, constant-strain) elements.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fem.tet_poisson import _tet_gradients


def constitutive_3d(E, nu):
    """6x6 constitutive matrix D in Voigt notation [xx, yy, zz, yz, xz, xy]."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def boundary_faces(mesh, name):
    """Exterior triangular faces on the named boundary.

    Returns (F, 3) int array of faces that belong to exactly one
    tetrahedron and whose 3 nodes all lie in mesh.boundaries[name].
    """
    ids = np.asarray(mesh.boundaries[name])
    idset = set(int(i) for i in ids)
    # count each tet face (sorted node triple) over all tetrahedra
    counts = {}
    faces = {}
    for el in mesh.elements:
        a, b, c, d = (int(el[0]), int(el[1]), int(el[2]), int(el[3]))
        for tri in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(tri))
            counts[key] = counts.get(key, 0) + 1
            if key not in faces:
                faces[key] = tri
    out = [faces[key] for key, cnt in counts.items()
           if cnt == 1 and all(n in idset for n in key)]
    if not out:
        return np.zeros((0, 3), dtype=int)
    return np.array(out, dtype=int)


def _strain_displacement(G):
    """Strain-displacement matrix B (6, 12) from basis gradients G (4, 3)."""
    B = np.zeros((6, 12))
    for k in range(4):
        gx, gy, gz = G[k]
        j = 3 * k
        B[0, j] = gx
        B[1, j + 1] = gy
        B[2, j + 2] = gz
        B[3, j + 1] = gz
        B[3, j + 2] = gy
        B[4, j] = gz
        B[4, j + 2] = gx
        B[5, j] = gy
        B[5, j + 1] = gx
    return B


def _dof_map(el):
    dofs = np.empty(12, dtype=int)
    dofs[0::3] = 3 * el
    dofs[1::3] = 3 * el + 1
    dofs[2::3] = 3 * el + 2
    return dofs


def _component_fns(vals):
    """Normalise a 3-component spec to callables f(x, y, z).

    Each component may be a float, a callable f(x, y, z), or None
    (treated as zero).
    """
    fns = []
    for v in vals:
        if v is None:
            fns.append(lambda x, y, z: 0.0)
        elif callable(v):
            fns.append(v)
        else:
            fns.append(lambda x, y, z, v=v: v)
    return fns


def assemble_elasticity_3d(mesh, E, nu, body=None, traction=None,
                           T=None, alpha=0.0, T_ref=0.0):
    """Assemble the 3N x 3N stiffness matrix and load vector.

    Parameters
    ----------
    mesh : TetMesh
    E, nu : Young's modulus, Poisson's ratio (floats).
    body : None, (bx, by, bz) floats, or tuple of 3 callables f(x, y, z).
        Evaluated at the element centroid, constant per element; each
        node receives b * V / 4.
    traction : None or dict mapping boundary name -> (tx, ty, tz).
        Each component may be a float, a callable f(x, y, z), or None.
        Integrated on the faces from boundary_faces, constant per face
        (centroid value); each face node receives t * A_face / 3.
    T : None, nodal temperature array, or callable f(x, y, z).
    alpha : thermal expansion coefficient (1/K).
    T_ref : stress-free reference temperature.

    Returns
    -------
    A : csr_matrix (3N, 3N), rhs : (3N,) array,
        dof order ux0, uy0, uz0, ux1, ...
    """
    D = constitutive_3d(E, nu)
    n3 = 3 * mesh.n_nodes
    rows, cols, data = [], [], []
    rhs = np.zeros(n3)

    if body is None:
        b_fns = [lambda x, y, z: 0.0] * 3
    else:
        b_fns = _component_fns(body)

    T_is_callable = callable(T)
    if T is not None and not T_is_callable:
        T = np.asarray(T, dtype=float)

    eye3 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # Voigt identity

    for el in mesh.elements:
        p = mesh.nodes[el]
        G, vol = _tet_gradients(p)
        B = _strain_displacement(G)
        ke = (B.T @ D @ B) * vol
        dofs = _dof_map(el)
        for i in range(12):
            for j in range(12):
                rows.append(dofs[i])
                cols.append(dofs[j])
                data.append(ke[i, j])

        # body force, constant per element evaluated at the centroid
        xc, yc, zc = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()
        bv = [fn(xc, yc, zc) for fn in b_fns]
        for k in range(4):
            for c in range(3):
                rhs[dofs[3 * k + c]] += bv[c] * vol / 4.0

        # thermal load: int B^T D eps_th dV, eps_th constant per element
        if T is not None and alpha != 0.0:
            Tc = T(xc, yc, zc) if T_is_callable else T[el].mean()
            eth = alpha * (Tc - T_ref) * eye3
            fth = (B.T @ D @ eth) * vol
            rhs[dofs] += fth

    # traction loads on boundary faces, constant per face (centroid)
    if traction:
        for name, vals in traction.items():
            t_fns = _component_fns(vals)
            faces = boundary_faces(mesh, name)
            for tri in faces:
                p = mesh.nodes[tri]
                xc, yc, zc = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()
                tv = [fn(xc, yc, zc) for fn in t_fns]
                # face area from the cross product of two edges
                cr = np.cross(p[1] - p[0], p[2] - p[0])
                area = 0.5 * float(np.linalg.norm(cr))
                for node in tri:
                    node = int(node)
                    for c in range(3):
                        rhs[3 * node + c] += tv[c] * area / 3.0

    A = sp.coo_matrix((data, (rows, cols)), shape=(n3, n3)).tocsr()
    return A, rhs


def apply_dirichlet_vec3(A, rhs, mesh, bc):
    """Apply displacement Dirichlet BCs to the 3D vector system.

    bc : dict mapping boundary name (or 'all') -> (ux, uy, uz). Each
         component may be a float, a callable f(x, y, z), or None
         (leave free).

    Returns the full (3N,) solution vector with constrained DOFs filled
    from the BC values.
    """
    rhs = np.asarray(rhs, dtype=float).copy()
    n3 = 3 * mesh.n_nodes
    fixed = {}

    for name, vals in bc.items():
        if name == "all":
            ids = np.arange(mesh.n_nodes)
        else:
            ids = np.asarray(mesh.boundaries[name])
        for comp, v in enumerate(vals):
            if v is None:
                continue
            v_fn = v if callable(v) else (lambda x, y, z, v=v: v)
            for i in ids:
                i = int(i)
                x, y, z = mesh.nodes[i]
                fixed[3 * i + comp] = v_fn(x, y, z)

    u = np.zeros(n3)
    fixed_idx = np.array(sorted(fixed), dtype=int)
    if len(fixed_idx):
        v = np.array([fixed[i] for i in fixed_idx])
        u[fixed_idx] = v
        free = np.ones(n3, dtype=bool)
        free[fixed_idx] = False
        A = A.tocsr()
        rhs_free = rhs[free] - A[free][:, fixed_idx] @ v
        A_free = A[free][:, free].tocsr()
        u[free] = spla.spsolve(A_free, rhs_free)
    else:
        u = np.asarray(spla.spsolve(A.tocsr(), rhs)).ravel()
    return u


def solve_elasticity_3d(mesh, E, nu, bc=None, body=None, traction=None,
                        T=None, alpha=0.0, T_ref=0.0):
    """Solve the 3D linear elasticity problem.

    Returns (ux, uy, uz), the nodal displacement components.
    """
    A, rhs = assemble_elasticity_3d(mesh, E, nu, body=body,
                                    traction=traction,
                                    T=T, alpha=alpha, T_ref=T_ref)
    if bc:
        u = apply_dirichlet_vec3(A, rhs, mesh, bc)
    else:
        u = np.asarray(spla.spsolve(A.tocsr(), rhs)).ravel()
    return u[0::3].copy(), u[1::3].copy(), u[2::3].copy()


def element_stress_strain_3d(mesh, u, E, nu, T=None, alpha=0.0, T_ref=0.0):
    """Per-element strain and stress in Voigt notation [xx, yy, zz, yz, xz, xy].

    u : (3N,) displacement vector, or (ux, uy, uz) tuple of nodal arrays.
    T : None, nodal temperature array, or callable f(x, y, z).
    Returns (sig, eps), each (M, 6).
    """
    if isinstance(u, tuple):
        ux, uy, uz = u
        uv = np.empty(3 * mesh.n_nodes)
        uv[0::3] = ux
        uv[1::3] = uy
        uv[2::3] = uz
    else:
        uv = np.asarray(u, dtype=float).ravel()

    D = constitutive_3d(E, nu)
    T_is_callable = callable(T)
    if T is not None and not T_is_callable:
        T = np.asarray(T, dtype=float)
    eye3 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    sig = np.zeros((mesh.n_elements, 6))
    eps = np.zeros((mesh.n_elements, 6))
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        G, _ = _tet_gradients(p)
        B = _strain_displacement(G)
        dofs = _dof_map(el)
        ee = B @ uv[dofs]
        eps[e] = ee
        eth = np.zeros(6)
        if T is not None and alpha != 0.0:
            xc, yc, zc = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()
            Tc = T(xc, yc, zc) if T_is_callable else T[el].mean()
            eth = alpha * (Tc - T_ref) * eye3
        sig[e] = D @ (ee - eth)
    return sig, eps


def von_mises_3d(sig):
    """Von Mises equivalent stress for Voigt stress [xx, yy, zz, yz, xz, xy].

    Accepts (M, 6) or (6,) arrays.
    """
    s = np.asarray(sig, dtype=float)
    sxx, syy, szz = s[..., 0], s[..., 1], s[..., 2]
    tyz, txz, txy = s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                          + (szz - sxx) ** 2)
                   + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2))
