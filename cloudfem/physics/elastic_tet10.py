"""Linear elasticity in 3D with Tet10 (quadratic) elements.

The P2 counterpart of :mod:`cloudfem.physics.elastic3d`: same
problem (small-strain isotropic linear elasticity with optional
thermal strain), same Voigt ordering [xx, yy, zz, yz, xz, xy], same
DOF layout (3 DOFs per node, interleaved), and the same
``bc``/``traction``/``body`` conventions, but with 10-node
tetrahedra and 30 DOFs per element.

Element integrals use the 5-point Keast rule of
:mod:`cloudfem.fem.tet10` (degree-3 exact). The stiffness integrand
``B^T D B`` is degree 2 on affine elements, so it is integrated
exactly for constant material data; body forces and thermal loads
are likewise sampled at the quadrature points.

Traction loads act on 6-node triangular boundary faces (the Tet4
face triple plus its three edge midpoints) and are integrated with
the 3-point degree-2 triangle rule -- exact for constant traction.

Motivation: P1 constant-strain tetrahedra are notoriously overstiff
in bending (parasitic shear); the quadratic element removes most of
that error at equal mesh size (see tests/test_tet10.py, where the
BIG-M6 cantilever geometry is re-solved with both element types).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fem.tet10 import element_operators
from ..mesh.tet10_mesh import EDGES
from .elastic3d import constitutive_3d, apply_dirichlet_vec3

__all__ = [
    "boundary_faces10", "assemble_elasticity_tet10",
    "solve_elasticity_tet10", "element_stress_tet10",
]

# Tet faces as (corner triple, midpoint triple); the midpoints are
# the edges (c0, c1), (c1, c2), (c2, c0) of the corner triple, in the
# local slot order used by 6-node triangle shape functions.
FACES10 = (
    ((0, 1, 2), (4, 5, 6)),
    ((0, 3, 1), (7, 8, 4)),
    ((0, 2, 3), (6, 9, 7)),
    ((1, 3, 2), (8, 9, 5)),
)

# 3-point degree-2 rule on a triangle (barycentric, weights sum to 1)
_TRI_L = np.array([[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                   [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                   [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
_TRI_W = np.full(3, 1.0 / 3.0)
# face-local edges (corner pairs) in midpoint order
_FACE_EDGES = ((0, 1), (1, 2), (2, 0))


def _face_shape(l):
    """6-node triangle shape functions at barycentric points l (Q, 3)."""
    N = np.empty(l.shape[:-1] + (6,))
    for i in range(3):
        N[..., i] = l[..., i] * (2.0 * l[..., i] - 1.0)
    for k, (i, j) in enumerate(_FACE_EDGES):
        N[..., 3 + k] = 4.0 * l[..., i] * l[..., j]
    return N


def boundary_faces10(mesh, name):
    """Exterior 6-node faces on the named boundary of a Tet10Mesh.

    Returns (F, 6) int array: [c0, c1, c2, m01, m12, m20] per face --
    faces whose corner triple belongs to exactly one tetrahedron and
    whose 3 corners all lie in mesh.boundaries[name] (the criterion
    of elastic3d.boundary_faces, lifted to Tet10).
    """
    ids = np.asarray(mesh.boundaries[name])
    idset = set(int(i) for i in ids)
    counts = {}
    faces = {}
    for el in mesh.elements:
        for corners, mids in FACES10:
            tri = tuple(int(el[c]) for c in corners)
            key = tuple(sorted(tri))
            counts[key] = counts.get(key, 0) + 1
            if key not in faces:
                faces[key] = tuple(int(el[c]) for c in corners + mids)
    out = [faces[key] for key, cnt in counts.items()
           if cnt == 1 and all(n in idset for n in key)]
    if not out:
        return np.zeros((0, 6), dtype=int)
    return np.array(out, dtype=int)


def _strain_displacement10(G):
    """Strain-displacement matrix B (6, 30) from gradients G (10, 3)."""
    B = np.zeros((6, 30))
    for k in range(10):
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


def _dof_map10(el):
    """Interleaved DOF indices for a node list (element or face)."""
    el = np.asarray(el, dtype=int)
    dofs = np.empty(3 * len(el), dtype=int)
    dofs[0::3] = 3 * el
    dofs[1::3] = 3 * el + 1
    dofs[2::3] = 3 * el + 2
    return dofs


def _component_fns(vals):
    """Normalise a 3-component spec to callables f(x, y, z)."""
    fns = []
    for v in vals:
        if v is None:
            fns.append(lambda x, y, z: 0.0)
        elif callable(v):
            fns.append(v)
        else:
            fns.append(lambda x, y, z, v=v: v)
    return fns


def assemble_elasticity_tet10(mesh, E, nu, body=None, traction=None,
                              T=None, alpha=0.0, T_ref=0.0):
    """Assemble the Tet10 elasticity system (3N x 3N) and load vector.

    Parameters mirror assemble_elasticity_3d, except that body-force
    components and the thermal field are sampled at the element
    quadrature points instead of the centroid (T may be a nodal array
    over all Tet10 nodes, interpolated by the shape functions, or a
    callable(x, y, z)).

    Returns (A csr_matrix, rhs (3N,)), DOF order ux0, uy0, uz0, ...
    """
    from cloudfem.fem.tet10 import quadrature
    _, wq = quadrature()
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
    eye3 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    for el in mesh.elements:
        G, N, xq, vol = element_operators(mesh, el)
        dofs = _dof_map10(el)
        B = np.stack([_strain_displacement10(G[q])
                      for q in range(G.shape[0])])          # (Q, 6, 30)
        DB = np.einsum("ab,qbi->qai", D, B)
        Ke = vol * np.einsum("q,qai,qaj->ij", wq, B, DB)
        rows.append(np.repeat(dofs, 30))
        cols.append(np.tile(dofs, 30))
        data.append(Ke.ravel())

        # body force: int N_a * b_c dV, b sampled at quadrature points
        bv = np.array([[fn(x, y, z) for fn in b_fns]
                       for x, y, z in xq])                  # (Q, 3)
        fe = vol * np.einsum("q,qa,qc->ac", wq, N, bv)      # (10, 3)
        rhs[dofs] += fe.ravel()

        # thermal load: int B^T D eps_th dV, eps_th from T at the qps
        if T is not None and alpha != 0.0:
            if T_is_callable:
                Tq = np.array([T(x, y, z) for x, y, z in xq])
            else:
                Tq = N @ T[el]
            eth = alpha * (Tq - T_ref)[:, None] * eye3[None, :]  # (Q, 6)
            Deth = eth @ D.T
            fth = vol * np.einsum("q,qai,qa->i", wq, B, Deth)
            rhs[dofs] += fth

    # traction loads on 6-node boundary faces
    if traction:
        Nf = _face_shape(_TRI_L)                            # (3, 6)
        for name, vals in traction.items():
            t_fns = _component_fns(vals)
            for face in boundary_faces10(mesh, name):
                pc = mesh.nodes[face[:3]]                    # corners
                cr = np.cross(pc[1] - pc[0], pc[2] - pc[0])
                area = 0.5 * float(np.linalg.norm(cr))
                xq = _TRI_L @ pc                             # (3, 3)
                tv = np.array([[fn(x, y, z) for fn in t_fns]
                               for x, y, z in xq])           # (3, 3)
                fe = area * np.einsum("q,qa,qc->ac",
                                      _TRI_W, Nf, tv)        # (6, 3)
                rhs[_dof_map10(face)] += fe.ravel()

    A = sp.coo_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n3, n3)).tocsr()
    return A, rhs


def solve_elasticity_tet10(mesh, E, nu, bc=None, body=None,
                           traction=None, T=None, alpha=0.0, T_ref=0.0):
    """Solve the 3D linear elasticity problem with Tet10 elements.

    bc : dict as in elastic3d.apply_dirichlet_vec3 (boundary name ->
    (ux, uy, uz), components float / callable / None).
    Returns (ux, uy, uz), nodal arrays over all Tet10 nodes.
    """
    A, rhs = assemble_elasticity_tet10(mesh, E, nu, body=body,
                                       traction=traction, T=T,
                                       alpha=alpha, T_ref=T_ref)
    if bc:
        u = apply_dirichlet_vec3(A, rhs, mesh, bc)
    else:
        u = np.asarray(spla.spsolve(A.tocsr(), rhs)).ravel()
    return u[0::3].copy(), u[1::3].copy(), u[2::3].copy()


def element_stress_tet10(mesh, u, E, nu, T=None, alpha=0.0, T_ref=0.0):
    """Per-element strain/stress at the element centroid (Voigt 6).

    u : (3N,) displacement vector or (ux, uy, uz) tuple.
    Returns (sig, eps), each (M, 6). The centroid value of a Tet10
    element is the mean of the linear strain field, a second-order
    representative value (not a constant-strain assumption).
    """
    from cloudfem.fem.tet10 import shape_functions
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
    Lc = np.array([[0.25, 0.25, 0.25, 0.25]])

    sig = np.zeros((mesh.n_elements, 6))
    eps = np.zeros((mesh.n_elements, 6))
    Nc, dN_dL = shape_functions(Lc)
    dN_dxi = np.stack([dN_dL[0, :, 1] - dN_dL[0, :, 0],
                       dN_dL[0, :, 2] - dN_dL[0, :, 0],
                       dN_dL[0, :, 3] - dN_dL[0, :, 0]], axis=-1)  # (10, 3)
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el[:4]]
        J = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)
        Gc = dN_dxi @ np.linalg.inv(J)                       # (10, 3)
        B = _strain_displacement10(Gc)
        dofs = _dof_map10(el)
        ee = B @ uv[dofs]
        eps[e] = ee
        eth = np.zeros(6)
        if T is not None and alpha != 0.0:
            if T_is_callable:
                Tc = T(*p.mean(axis=0))
            else:
                Tc = float(Nc[0] @ T[el])
            eth = alpha * (Tc - T_ref) * eye3
        sig[e] = D @ (ee - eth)
    return sig, eps
