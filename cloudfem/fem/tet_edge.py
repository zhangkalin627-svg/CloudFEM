"""Lowest-order Nedelec (Whitney) edge elements on tetrahedra (v1.3).

Each tetrahedron has 6 edges; the global edge set is the union over
all elements, an edge being an unordered node pair stored in
increasing node order. The single degree of freedom on edge
``e = (n0, n1)`` (``n0 < n1``) is the circulation

    a_e = int_e A . dl          (from n0 to n1),

and the associated Whitney 1-form on a tetrahedron, for the local
edge ``(i, j)``, is

    N_ij = lam_i grad(lam_j) - lam_j grad(lam_i),

with lam the P1 barycentric coordinates. Its tangential component
integrates to exactly 1 along its own edge (in the i -> j direction)
and to 0 along the other five. ``curl N_ij = 2 grad(lam_i) x
grad(lam_j)`` is constant per tetrahedron, so the curl-curl matrix
is exact, and the mass-type integrals have the closed forms used
below (from int lam_p lam_q dV = V/20 * (1 + delta_pq)):

    int N_m . N_n dV = (V/20) [ (1+d_ik) g_j.g_l - (1+d_il) g_j.g_k
                                - (1+d_jk) g_i.g_l + (1+d_jl) g_i.g_k ]
    int N_m dV       = (V/4) (g_j - g_i)
    int grad(lam_p) . N_m dV = (V/4) g_p . (g_j - g_i)

for local edges m = (i, j), n = (k, l) and g_p = grad(lam_p).
Local edge orientation is ``i -> j`` with ``i < j`` in *local* node
numbering; :func:`build_edges` also returns, per element and local
edge, the sign that maps the local orientation onto the global one,
so element matrices computed in local orientation scatter into the
global system with a sign product.
"""

import numpy as np

from .tet_poisson import _tet_gradients

# local edges of a tetrahedron, in local node numbering (i < j)
LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def build_edges(mesh):
    """Enumerate the global edge set of a TetMesh.

    Returns
    -------
    edges : (E, 2) int array, node pairs with edges[:, 0] < edges[:, 1].
    elem_edges : (M, 6) int array, global edge id per local edge.
    elem_signs : (M, 6) float array, +1 where the local orientation
        (local node i -> j, i < j locally) agrees with the global one
        (smaller global node id -> larger), else -1.
    """
    elems = mesh.elements
    key_to_id = {}
    edges = []
    elem_edges = np.zeros((mesh.n_elements, 6), dtype=int)
    elem_signs = np.ones((mesh.n_elements, 6))
    for e, el in enumerate(elems):
        for m, (i, j) in enumerate(LOCAL_EDGES):
            a, b = int(el[i]), int(el[j])
            key = (a, b) if a < b else (b, a)
            gid = key_to_id.get(key)
            if gid is None:
                gid = len(edges)
                key_to_id[key] = gid
                edges.append(key)
            elem_edges[e, m] = gid
            elem_signs[e, m] = 1.0 if a < b else -1.0
    return np.array(edges, dtype=int), elem_edges, elem_signs


def element_edge_data(p):
    """Whitney-element data for one tetrahedron.

    p : (4, 3) node coordinates. Returns a dict with

    grads : (4, 3) P1 gradients g_p = grad(lam_p),
    vol : tetrahedron volume,
    curl : (6, 3) curl of each local Whitney form (constant),
    mass : (6, 6) int N_m . N_n dV in local orientation,
    int_n : (6, 3) int N_m dV in local orientation,
    grad_n : (6, 4) int grad(lam_p) . N_m dV (rows m, columns p).
    """
    g, vol = _tet_gradients(p)
    curl = np.zeros((6, 3))
    mass = np.zeros((6, 6))
    int_n = np.zeros((6, 3))
    grad_n = np.zeros((6, 4))
    for m, (i, j) in enumerate(LOCAL_EDGES):
        curl[m] = 2.0 * np.cross(g[i], g[j])
        int_n[m] = 0.25 * vol * (g[j] - g[i])
        for pidx in range(4):
            grad_n[m, pidx] = g[pidx] @ int_n[m]
        for n, (k, l) in enumerate(LOCAL_EDGES):
            mass[m, n] = (vol / 20.0) * (
                (1.0 + (i == k)) * (g[j] @ g[l])
                - (1.0 + (i == l)) * (g[j] @ g[k])
                - (1.0 + (j == k)) * (g[i] @ g[l])
                + (1.0 + (j == l)) * (g[i] @ g[k]))
    return {"grads": g, "vol": vol, "curl": curl, "mass": mass,
            "int_n": int_n, "grad_n": grad_n}
