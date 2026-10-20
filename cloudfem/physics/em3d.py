"""3D time-harmonic eddy currents: edge-element A-V formulation (v1.3).

This is the 3D counterpart of the 2D A-formulation in
:mod:`cloudfem.physics.em` (same e^{i*omega*t} phasor convention,
same material callables, same Joule-heating output convention
``Q = 0.5 * sigma * |E|^2`` per element, so the v1.0 transient heat
chain consumes the result unchanged).

Discretisation: lowest-order Nedelec (Whitney) edge elements for the
magnetic vector potential A (one circulation DOF per mesh edge, see
:mod:`cloudfem.fem.tet_edge`), plus nodal P1 elements for the
electric scalar potential V in conducting regions only. With
``E = -i*omega*A - grad(V)`` and ``J = sigma * E`` the system is

    curl(nu curl A) + sigma * (i*omega*A + grad V) = J_s   (Ampere)
    div( sigma * (i*omega*A + grad V) ) = 0                 (continuity)

i.e. the block system, with K the edge curl-curl matrix, M_sigma the
sigma-weighted edge mass matrix and G the coupling
``G[m, p] = int sigma * N_m . grad(lam_p) dV``,

    [ K + i*omega*M_sigma ,  G     ] [a]   [rhs]
    [ i*omega*G^T           ,  S   ] [v] = [ 0 ]

where S is the sigma-weighted P1 stiffness on conductor nodes. V is
defined up to one constant per connected conducting component; one
node per component is pinned to V = 0.

Gauge treatment in non-conducting regions: the curl-curl operator
annihilates gradients, so without a gauge the edge system is
singular wherever sigma = 0. Whitney 1-forms are elementwise
divergence-free, so an elementwise grad-div penalty is identically
zero and cannot be used. Instead the air region is regularised with
a small artificial conductivity ``sigma_reg = gauge_eps * sigma_max``
(default ``gauge_eps = 1e-6``) in the *mass term only*: it damps
exactly the gradient (null-space) component of A with weight
``omega * sigma_reg`` -- the discrete analogue of a grad-div penalty
-- while the physical current, field and loss outputs always use the
true sigma (zero in air). Sensitivity to ``gauge_eps`` is a measured
validation item (tests/test_em3d.py), not an assumption.

Excitation: either a uniform applied (background) field
``B_applied`` [T], imposed through tangential-A Dirichlet data on
the outer boundary edges from the analytic potential
``A0 = (1/2) B x r`` (so ``curl A0 = B``; the far boundary is the
mesh boundary itself, i.e. the applied field is enforced at a finite
distance -- the body's dipole perturbation there decays like
(a/L)^3 and is a documented error source), or an imposed source
current density ``J_s`` (callable (x, y, z) -> (Jx, Jy, Jz),
integrated against the Whitney forms with the centroid value;
tangential A then vanishes on the outer boundary).

Limitations: linear materials only (constant nu per element);
single-frequency phasor solve with a sparse direct solver, so edge
counts should stay in the tens of thousands on a small machine; at
very low frequency the regularisation weakens with omega and the
direct solve loses accuracy (the A-V coupling itself is the standard
cure for low-frequency breakdown of a pure A formulation, but
omega -> 0 magnetostatics is out of scope).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fem.tet_edge import build_edges, element_edge_data, LOCAL_EDGES
from .em import MU0


def _as_fn3(q):
    return q if callable(q) else (lambda x, y, z, q=q: q)


def _centroids(mesh):
    return mesh.nodes[mesh.elements].mean(axis=1)


def _element_values(mesh, q):
    """Evaluate a float-or-callable(x, y, z) spec at element centroids."""
    fn = _as_fn3(q)
    cen = _centroids(mesh)
    return np.array([fn(x, y, z) for x, y, z in cen], dtype=float)


def _boundary_nodes(mesh, boundary):
    """Resolve the `boundary` spec to a node-index array.

    None -> 'outer' tag if present, else 'boundary', else the union
    of all named sets. A string names one set, a list of strings is
    unioned, an array is used as-is.
    """
    if boundary is None:
        if "outer" in mesh.boundaries:
            return np.asarray(mesh.boundaries["outer"], dtype=int)
        if "boundary" in mesh.boundaries:
            return np.asarray(mesh.boundaries["boundary"], dtype=int)
        sets = [np.asarray(v, dtype=int) for v in mesh.boundaries.values()]
        return np.unique(np.concatenate(sets)) if sets else np.array([], int)
    if isinstance(boundary, str):
        return np.asarray(mesh.boundaries[boundary], dtype=int)
    if isinstance(boundary, (list, tuple)) and boundary \
            and isinstance(boundary[0], str):
        return np.unique(np.concatenate(
            [np.asarray(mesh.boundaries[k], dtype=int) for k in boundary]))
    return np.asarray(boundary, dtype=int)


def _eliminate(A, rhs, fixed):
    """Dirichlet elimination for {index: value} on a sparse system.

    Always returns (A_free, rhs_free, keep, fixed_idx, fixed_val).
    """
    if not fixed:
        n = A.shape[0]
        return A, rhs, np.ones(n, dtype=bool), np.array([], int), \
            np.array([], dtype=complex)
    idx = np.array(sorted(fixed), dtype=int)
    val = np.array([fixed[i] for i in idx], dtype=complex)
    rhs = rhs - A[:, idx] @ val
    keep = np.ones(A.shape[0], dtype=bool)
    keep[idx] = False
    A = A.tocsr()
    A = A[keep][:, keep]
    rhs = rhs[keep]
    return A, rhs, keep, idx, val


def solve_eddy_current_3d(mesh, nu=1.0 / MU0, sigma=0.0, omega=0.0,
                          B_applied=None, J_s=None, boundary=None,
                          gauge_eps=1e-6, edge_data=None):
    """Solve the 3D A-V eddy-current problem on a TetMesh.

    Parameters
    ----------
    mesh : TetMesh covering conductor and air regions.
    nu : float or callable(x, y, z), reluctivity 1/mu (centroid value).
    sigma : float or callable(x, y, z), conductivity [S/m]; elements
        with sigma > 0 at the centroid are the conducting region.
    omega : angular frequency [rad/s] (> 0).
    B_applied : (3,) uniform applied flux density [T] for the
        background-field excitation, or None.
    J_s : callable(x, y, z) -> (3,) imposed current density [A/m^2],
        or None. At most one of B_applied / J_s is normally used;
        with J_s the outer-boundary tangential A is set to zero.
    boundary : outer-boundary node spec (see _boundary_nodes).
    gauge_eps : air-region regularisation ratio (module docstring).
    edge_data : optional (edges, elem_edges, elem_signs) from
        cloudfem.fem.tet_edge.build_edges, to avoid recomputation.

    Returns
    -------
    a : (E,) complex edge circulations.
    v : (N,) complex nodal electric potential (zero outside the
        conducting region and at the pinned node of each component).
    """
    if omega <= 0:
        raise ValueError("omega must be positive (phasor eddy-current "
                         "solve; omega = 0 magnetostatics is out of scope)")
    if B_applied is None and J_s is None:
        raise ValueError("no excitation: pass B_applied (background "
                         "field) or J_s (imposed current density)")
    if edge_data is None:
        edge_data = build_edges(mesh)
    edges, elem_edges, elem_signs = edge_data
    n_edge = len(edges)

    nu_e = _element_values(mesh, nu)
    sig_e = _element_values(mesh, sigma)
    conducting = sig_e > 0.0
    if not conducting.any():
        raise ValueError("no conducting elements (sigma > 0) in mesh")
    sig_reg = gauge_eps * sig_e.max()
    sig_eff = np.where(conducting, sig_e, sig_reg)

    # --- conductor nodes, connected components, V pinning ----------
    cond_nodes = np.unique(mesh.elements[conducting])
    parent = {int(n): int(n) for n in cond_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for el in mesh.elements[conducting]:
        ids = [int(i) for i in el]
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb:
                parent[rb] = ra
    comps = {}
    for n in cond_nodes:
        comps.setdefault(find(int(n)), []).append(int(n))
    pinned = {min(members) for members in comps.values()}
    v_nodes = np.array([n for n in cond_nodes if n not in pinned],
                       dtype=int)
    v_col = {int(n): j for j, n in enumerate(v_nodes)}
    n_v = len(v_nodes)

    # --- assembly ---------------------------------------------------
    rows, cols, data = [], [], []

    def add(i, j, val):
        rows.append(i)
        cols.append(j)
        data.append(val)

    rhs = np.zeros(n_edge + n_v, dtype=complex)
    Js_fn = J_s if callable(J_s) else None
    cen = _centroids(mesh)
    vols = mesh.element_volumes()
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        d = element_edge_data(p)
        g = elem_edges[e]
        s = elem_signs[e]
        ke = nu_e[e] * vols[e] * (d["curl"] @ d["curl"].T)
        me = sig_eff[e] * d["mass"]
        blk = ke + 1j * omega * me
        for m in range(6):
            gm = int(g[m])
            sm = s[m]
            for n in range(6):
                add(gm, int(g[n]), sm * s[n] * blk[m, n])
            if Js_fn is not None:
                jv = np.asarray(Js_fn(*cen[e]), dtype=float)
                rhs[gm] += sm * float(jv @ d["int_n"][m])
        if conducting[e]:
            se = sig_e[e]
            for m in range(6):
                gm = int(g[m])
                sm = s[m]
                for pidx in range(4):
                    node = int(el[pidx])
                    if node not in v_col:
                        continue
                    vc = n_edge + v_col[node]
                    gval = sm * se * d["grad_n"][m, pidx]
                    add(gm, vc, gval)                    # G block
                    add(vc, gm, 1j * omega * gval)       # i*omega*G^T
            sgrad = d["grads"]
            selem = se * vols[e] * (sgrad @ sgrad.T)
            for pidx in range(4):
                np_ = int(el[pidx])
                if np_ not in v_col:
                    continue
                vp = n_edge + v_col[np_]
                for qidx in range(4):
                    nq = int(el[qidx])
                    if nq not in v_col:
                        continue
                    add(vp, n_edge + v_col[nq], selem[pidx, qidx])

    A = sp.coo_matrix((data, (rows, cols)),
                      shape=(n_edge + n_v, n_edge + n_v),
                      dtype=complex).tocsr()

    # --- Dirichlet data on outer-boundary edges ---------------------
    bnd_nodes = set(int(i) for i in _boundary_nodes(mesh, boundary))
    fixed = {}
    if B_applied is not None:
        Bv = np.asarray(B_applied, dtype=float)

        def a0_at(x):
            return 0.5 * np.cross(Bv, x)
    for gi, (n0, n1) in enumerate(edges):
        if int(n0) in bnd_nodes and int(n1) in bnd_nodes:
            if B_applied is None:
                fixed[gi] = 0.0
            else:
                p0, p1 = mesh.nodes[n0], mesh.nodes[n1]
                fixed[gi] = float(a0_at(0.5 * (p0 + p1)) @ (p1 - p0))
    A2, rhs2, keep, idx, val = _eliminate(A, rhs, fixed)
    try:
        sol = spla.spsolve(A2.tocsc(), rhs2)
    except RuntimeError as exc:
        raise RuntimeError(f"3D eddy-current solve failed: {exc}") from exc
    sol = np.asarray(sol).ravel()
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("3D eddy-current solve produced non-finite "
                           "values (singular system; check gauge_eps, "
                           "boundary spec and sigma)")
    full = np.zeros(n_edge + n_v, dtype=complex)
    full[keep] = sol
    if len(idx):
        full[idx] = val
    a = full[:n_edge]
    v = np.zeros(mesh.n_nodes, dtype=complex)
    v[v_nodes] = full[n_edge:]
    return a, v


def element_flux_density_3d(mesh, a, edge_data=None):
    """Element-wise flux density B = curl A [T], shape (M, 3), complex."""
    if edge_data is None:
        edge_data = build_edges(mesh)
    _, elem_edges, elem_signs = edge_data
    B = np.zeros((mesh.n_elements, 3), dtype=complex)
    for e, el in enumerate(mesh.elements):
        d = element_edge_data(mesh.nodes[el])
        B[e] = (elem_signs[e] * a[elem_edges[e]]) @ d["curl"]
    return B


def element_electric_field_3d(mesh, a, v, omega, edge_data=None):
    """Element electric field E = -i*omega*A - grad V at centroids.

    A is evaluated at the element centroid (the Whitney forms are
    linear, so the centroid value is also the element mean); grad V
    is the constant P1 gradient. Returns (M, 3) complex [V/m].
    """
    if edge_data is None:
        edge_data = build_edges(mesh)
    _, elem_edges, elem_signs = edge_data
    E = np.zeros((mesh.n_elements, 3), dtype=complex)
    for e, el in enumerate(mesh.elements):
        d = element_edge_data(mesh.nodes[el])
        a_loc = elem_signs[e] * a[elem_edges[e]]
        A_c = (a_loc @ d["int_n"]) / d["vol"]
        E[e] = -1j * omega * A_c - v[el] @ d["grads"]
    return E


def _element_E_nodal(d, a_loc, v_el):
    """Vector potential at the 4 nodes of one tetrahedron, and grad V.

    The Whitney interpolant is affine per element, so its nodal
    values determine it exactly: at node p the Whitney form of edge
    (i, j) equals delta_ip * grad(lam_j) - delta_jp * grad(lam_i).
    a_loc : (6,) local-orientation edge coefficients; v_el : (4,)
    nodal potentials. Returns (A_nod, grad_v): A_nod is (4, 3)
    complex, grad_v the constant (3,) P1 potential gradient, so the
    nodal electric fields are -i*omega*A_nod - grad_v.
    """
    A_nod = np.zeros((4, 3), dtype=complex)
    for m, (i, j) in enumerate(LOCAL_EDGES):
        A_nod[i] += a_loc[m] * d["grads"][j]
        A_nod[j] -= a_loc[m] * d["grads"][i]
    return A_nod, v_el @ d["grads"]


def joule_heating_density_3d(mesh, a, v, sigma, omega, edge_data=None):
    """Per-element time-averaged Joule heating Q = 0.5*sigma*|E|^2.

    Uses the *true* sigma (zero in air), matching the convention of
    physics.em.joule_heating_density. |E|^2 is integrated *exactly*
    for the affine Whitney field: with E_p the nodal values,
    int |E|^2 dV = (V/20) * (|sum_p E_p|^2 + sum_p |E_p|^2) per
    element (centroid evaluation of this convex integrand would
    underestimate the loss by O(h^2) -- measured ~10 % at h ~ R/2 in
    the low-frequency tests). Returns (Q_elem, volumes).
    """
    if edge_data is None:
        edge_data = build_edges(mesh)
    _, elem_edges, elem_signs = edge_data
    sig_e = _element_values(mesh, sigma)
    vols = mesh.element_volumes()
    Q = np.zeros(mesh.n_elements)
    for e, el in enumerate(mesh.elements):
        if sig_e[e] <= 0.0:
            continue
        d = element_edge_data(mesh.nodes[el])
        a_loc = elem_signs[e] * a[elem_edges[e]]
        A_nod, grad_v = _element_E_nodal(d, a_loc, v[el])
        E_nod = -1j * omega * A_nod - grad_v
        integ = (np.sum(np.abs(E_nod.sum(axis=0)) ** 2)
                 + np.sum(np.abs(E_nod) ** 2))
        Q[e] = 0.5 * sig_e[e] * integ / 20.0
    return Q, vols


def total_joule_power_3d(mesh, a, v, sigma, omega, edge_data=None):
    """Total time-averaged Joule power [W] (sum of Q * element volume)."""
    Q, vols = joule_heating_density_3d(mesh, a, v, sigma, omega,
                                       edge_data=edge_data)
    return float(np.sum(Q * vols))
