"""3D time-harmonic eddy currents: Tet10 nodal A-V formulation (v1.7).

Nodal counterpart of the edge-element A-V formulation in
:mod:`cloudfem.physics.em3d`: the magnetic vector potential A is
carried by quadratic (10-node) Lagrange elements (three complex
components per mesh node) and the electric scalar potential V by
the same scalar basis on conducting nodes only. Same e^{i*omega*t}
phasor convention, same material callables, same Joule-heating
output convention ``Q = 0.5 * sigma * |E|^2`` per element, with
``E = -i*omega*A - grad(V)``.

Weak form (A vector test function W, scalar test function q)::

    int nu (curl A).(curl W) + int_Omc i*omega*sigma A.W
                              + int_Omc sigma grad(V).W = int J_s.W
    int_Omc i*omega*sigma A.grad(q) + int_Omc sigma grad(V).grad(q) = 0

i.e. the block system, with K_AA the nodal curl-curl matrix,
M_AA the sigma-weighted vector mass matrix and
``G[(p,m), j] = int sigma * N_p * d_m N_j dV``,

    [ K_AA + i*omega*M_AA ,  G       ] [a]   [rhs]
    [ i*omega*G^T          ,  S      ] [v] = [ 0 ]

where S is the sigma-weighted scalar stiffness on conductor nodes.
V is defined up to one constant per connected conducting
component; one node per component is pinned to V = 0.

Exactness of the element integrals (affine elements, 5-point Keast
rule of :mod:`cloudfem.fem.tet10`, degree-3 exact):
  * curl-curl integrand ``(curl(N_p e_m)).(curl(N_q e_m'))`` is
    degree 2 -- exact;
  * the A-V coupling ``N_p * d_m N_j`` is degree 3 -- exact;
  * the V stiffness ``grad N_i . grad N_j`` is degree 2 -- exact;
  * the sigma mass ``sigma * N_p * N_q`` is degree 4, integrated in
    closed form with the exact barycentric mass matrix
    ``fem.tet10._MASS_REF`` (exact for elementwise-constant sigma;
    spatially varying sigma is sampled at the element centroid,
    the P1-code convention).

Gauge treatment in non-conducting regions: identical to
:mod:`cloudfem.physics.em3d` -- a small artificial conductivity
``sigma_reg = gauge_eps * sigma_max`` (default ``gauge_eps = 1e-6``)
in the *mass term only* damps the gradient null-space component of
A; physical current, field and loss outputs always use the true
sigma (zero in air).

Dirichlet data: full nodal Dirichlet on the outer boundary --
``A = (1/2) B x r`` for a uniform applied field ``B_applied``
(``curl A0 = B``; fixing the normal component as well selects a
boundary gauge, harmless for the interior fields), or ``A = 0``
with an imposed source current ``J_s`` (the nodal analogue of the
edge code's vanishing tangential A). An explicit vector Dirichlet
map ``a_bc`` overrides these defaults on named boundary sets.

Limitations: linear materials only (constant nu / sigma per
element, centroid-sampled); single-frequency phasor solve with a
sparse direct solver -- Tet10 carries ~4 unknowns per node
(3*A + V), so meshes should stay at a few x10^4 nodes on a small
machine; omega = 0 magnetostatics is out of scope, as in em3d.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..fem.tet10 import (_MASS_REF, element_operators, quadrature,
                         shape_functions)
from ..mesh.tet10_mesh import EDGES
from .em import MU0
from .em3d import _boundary_nodes, _eliminate

__all__ = [
    "assemble_tet10_av", "solve_eddy_current_tet10",
    "element_flux_density_tet10", "total_joule_power_tet10",
    "source_complex_power_tet10", "transient_eddy_current_tet10",
    "transient_joule_power_tet10", "magnetic_energy_tet10",
]

# Levi-Civita tensor: (a x b)[k] = _EPS[k, l, m] a[l] b[m].
_EPS = np.zeros((3, 3, 3))
for _i, _j, _k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    _EPS[_i, _j, _k] = 1.0
for _i, _j, _k in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
    _EPS[_i, _j, _k] = -1.0
del _i, _j, _k

# Barycentric coordinates of the 10 Tet10 nodes (corners, then edge
# midpoints in EDGES order) -- for evaluating basis gradients at
# the nodes (the E-field power identity needs grad V at nodes).
_L_NODES = np.vstack([
    np.eye(4),
    np.array([[0.5, 0.5, 0.0, 0.0],
              [0.0, 0.5, 0.5, 0.0],
              [0.5, 0.0, 0.5, 0.0],
              [0.5, 0.0, 0.0, 0.5],
              [0.0, 0.5, 0.0, 0.5],
              [0.0, 0.0, 0.5, 0.5]]),
])


def _as_fn(q):
    return q if callable(q) else (lambda x, y, z, q=q: q)


def _element_values(mesh, q):
    """Evaluate a float-or-callable(x, y, z) spec at element centroids."""
    fn = _as_fn(q)
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    out = np.empty(mesh.n_elements, dtype=float)
    for e, (x, y, z) in enumerate(cen):
        out[e] = fn(x, y, z)
    return out


def _nodal_gradients(mesh, el):
    """Physical gradients of the 10 basis functions at the 10 nodes.

    Returns grads with grads[p, j, :] = grad(N_j)(x_p), shape
    (10, 10, 3). Used for the nodal E-field in the exact power
    identity (E is quadratic per element, hence fixed by its nodal
    values).
    """
    el = np.asarray(el, dtype=int)
    p = mesh.nodes[el[:4]]
    J = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)
    _, dN_dL = shape_functions(_L_NODES)          # (10, 10, 4)
    dN_dxi = np.stack([dN_dL[..., 1] - dN_dL[..., 0],
                       dN_dL[..., 2] - dN_dL[..., 0],
                       dN_dL[..., 3] - dN_dL[..., 0]], axis=-1)
    return dN_dxi @ np.linalg.inv(J)             # (10, 10, 3)


def _conductor_components(mesh, conducting):
    """Union-find connected conducting components; pin one node each.

    Returns (v_nodes, v_col): v_nodes the conductor node ids carrying
    a V unknown (one pinned node per component removed), v_col the
    node -> V-column map. Same convention as physics.em3d.
    """
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
    return v_nodes, v_col


def assemble_tet10_av(mesh, nu=1.0 / MU0, sigma=0.0, gauge_eps=1e-6):
    """Assemble the real blocks of the Tet10 nodal A-V system.

    Returns (K, M, G, S, v_nodes, sig_e, conducting):
      K : (3N, 3N) nodal curl-curl matrix, node-major dof 3*p+m;
      M : (3N, 3N) sigma-weighted vector mass (air entries carry
          the gauge conductivity gauge_eps * sigma_max);
      G : (3N, Nv) A-V coupling on conductor elements;
      S : (Nv, Nv) sigma-weighted scalar stiffness (conductor V
          nodes only, one pinned per component);
      v_nodes : conductor node ids owning the Nv V unknowns;
      sig_e : per-element true conductivity (centroid value);
      conducting : boolean per-element mask (sig_e > 0).
    nu / sigma : floats or callables(x, y, z), centroid-sampled.
    """
    n = mesh.n_nodes
    nu_e = _element_values(mesh, nu)
    sig_e = _element_values(mesh, sigma)
    conducting = sig_e > 0.0
    sig_max = sig_e.max() if conducting.any() else 0.0
    sig_reg = gauge_eps * sig_max

    v_nodes, v_col = _conductor_components(mesh, conducting)
    n_v = len(v_nodes)

    _, wq = quadrature()
    k_rows, k_cols, k_data = [], [], []
    m_rows, m_cols, m_data = [], [], []
    g_rows, g_cols, g_data = [], [], []
    s_rows, s_cols, s_data = [], [], []

    for e, el in enumerate(mesh.elements):
        el = np.asarray(el, dtype=int)
        G, N, xq, vol = element_operators(mesh, el)
        # curl basis C[q, p, m, k] = (grad N_p x e_m)[k]
        C = np.einsum("klm,qpl->qpmk", _EPS, G)
        ke = float(nu_e[e]) * vol * np.einsum("q,qpmk,qrnk->pmrn",
                                              wq, C, C)
        ke = ke.reshape(30, 30)
        dofs = np.array([[3 * int(el[p]) + m for m in range(3)]
                         for p in range(10)]).ravel()
        k_rows.append(np.repeat(dofs, 30))
        k_cols.append(np.tile(dofs, 30))
        k_data.append(ke.ravel())

        seff = float(sig_e[e]) if conducting[e] else sig_reg
        if seff > 0.0:
            me = seff * vol * _MASS_REF          # (10, 10) exact
            for m in range(3):
                dm = dofs[m::3]
                m_rows.append(np.repeat(dm, 10))
                m_cols.append(np.tile(dm, 10))
                m_data.append(me.ravel())

        if conducting[e]:
            se = float(sig_e[e])
            ge = se * vol * np.einsum("q,qp,qjm->pmj", wq, N, G)
            scol = np.array([v_col[int(el[j])] for j in range(10)
                             if int(el[j]) in v_col])
            smask = np.array([int(el[j]) in v_col for j in range(10)])
            if scol.size:
                ge = ge[:, :, smask]             # (10, 3, nvc)
                # dofs is (30,) C-order over (p, m); repeat each
                # (p, m) nvc times to match ge.ravel() ordering.
                rows = np.repeat(dofs, scol.size)
                cols = np.tile(scol, 30)
                g_rows.append(rows)
                g_cols.append(cols)
                # ge is (10, 3, nvc); C-order ravel matches rows =
                # repeat over (p, m) outer, j inner.
                g_data.append(ge.ravel())
            se_mat = se * vol * np.einsum("q,qil,qjl->ij", wq, G, G)
            ii = np.array([v_col[int(el[i])] for i in range(10)
                           if int(el[i]) in v_col])
            if ii.size:
                se_mat = se_mat[np.ix_(smask, smask)]
                s_rows.append(np.repeat(ii, ii.size))
                s_cols.append(np.tile(ii, ii.size))
                s_data.append(se_mat.ravel())

    def _coo(rows, cols, data, shape):
        if data:
            return sp.coo_matrix((np.concatenate(data),
                                  (np.concatenate(rows),
                                   np.concatenate(cols))),
                                 shape=shape).tocsr()
        return sp.csr_matrix(shape)

    K = _coo(k_rows, k_cols, k_data, (3 * n, 3 * n))
    M = _coo(m_rows, m_cols, m_data, (3 * n, 3 * n))
    Gs = _coo(g_rows, g_cols, g_data, (3 * n, n_v))
    S = _coo(s_rows, s_cols, s_data, (n_v, n_v))
    return K, M, Gs, S, v_nodes, sig_e, conducting


def solve_eddy_current_tet10(mesh, nu=1.0 / MU0, sigma=0.0, omega=0.0,
                             B_applied=None, J_s=None, boundary=None,
                             a_bc=None, gauge_eps=1e-6,
                             permc_spec=None):
    """Solve the 3D Tet10 nodal A-V eddy-current problem.

    Parameters
    ----------
    mesh : Tet10Mesh covering conductor and air regions.
    nu : float or callable(x, y, z), reluctivity 1/mu.
    sigma : float or callable(x, y, z), conductivity [S/m];
        elements with sigma > 0 at the centroid are conductors.
    omega : angular frequency [rad/s] (> 0).
    B_applied : (3,) uniform applied flux density [T], imposed via
        full nodal Dirichlet A = (1/2) B x r on the outer boundary.
    J_s : callable(x, y, z) -> (3,) imposed current density
        [A/m^2] (may be complex); outer-boundary A is then 0.
    boundary : outer-boundary node spec (see physics.em3d).
    a_bc : optional dict boundary-name -> (3,) constant or
        callable(x, y, z) -> (3,) vector Dirichlet data for A;
        entries override the B_applied / J_s default on those sets
        (used e.g. by manufactured-solution tests).
    gauge_eps : air-region regularisation ratio (module docstring).
    permc_spec : optional SuperLU column ordering passed to
        scipy.sparse.linalg.spsolve (e.g. "MMD_AT_PLUS_A"). The
        default (None, scipy's COLAMD) is unchanged; the ordering
        does not change results, it exists because factorisation
        memory/time of the A-V system is ordering-sensitive on
        small machines (the v1.3 edge-element precedent).

    Returns
    -------
    a : (3N,) complex nodal vector-potential components (node-major).
    v : (N,) complex nodal electric potential (zero outside the
        conducting region and at each component's pinned node).
    """
    if omega <= 0:
        raise ValueError("omega must be positive (phasor eddy-current "
                         "solve; omega = 0 magnetostatics is out of scope)")
    if B_applied is None and J_s is None:
        raise ValueError("no excitation: pass B_applied (background "
                         "field) or J_s (imposed current density)")

    K, M, Gs, S, v_nodes, sig_e, conducting = assemble_tet10_av(
        mesh, nu=nu, sigma=sigma, gauge_eps=gauge_eps)
    if not conducting.any():
        raise ValueError("no conducting elements (sigma > 0) in mesh")
    n = mesh.n_nodes
    n_v = len(v_nodes)

    A_sys = sp.bmat([[K + 1j * omega * M, Gs],
                     [1j * omega * Gs.T, S]], format="csr")

    rhs = np.zeros(3 * n + n_v, dtype=complex)
    if J_s is not None:
        _, wq = quadrature()
        for e, el in enumerate(mesh.elements):
            el = np.asarray(el, dtype=int)
            _, N, xq, vol = element_operators(mesh, el)
            Jq = np.array([np.asarray(J_s(x, y, z), dtype=complex)
                           for x, y, z in xq])          # (5, 3)
            fe = vol * np.einsum("q,qm,qp->pm", wq, Jq, N)  # (10, 3)
            for p in range(10):
                rhs[3 * int(el[p]):3 * int(el[p]) + 3] += fe[p]

    bnd = set(int(i) for i in _boundary_nodes(mesh, boundary))
    fixed = {}
    if B_applied is not None:
        Bv = np.asarray(B_applied, dtype=float)

        def a0_at(x):
            return 0.5 * np.cross(Bv, x)
    for p in bnd:
        x = mesh.nodes[p]
        val = a0_at(x) if B_applied is not None else np.zeros(3)
        for m in range(3):
            fixed[3 * p + m] = complex(val[m])
    if a_bc:
        for name, vals in a_bc.items():
            fn = (lambda x, y, z, v=vals: np.asarray(v, dtype=complex))
            if callable(vals):
                fn = vals
            for p in np.asarray(mesh.boundaries[name], dtype=int):
                vv = np.asarray(fn(*mesh.nodes[int(p)]),
                                dtype=complex).ravel()
                for m in range(3):
                    fixed[3 * int(p) + m] = vv[m]
    A2, rhs2, keep, idx, val = _eliminate(A_sys, rhs, fixed)
    try:
        if permc_spec is not None:
            sol = spla.spsolve(A2.tocsc(), rhs2, permc_spec=permc_spec)
        else:
            sol = spla.spsolve(A2.tocsc(), rhs2)
    except RuntimeError as exc:
        raise RuntimeError(f"Tet10 eddy-current solve failed: {exc}") \
            from exc
    sol = np.asarray(sol).ravel()
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("Tet10 eddy-current solve produced "
                           "non-finite values (singular system; check "
                           "gauge_eps, boundary spec and sigma)")
    full = np.zeros(3 * n + n_v, dtype=complex)
    full[keep] = sol
    if len(idx):
        full[idx] = val
    a = full[:3 * n]
    v = np.zeros(n, dtype=complex)
    v[v_nodes] = full[3 * n:]
    return a, v


def element_flux_density_tet10(mesh, a):
    """Element flux density B = curl A at element centroids [T].

    A is quadratic, so B is linear per element; the centroid value
    is reported. Returns (M, 3) complex.
    """
    a = np.asarray(a, dtype=complex).reshape(-1, 3)
    B = np.zeros((mesh.n_elements, 3), dtype=complex)
    for e, el in enumerate(mesh.elements):
        G, _, _, _ = element_operators(mesh, np.asarray(el, int))
        B[e] = np.einsum("klm,pl,pm->k", _EPS, G[0],
                         a[np.asarray(el, dtype=int)])
    return B


def total_joule_power_tet10(mesh, a, v, sigma, omega):
    """Total time-averaged Joule power [W], exact for the discrete E.

    E = -i*omega*A - grad(V) is quadratic per element, hence fixed
    by its nodal values E_p; with the exact mass matrix,
    ``int |E|^2 dV = vol * sum_{p,q} M_ref[p,q] Re(E_p.E_q*)`` --
    no quadrature error. Uses the *true* sigma (zero in air).
    Returns (P_total, Q_elem): Q_elem[e] is the mean heating density
    [W/m^3] (0 in air).
    """
    a = np.asarray(a, dtype=complex).reshape(-1, 3)
    v = np.asarray(v, dtype=complex)
    sig_e = _element_values(mesh, sigma)
    vols = mesh.element_volumes()
    Q = np.zeros(mesh.n_elements)
    for e, el in enumerate(mesh.elements):
        if sig_e[e] <= 0.0:
            continue
        el = np.asarray(el, dtype=int)
        grads = _nodal_gradients(mesh, el)        # (10, 10, 3)
        Ep = (-1j * omega * a[el]
              - np.einsum("pjx,j->px", grads, v[el]))
        re = np.real(np.einsum("pm,qm->pq", Ep, Ep.conj()))
        P_e = 0.5 * sig_e[e] * vols[e] * float(
            np.einsum("pq,pq->", _MASS_REF, re))
        Q[e] = P_e / vols[e]
    return float(np.sum(Q * vols)), Q


def source_complex_power_tet10(mesh, a, v, J_s, omega):
    """Complex power delivered by the imposed source current [VA].

    S = -0.5 * int J_s* . E dV with the discrete E field, integrated
    with the 5-point rule (exact when J_s is elementwise polynomial
    of degree <= 1).  The minus sign is the e^{iwt} Lorentz power
    (-E.J_s, power delivered *by* the source); with it, Re(S) is the
    Joule loss (== total_joule_power_tet10 up to quadrature) and
    Im(S) = 2*w*(W_m - W_e).  For a coil driven with total current
    phasor I, the terminal impedance is Z = 2*S / |I|^2.

    (v1.8 fix: v1.7 returned +0.5*int J_s.E* dV, whose real part is
    the *negative* Joule loss -- visible as a negative Re(Z) that
    the D1 test printed but never asserted.)
    """
    a = np.asarray(a, dtype=complex).reshape(-1, 3)
    v = np.asarray(v, dtype=complex)
    _, wq = quadrature()
    S = 0.0j
    for e, el in enumerate(mesh.elements):
        el = np.asarray(el, dtype=int)
        G, N, xq, vol = element_operators(mesh, el)
        Jq = np.array([np.asarray(J_s(x, y, z), dtype=complex)
                       for x, y, z in xq])                    # (5, 3)
        Eq = (-1j * omega * np.einsum("qp,pm->qm", N, a[el])
              - np.einsum("qjm,j->qm", G, v[el]))
        S += vol * np.einsum("q,qm,qm->", wq, Jq.conj(), Eq)
    return -0.5 * S


# ----------------------------------------------------------------------
# 3D transient eddy currents (theta method) -- v1.8
# ----------------------------------------------------------------------

_THETA_EM = {"be": 1.0, "implicit_euler": 1.0,
             "cn": 0.5, "crank_nicolson": 0.5}


def _rhs_Js_tet10(mesh, J_s):
    """Assemble the A-block load vector for a steady source current.

    J_s is (x, y, z) -> (3,). Returns a real (3N,) vector.
    """
    n = mesh.n_nodes
    rhs = np.zeros(3 * n)
    _, wq = quadrature()
    for e, el in enumerate(mesh.elements):
        el = np.asarray(el, dtype=int)
        _, N, xq, vol = element_operators(mesh, el)
        Jq = np.array([np.asarray(J_s(x, y, z), dtype=float).ravel()
                       for x, y, z in xq])                    # (5, 3)
        fe = vol * np.einsum("q,qm,qp->pm", wq, Jq, N)     # (10, 3)
        for p in range(10):
            rhs[3 * int(el[p]):3 * int(el[p]) + 3] += fe[p]
    return rhs


def transient_eddy_current_tet10(mesh, nu=1.0 / MU0, sigma=0.0,
                                 t_end=1.0, dt=0.1, method="cn",
                                 J_s=None, time_scale=None, B_applied=None,
                                 boundary=None, a_bc=None, gauge_eps=1e-6,
                                 history=False):
    """3D transient eddy currents on the Tet10 nodal A-V discretisation.

    Time-domain system (real unknowns x = [a; v])::

        [M_AA   0  ] [adot]   [K_AA  G] [a]   [rhs(t)]
        [G^T    0  ] [vdot] + [0     S] [v] = [  0   ]

    i.e. curl(nu curl A) + sigma*(dA/dt + grad V) = J_s(t) and
    div(sigma*(dA/dt + grad V)) = 0 in the conductor.  The theta
    step is applied monolithically; the system matrix
    ``E/h + theta*K`` is LU-factorised once.  The algebraic V row is
    enforced in theta-weighted form, consistent with the DAE.

    The source is separable: J(x, y, z, t) = time_scale(t) * J_s(x,y,z);
    the spatial pattern is assembled once and only scaled per step.

    Dirichlet data for A on the outer boundary: A = 0 with an
    imposed J_s (default), A = (1/2) B x r for a constant
    B_applied, or an explicit ``a_bc`` map (constant in time).
    J_s and B_applied may be combined (boundary takes the
    B_applied value, the source still drives); normally at most one
    excitation is used.  V keeps the phasor-solve pinning (one node
    per conducting component).

    Parameters
    ----------
    J_s : callable, optional. (x, y, z) -> (3,) imposed current
        density pattern [A/m^2].
    time_scale : callable, optional. t -> float multiplier; the
        effective source is time_scale(t) * J_s.  Default 1.
    t_end, dt : floats; the step used is t_end / ceil(t_end / dt).
    method : "be"/"implicit_euler" (theta = 1) or "cn"/"crank_nicolson"
        (theta = 1/2).
    history : if True, also return the A field at every step
        ("t_hist", "a_hist"); if a (t0, t1) tuple, record only steps
        with t0 <= t <= t1 (for last-period Fourier extraction
        without storing the whole run).

    Returns
    -------
    dict with times (n_steps+1,), P (instantaneous Joule power [W]
    per step, E = -dA/dt - grad V via the backward difference with
    the constraint-consistent V = -S^-1 G^T dA/dt; real field, no
    1/2 phasor averaging), a_end (3N,) and v_end (N,) real nodal
    vectors.  Note: the raw DAE unknown v is not returned -- CN is
    not stiffly accurate and would leave a +/- flip-flop on it; the
    a-sequence is unaffected (it equals CN on the underlying ODE
    exactly).
    """
    if method not in _THETA_EM:
        raise ValueError(f"unknown method {method!r}; use one of "
                         f"{sorted(_THETA_EM)}")
    theta = _THETA_EM[method]
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if J_s is None and B_applied is None:
        raise ValueError("no excitation: pass J_s and/or B_applied")

    K, M, Gs, S, v_nodes, sig_e, conducting = assemble_tet10_av(
        mesh, nu=nu, sigma=sigma, gauge_eps=gauge_eps)
    if not conducting.any():
        raise ValueError("no conducting elements (sigma > 0) in mesh")
    n = mesh.n_nodes
    n_v = len(v_nodes)
    nx = 3 * n + n_v

    Z_av = sp.csr_matrix((3 * n, n_v))
    Z_va = sp.csr_matrix((n_v, 3 * n))
    Z_vv = sp.csr_matrix((n_v, n_v))
    E = sp.bmat([[M, Z_av], [Gs.T, Z_vv]], format="csr")
    Kmat = sp.bmat([[K, Gs], [Z_va, S]], format="csr")
    n_steps = int(np.ceil(t_end / dt - 1e-12))
    h = t_end / n_steps

    A_lhs = (E / h + theta * Kmat).tocsr()
    A_rhs = (E / h - (1.0 - theta) * Kmat).tocsr()

    # Dirichlet for A on the outer boundary (constant in time).
    bnd = set(int(i) for i in _boundary_nodes(mesh, boundary))
    fixed = {}
    Bv = None
    if B_applied is not None:
        Bv = np.asarray(B_applied, dtype=float)
    for p in bnd:
        x = mesh.nodes[p]
        val = 0.5 * np.cross(Bv, x) if Bv is not None else np.zeros(3)
        for m in range(3):
            fixed[3 * p + m] = float(val[m])
    if a_bc:
        for name, vals in a_bc.items():
            fn = (lambda x, y, z, v=vals: np.asarray(v, dtype=float))
            if callable(vals):
                fn = vals
            for p in np.asarray(mesh.boundaries[name], dtype=int):
                vv = np.asarray(fn(*mesh.nodes[int(p)]),
                                dtype=float).ravel()
                for m in range(3):
                    fixed[3 * int(p) + m] = vv[m]

    A2, rhs0, keep, idx, val = _eliminate(A_lhs, np.zeros(nx), fixed)
    keep = np.asarray(keep, dtype=bool)
    val = np.asarray(np.real(val), dtype=float)  # real time-domain
    solve = spla.factorized(A2.tocsc())
    # Fixed-column contribution of A_lhs, precomputed once.
    A_kf = A_lhs.tocsr()[keep][:, idx] if len(idx) else None
    # The algebraic V row makes CN/BE monolithic stepping exact for a,
    # but CN is not stiffly accurate: the raw DAE unknown v would keep
    # a +/- flip-flop.  Diagnostics (power, returned V) therefore use
    # the constraint-consistent V = -S^-1 G^T adot (S is tiny; one
    # factorisation, cheap per-step apply).
    solve_S = spla.factorized(S.tocsc())

    if J_s is not None:
        f_static = _rhs_Js_tet10(mesh, J_s)
        time_scale = (lambda t: 1.0) if time_scale is None else time_scale
    else:
        f_static = None

    times = np.linspace(0.0, t_end, n_steps + 1)
    P = np.zeros(n_steps + 1)
    if history is True:
        rec = np.ones(n_steps + 1, dtype=bool)
    elif isinstance(history, (tuple, list)):
        t0h, t1h = history
        rec = (times >= t0h - 1e-12) & (times <= t1h + 1e-12)
    else:
        rec = np.zeros(n_steps + 1, dtype=bool)
    a_hist = np.zeros((int(rec.sum()), 3 * n)) if rec.any() else None
    t_hist = times[rec] if rec.any() else None
    if isinstance(history, (tuple, list)) and not rec.any():
        raise ValueError(f"history window {tuple(history)} selects no "
                         f"steps in [0, {t_end}]")
    x = np.zeros(nx)
    if len(idx):
        x[idx] = val
    a_prev = x[:3 * n].copy()
    _rh = 0
    if rec[0]:
        a_hist[0] = a_prev
        _rh = 1
    v_end = np.zeros(n)  # overwritten below; consistent V of last step

    for k in range(1, n_steps + 1):
        f = np.zeros(nx)
        if f_static is not None:
            s = (theta * time_scale(times[k])
                 + (1.0 - theta) * time_scale(times[k - 1]))
            f[:3 * n] = s * f_static
        b = A_rhs @ x + f
        b_red = b[keep]
        if A_kf is not None:
            b_red = b_red - A_kf @ val
        xk = np.zeros(nx)
        xk[keep] = solve(b_red)
        if len(idx):
            xk[idx] = val
        x = xk
        a_new = x[:3 * n]
        # Constraint-consistent V for diagnostics (see factorisation note
        # above): V = -S^-1 G^T adot, adot by backward difference.
        adot = (a_new - a_prev) / h
        v_new = np.zeros(n)
        v_new[v_nodes] = -solve_S(Gs.T @ adot)
        P[k] = transient_joule_power_tet10(mesh, a_prev, a_new, v_new,
                                           sigma, h)[0]
        a_prev = a_new.copy()
        v_end = v_new
        if rec[k]:
            a_hist[_rh] = a_new
            _rh += 1

    out = {"times": times, "P": P, "a_end": x[:3 * n].copy(),
           "v_end": v_end}
    if a_hist is not None:
        out["t_hist"] = t_hist
        out["a_hist"] = a_hist
    return out


def transient_joule_power_tet10(mesh, a_prev, a_new, v, sigma, h):
    """Total instantaneous Joule power [W] for one transient step.

    E = -(a_new - a_prev)/h - grad(V) (backward-difference E) is a
    *real instantaneous* field, so the power density is sigma*|E|^2
    with no 1/2 phasor time-averaging factor (that 1/2 belongs to
    the e^{iwt} phasor convention of total_joule_power_tet10).
    Exact for the discrete E via the exact Tet10 mass matrix and the
    *true* sigma (zero in air). Returns (P_total, Q_elem).
    """
    a_prev = np.asarray(a_prev, dtype=float).reshape(-1, 3)
    a_new = np.asarray(a_new, dtype=float).reshape(-1, 3)
    v = np.asarray(v, dtype=float)
    sig_e = _element_values(mesh, sigma)
    vols = mesh.element_volumes()
    Q = np.zeros(mesh.n_elements)
    for e, el in enumerate(mesh.elements):
        if sig_e[e] <= 0.0:
            continue
        el = np.asarray(el, dtype=int)
        grads = _nodal_gradients(mesh, el)          # (10, 10, 3)
        Ep = (-(a_new[el] - a_prev[el]) / h
              - np.einsum("pjx,j->px", grads, v[el]))
        re = np.einsum("pm,qm->pq", Ep, Ep)
        P_e = sig_e[e] * vols[e] * float(
            np.einsum("pq,pq->", _MASS_REF, re))
        Q[e] = P_e / vols[e]
    return float(np.sum(Q * vols)), Q


def magnetic_energy_tet10(mesh, a, nu=1.0 / MU0, K=None):
    """Time-averaged stored magnetic energy [J] from a phasor A field.

    W_m = 1/4 Re int B.H* dV.  Computed *exactly* for the discrete
    field as W_m = Re(a* . K . a)/4 with the assembled curl-curl
    matrix K (a*Ka = int nu|curl A_h|^2 by construction of K -- no
    quadrature error, unlike a centroid-B estimate which can be
    10x off near wire-like singularities).  Pass the assembled K
    when available; otherwise it is assembled here.
    """
    a = np.asarray(a, dtype=complex).reshape(-1, 3).ravel()
    if K is None:
        K, _, _, _, _, _, _ = assemble_tet10_av(mesh, nu=nu)
    W = 0.25 * float(np.real(np.vdot(a, K @ a)))
    return W
