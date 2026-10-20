"""Transient structural dynamics in 3D: Newmark time integration.

Extends the static tet-P1 elasticity solver (elastic3d) and the
v0.9 mass matrices (modal3d) to the semi-discrete equation of motion::

    M u'' + C u' + K u = F(t)

on the same interleaved DOF layout (DOF of component c of node i is
3*i + c). K comes from
:func:`cloudfem.physics.elastic3d.assemble_elasticity_3d`, M from
:func:`cloudfem.physics.modal3d.assemble_mass_3d` (consistent or
lumped), and the damping matrix is Rayleigh (proportional) damping::

    C = alpha * M + beta_r * K

with ``alpha`` / ``beta_r`` the mass / stiffness coefficients (either
may be zero; both zero gives the undamped system). The modal damping
ratio of mode j under Rayleigh damping is
``zeta_j = (alpha + beta_r * omega_j**2) / (2 * omega_j)``.

Time integration is the Newmark-beta family::

    u'_{n+1}  = u'_n + dt * [(1 - gamma) u''_n + gamma u''_{n+1}]
    u_{n+1}   = u_n + dt u'_n
                + dt^2 * [(1/2 - beta) u''_n + beta u''_{n+1}]

solved in the standard effective-stiffness form: with

    Khat = K + gamma/(beta dt) C + 1/(beta dt^2) M

(factorised once per run, mirroring the v1.0 transient heat solver),
each step solves Khat u_{n+1} = F_eff(t_{n+1}) and recovers
u''_{n+1}, u'_{n+1} from the Newmark relations. The initial
acceleration follows from equilibrium at t = 0,
``M u''_0 = F(0) - C u'_0 - K u_0``.

Two named parameter sets are provided:

- ``method="average"`` (default): beta = 1/4, gamma = 1/2, the
  constant-average-acceleration (trapezoidal) scheme. Second-order
  accurate and *unconditionally stable* for linear problems; for the
  undamped system it conserves the total mechanical energy
  1/2 u'^T M u' + 1/2 u^T K u to round-off level.
- ``method="linear"``: beta = 1/6, gamma = 1/2, the
  linear-acceleration scheme. Also second-order, but only
  *conditionally stable*: it requires ``omega_max dt <= 2*sqrt(3)``,
  i.e. the step must resolve the highest modal period of the mesh,
  ``dt <= 0.551 * T_min``. On a fine mesh T_min is tiny, so this
  variant is mainly of academic / comparison interest.

Explicit ``beta`` / ``gamma`` arguments override ``method``. The
general stability condition is ``gamma >= 1/2`` together with
``beta >= (gamma + 1/2)^2 / 4`` for unconditional stability;
``gamma > 1/2`` adds algorithmic damping (and drops the accuracy to
first order), which is not the default anywhere here.

Boundary conditions follow the semantics of modal3d / elastic3d
(``bc`` maps a boundary name or ``'all'`` to (ux, uy, uz)), with one
restriction: only *homogeneous* Dirichlet conditions are supported --
constrained DOFs are held at zero for all time. Nonzero or
time-varying prescribed motion (base excitation) would add
effective-load terms from the constrained DOFs and is not
implemented. Loads are nodal force vectors on the full 3N DOFs:
either a constant (3N,) array or a callable ``F(t) -> (3N,)``, so
step, constant and harmonic loading are all expressible. Static body
forces / tractions from elastic3d can be pre-assembled with
:func:`assemble_elasticity_3d` and passed as a constant vector.

Assumptions: linear elasticity, small strains and displacements,
constant density, time-constant M, C, K (no geometric or material
nonlinearity, no contact).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elastic3d import assemble_elasticity_3d
from .modal3d import assemble_mass_3d, solve_modes_3d, _fixed_dofs

#: Named Newmark parameter sets: name -> (beta, gamma).
NEWMARK_METHODS = {
    "average": (0.25, 0.5),        # constant average acceleration
    "linear": (1.0 / 6.0, 0.5),    # linear acceleration
}


def _newmark_params(method, beta, gamma):
    if beta is None or gamma is None:
        if method not in NEWMARK_METHODS:
            raise ValueError(f"unknown method {method!r}; use one of "
                             f"{sorted(NEWMARK_METHODS)} or pass "
                             "beta/gamma explicitly")
        b_def, g_def = NEWMARK_METHODS[method]
        beta = b_def if beta is None else beta
        gamma = g_def if gamma is None else gamma
    if beta <= 0.0:
        raise ValueError("Newmark beta must be positive")
    return float(beta), float(gamma)


def _as_force_fn(load, n):
    """Normalise a load spec to a callable F(t) -> (n,) array."""
    if load is None:
        return lambda t: np.zeros(n)
    if callable(load):
        def fn(t, _load=load):
            F = np.asarray(_load(t), dtype=float).ravel()
            if F.shape != (n,):
                raise ValueError(f"load callable returned shape "
                                 f"{F.shape}, expected ({n},)")
            return F
        return fn
    F = np.asarray(load, dtype=float).ravel()
    if F.shape != (n,):
        raise ValueError(f"constant load has shape {F.shape}, "
                         f"expected ({n},)")
    return lambda t: F


def newmark_integrate(M, C, K, load, t_end, dt, u0=None, v0=None,
                      method="average", beta=None, gamma=None):
    """Integrate ``M u'' + C u' + K u = F(t)`` with Newmark-beta.

    Low-level core: operates directly on the (possibly reduced)
    system matrices, so it also serves small dense systems (e.g. a
    single-DOF oscillator in the validation suite).

    Parameters
    ----------
    M, C, K : (n, n) sparse matrices or dense arrays. ``C`` may be
        None (no damping). ``M`` must be nonsingular.
    load : None, (n,) constant array, or callable F(t) -> (n,).
    t_end, dt : final time and nominal step. The step actually used
        is ``t_end / ceil(t_end / dt)``, so the last step lands
        exactly on ``t_end`` (same convention as the theta-method
        heat solver).
    u0, v0 : None, scalar, or (n,) initial displacement / velocity.
    method, beta, gamma : Newmark parameters (module docstring).

    Returns
    -------
    times : (S + 1,) array including t = 0.
    U, V, A : (S + 1, n) displacement / velocity / acceleration at
        each output time.
    """
    beta, gamma = _newmark_params(method, beta, gamma)
    M = sp.csr_matrix(M, dtype=float)
    K = sp.csr_matrix(K, dtype=float)
    n = M.shape[0]
    C = (sp.csr_matrix((n, n), dtype=float) if C is None
         else sp.csr_matrix(C, dtype=float))
    F = _as_force_fn(load, n)

    n_steps = int(np.ceil(t_end / dt - 1e-12))
    if n_steps < 1:
        raise ValueError("t_end must be positive")
    h = t_end / n_steps

    u = (np.zeros(n) if u0 is None else
         np.broadcast_to(np.asarray(u0, dtype=float), (n,)).copy())
    v = (np.zeros(n) if v0 is None else
         np.broadcast_to(np.asarray(v0, dtype=float), (n,)).copy())

    # effective stiffness, factorised once and reused every step
    a1 = 1.0 / (beta * h * h)
    a2 = 1.0 / (beta * h)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / (beta * h)
    a5 = gamma / beta - 1.0
    a6 = h * (gamma / (2.0 * beta) - 1.0)
    Khat = (K + a4 * C + a1 * M).tocsc()
    solve_hat = spla.factorized(Khat)
    solve_M = spla.factorized(M.tocsc())

    # initial acceleration from equilibrium M a0 = F - C v0 - K u0
    a = np.asarray(solve_M(F(0.0) - C @ v - K @ u)).ravel()

    times = [0.0]
    U, V, A = [u.copy()], [v.copy()], [a.copy()]
    t = 0.0
    for _ in range(n_steps):
        t_next = t + h
        F_eff = (F(t_next)
                 + M @ (a1 * u + a2 * v + a3 * a)
                 + C @ (a4 * u + a5 * v + a6 * a))
        u_next = np.asarray(solve_hat(F_eff)).ravel()
        a_next = a1 * (u_next - u) - a2 * v - a3 * a
        v_next = v + h * ((1.0 - gamma) * a + gamma * a_next)
        u, v, a = u_next, v_next, a_next
        t = t_next
        times.append(t)
        U.append(u.copy())
        V.append(v.copy())
        A.append(a.copy())
    times[-1] = t_end  # exact landing (h * n_steps may round)
    return np.array(times), np.array(U), np.array(V), np.array(A)


def _check_homogeneous_bc(mesh, bc):
    """Reject nonzero prescribed displacements (dynamics is
    homogeneous-only; see module docstring)."""
    if not bc:
        return
    for name, vals in bc.items():
        for v in vals:
            if v is None:
                continue
            if callable(v):
                ids = (np.arange(mesh.n_nodes) if name == "all"
                       else np.asarray(mesh.boundaries[name]))
                for i in ids:
                    x, y, z = mesh.nodes[int(i)]
                    if v(x, y, z) != 0.0:
                        raise ValueError(
                            "transient_3d supports only homogeneous "
                            "Dirichlet BCs (prescribed value 0)")
            elif float(v) != 0.0:
                raise ValueError(
                    "transient_3d supports only homogeneous "
                    "Dirichlet BCs (prescribed value 0)")


def transient_3d(mesh, E, nu, rho, bc=None, t_end=1.0, dt=0.01,
                 load=None, u0=None, v0=None, method="average",
                 beta=None, gamma=None, rayleigh=(0.0, 0.0),
                 lumped=False):
    """Transient dynamics of a tetrahedral mesh (direct Newmark).

    Parameters
    ----------
    mesh : TetMesh
    E, nu, rho : Young's modulus, Poisson's ratio, density.
    bc : None or Dirichlet dict as in elastic3d; only homogeneous
        (zero) values are supported -- constrained DOFs are held at
        zero for all time.
    t_end, dt : final time and nominal step (see newmark_integrate).
    load : None, (3N,) constant nodal force vector, or callable
        F(t) -> (3N,). Step / harmonic loads are callables; static
        body or traction loads can be pre-assembled with
        elastic3d.assemble_elasticity_3d and passed as a constant.
    u0, v0 : None, scalar, or (3N,) initial displacement / velocity;
        values at constrained DOFs are overwritten with zero.
    method, beta, gamma : Newmark parameters (module docstring).
    rayleigh : (alpha, beta_r) Rayleigh damping coefficients,
        C = alpha * M + beta_r * K. Default (0, 0): undamped.
    lumped : use the lumped (row-sum) mass matrix instead of the
        consistent one.

    Returns
    -------
    times : (S + 1,) array including t = 0.
    U, V, A : (S + 1, 3N) displacement / velocity / acceleration
        histories on the full DOF vector (zero at constrained DOFs).
    """
    _check_homogeneous_bc(mesh, bc)
    K, _ = assemble_elasticity_3d(mesh, E, nu)
    M = assemble_mass_3d(mesh, rho, lumped=lumped)
    alpha_r, beta_r = rayleigh
    C = alpha_r * M + beta_r * K
    n3 = 3 * mesh.n_nodes

    fixed = _fixed_dofs(mesh, bc) if bc else np.zeros(0, dtype=int)
    free = np.ones(n3, dtype=bool)
    free[fixed] = False
    free_idx = np.where(free)[0]

    def expand0(w):
        if w is None:
            return None
        w = np.broadcast_to(np.asarray(w, dtype=float), (n3,)).copy()
        w[fixed] = 0.0
        return w[free_idx]

    F_full = _as_force_fn(load, n3)
    times, Uf, Vf, Af = newmark_integrate(
        M[free][:, free], C[free][:, free], K[free][:, free],
        lambda t: F_full(t)[free_idx], t_end, dt,
        u0=expand0(u0), v0=expand0(v0),
        method=method, beta=beta, gamma=gamma)

    S = len(times)
    U = np.zeros((S, n3))
    V = np.zeros((S, n3))
    A = np.zeros((S, n3))
    U[:, free_idx] = Uf
    V[:, free_idx] = Vf
    A[:, free_idx] = Af
    return times, U, V, A


def modal_superposition_3d(mesh, E, nu, rho, bc=None, t_end=1.0,
                           dt=0.01, load=None, u0=None, v0=None,
                           n_modes=12, method="average", beta=None,
                           gamma=None, rayleigh=(0.0, 0.0),
                           lumped=False):
    """Transient dynamics by modal superposition (Newmark per mode).

    The response is expanded in the lowest ``n_modes`` modes of
    :func:`cloudfem.physics.modal3d.solve_modes_3d` (M-orthonormal),
    each modal coordinate obeying the scalar oscillator equation

        q_j'' + (alpha + beta_r omega_j^2) q_j' + omega_j^2 q_j
            = phi_j^T F(t)

    with Rayleigh damping diagonal in the modal basis. The scalar
    equations are integrated with the same Newmark core as the direct
    solver, so the only difference from :func:`transient_3d` is the
    modal truncation: modes above ``n_modes`` (and their share of the
    load) are dropped, which mainly affects the acceleration and the
    quasi-static content of the response.

    Parameters and return value mirror :func:`transient_3d`, except
    that ``n_modes`` selects the modal basis size. Initial conditions
    are projected as q_j(0) = phi_j^T M u0, q'_j(0) = phi_j^T M v0.
    """
    _check_homogeneous_bc(mesh, bc)
    freqs, modes = solve_modes_3d(mesh, E, nu, rho, bc=bc,
                                  n_modes=n_modes, lumped=lumped)
    M = assemble_mass_3d(mesh, rho, lumped=lumped)
    n3 = 3 * mesh.n_nodes
    omega2 = (2.0 * np.pi * freqs) ** 2
    alpha_r, beta_r = rayleigh

    F_full = _as_force_fn(load, n3)
    q0 = (modes.T @ (M @ np.broadcast_to(
        np.asarray(u0, dtype=float), (n3,))) if u0 is not None
        else np.zeros(n_modes))
    r0 = (modes.T @ (M @ np.broadcast_to(
        np.asarray(v0, dtype=float), (n3,))) if v0 is not None
        else np.zeros(n_modes))

    # diagonal modal system: M = I, C = diag(alpha + beta_r omega^2),
    # K = diag(omega^2); integrated by the same Newmark core.
    times, Q, Qd, Qdd = newmark_integrate(
        sp.diags(np.ones(n_modes), format="csr"),
        sp.diags(alpha_r + beta_r * omega2, format="csr"),
        sp.diags(omega2, format="csr"),
        lambda t: modes.T @ F_full(t), t_end, dt,
        u0=q0, v0=r0, method=method, beta=beta, gamma=gamma)
    return times, Q @ modes.T, Qd @ modes.T, Qdd @ modes.T


def node_series(history, node, comp):
    """Time series of one displacement/velocity component.

    ``history`` is one of the (S + 1, 3N) arrays returned by
    :func:`transient_3d`; returns the (S + 1,) series of component
    ``comp`` (0 = x, 1 = y, 2 = z) of node ``node``.
    """
    return history[:, 3 * int(node) + int(comp)]


def peak_response(times, U):
    """Peak summary of a displacement history.

    Returns (peak_per_dof, t_peak, dof_of_max): the maximum absolute
    displacement of every DOF over the run, the time at which the
    overall maximum occurs, and the DOF attaining it.
    """
    peak = np.abs(U).max(axis=0)
    s, dof = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    return peak, float(times[s]), int(dof)
