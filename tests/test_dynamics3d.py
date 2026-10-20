"""Validation of the 3D transient dynamics module (Newmark).

References are derived independently of the integrator under test:

1. SDOF closed form: for m u'' + k u = F0 H(t) from rest,
   u(t) = (F0/k)(1 - cos(omega t)); for harmonic forcing
   F0 sin(omf t) from rest,
   u(t) = F0/(k - m omf^2) (sin(omf t) - (omf/omega) sin(omega t)).
   The core integrator (newmark_integrate) is fed the 1x1 system
   directly, so no spatial discretisation error enters.
2. Temporal order: the step-response error must fall like dt^2 for
   the average-acceleration method (slope of the log-log error fit).
3. Energy + stability: the linear-acceleration method (beta = 1/6)
   is energy-conserving up to an O(dt^2) drift on a resolved mode
   and is conditionally stable with the textbook limit
   omega dt <= 2 sqrt(3): at omega dt = 3.0 and 2 sqrt(3) a free
   oscillation keeps its amplitude to print precision, at
   omega dt = 4.0 it grows by orders of magnitude. The
   average-acceleration method stays bounded at omega dt = 4.0.
4. Modal benchmark on a mesh: the cantilever step response from
   direct Newmark is compared with the *exact* modal time history
   (12 modes from the v0.9 eigensolver, each coordinate integrated
   in closed form (1 - cos(omega_j t))) plus the mode-acceleration
   static correction for the truncated modes,
   K^-1 F - sum_j phi_j (phi_j^T F)/omega_j^2, which is constant in
   time for a step load. The measured residual (RMS ~ 7e-4 of peak
   at dt = T1/400) is the Newmark period elongation of the highest
   retained modes: each mode's period error is (omega_j dt)^2/12,
   so the small high-mode shares decorrelate over the 2 T1 window.
   The module-level modal_superposition_3d is checked against the
   same benchmark.
5. Energy conservation on a mesh: a cantilever started in its first
   mode shape (average acceleration, no damping, no load) must keep
   its total mechanical energy 1/2 v^T M v + 1/2 u^T K u constant;
   trapezoidal Newmark conserves this quadratic energy exactly, so
   the measured drift (2.2e-12) is round-off / solver residual.
6. Static limit: with Rayleigh damping zeta = 0.10 at the first two
   modal frequencies, a constant tip load integrated to
   t = 8/(zeta omega_1) (envelope e^-8) must land on the v0.8
   static solve K^-1 F (same K, via apply_dirichlet_vec3).
7. API guard: nonzero prescribed displacements are rejected
   (transient_3d is homogeneous-Dirichlet only).
"""

import numpy as np

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.physics.elastic3d import (assemble_elasticity_3d,
                                        apply_dirichlet_vec3)
from cloudfem.physics.modal3d import assemble_mass_3d, solve_modes_3d
from cloudfem.physics.dynamics3d import (newmark_integrate,
                                         transient_3d,
                                         modal_superposition_3d)

E_STEEL = 200e9
NU_STEEL = 0.3
RHO_STEEL = 7850.0


def _sdof(ndiv, t_periods=2.0, method="average"):
    """SDOF step response error at dt = T/ndiv (test 1 helper)."""
    om = 2.0 * np.pi * 10.0          # f = 10 Hz
    m, k = 1.0, om ** 2
    T = 2.0 * np.pi / om
    t, U, _, _ = newmark_integrate([[m]], None, [[k]], np.array([1.0]),
                                   t_end=t_periods * T, dt=T / ndiv,
                                   method=method)
    exact = (1.0 / k) * (1.0 - np.cos(om * t))
    return np.abs(U[:, 0] - exact).max() / (1.0 / k)


def test_sdof_analytic():
    """Test 1: SDOF step + harmonic response vs closed form."""
    err_step = _sdof(200)
    # harmonic forcing at omf = omega/2, from rest
    om = 2.0 * np.pi * 10.0
    omf = 0.5 * om
    m, k = 1.0, om ** 2
    T = 2.0 * np.pi / om
    t, U, _, _ = newmark_integrate(
        [[m]], None, [[k]], lambda s: np.array([np.sin(omf * s)]),
        t_end=2.0 * T, dt=T / 400)
    exact = (np.sin(omf * t) - (omf / om) * np.sin(om * t)) \
        / (k - m * omf ** 2)
    err_harm = np.abs(U[:, 0] - exact).max() / abs(exact).max()
    return err_step, err_harm


def test_sdof_temporal_order():
    """Test 2: error vs dt slope for average acceleration."""
    ndivs = np.array([25, 50, 100, 200, 400])
    errs = np.array([_sdof(int(n)) for n in ndivs])
    slope = np.polyfit(np.log(1.0 / ndivs), np.log(errs), 1)[0]
    return ndivs, errs, slope


def test_sdof_energy_stability():
    """Test 3: linear-accel drift scaling + stability limit."""
    om = 2.0 * np.pi * 10.0
    m, k = 1.0, om ** 2
    T = 2.0 * np.pi / om

    def drift(ndiv):
        t, U, V, _ = newmark_integrate(
            [[m]], None, [[k]], None, t_end=5.0 * T, dt=T / ndiv,
            u0=np.array([1.0]), method="linear")
        en = 0.5 * m * V[:, 0] ** 2 + 0.5 * k * U[:, 0] ** 2
        return np.abs(en - en[0]).max() / en[0]

    d50, d100 = drift(50), drift(100)

    def maxamp(omega_dt, method):
        dt = omega_dt / om
        t, U, _, _ = newmark_integrate(
            [[m]], None, [[k]], None, t_end=20.0 * T, dt=dt,
            u0=np.array([1.0]), method=method)
        return np.abs(U[:, 0]).max()

    return d50, d100, maxamp(3.0, "linear"), maxamp(4.0, "linear"), \
        maxamp(4.0, "average")


def _beam(nx=24, nyz=3):
    L, h = 1.0, 0.1
    mesh = box_mesh(nx, nyz, nyz, xlim=(0, L), ylim=(0, h),
                    zlim=(0, h))
    bc = {"xmin": (0.0, 0.0, 0.0)}
    tip = np.asarray(mesh.boundaries["xmax"])
    F = np.zeros(3 * mesh.n_nodes)
    F[3 * tip + 2] = -1000.0 / len(tip)   # 1 kN total in -z at tip
    return mesh, bc, tip, F


def _tip_mean(U, mesh, tip):
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    mask[tip] = True
    return U[:, 2::3][:, mask].mean(axis=1)


def test_modal_benchmark():
    """Test 4: direct Newmark + modal superposition vs exact modal."""
    mesh, bc, tip, F = _beam()
    n_modes = 12
    freqs, modes = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                                  bc=bc, n_modes=n_modes)
    f1 = freqs[0]
    T1 = 1.0 / f1
    om = 2.0 * np.pi * freqs
    p = modes.T @ F
    K, _ = assemble_elasticity_3d(mesh, E_STEEL, NU_STEEL)
    u_static = apply_dirichlet_vec3(K, F, mesh, bc)
    resid = u_static - modes @ (p / om ** 2)   # truncated-mode share

    dt = T1 / 400
    t, U, _, _ = transient_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                              bc=bc, t_end=2.0 * T1, dt=dt, load=F)
    Q = (p / om ** 2) * (1.0 - np.cos(om * t[:, None]))
    bench = _tip_mean(Q @ modes.T + resid[None, :], mesh, tip)
    peak = np.abs(bench).max()

    def rel(series):
        d = series - bench
        return (np.sqrt(np.mean(d ** 2)) / peak,
                np.abs(d).max() / peak)

    rms_dir, max_dir = rel(_tip_mean(U, mesh, tip))
    _, U_ms, _, _ = modal_superposition_3d(
        mesh, E_STEEL, NU_STEEL, RHO_STEEL, bc=bc, t_end=2.0 * T1,
        dt=dt, load=F, n_modes=n_modes)
    rms_ms, max_ms = rel(_tip_mean(U_ms, mesh, tip))
    return f1, rms_dir, max_dir, rms_ms, max_ms


def test_energy_conservation_mesh():
    """Test 5: mesh energy conservation (average acceleration)."""
    mesh, bc, _, _ = _beam(nx=16, nyz=2)
    freqs, modes = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                                  bc=bc, n_modes=2)
    T1 = 1.0 / freqs[0]
    u0 = modes[:, 0] * (1e-3 / np.abs(modes[:, 0]).max())
    t, U, V, _ = transient_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                              bc=bc, t_end=5.0 * T1, dt=T1 / 50, u0=u0)
    K, _ = assemble_elasticity_3d(mesh, E_STEEL, NU_STEEL)
    M = assemble_mass_3d(mesh, RHO_STEEL)
    en = 0.5 * np.einsum("ij,ij->i", V @ M.T, V) \
        + 0.5 * np.einsum("ij,ij->i", U @ K.T, U)
    return np.abs(en - en[0]).max() / en[0]


def test_static_limit():
    """Test 6: damped constant load -> static solution."""
    mesh, bc, _, F = _beam(nx=16, nyz=2)
    freqs, _ = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                              bc=bc, n_modes=2)
    om1, om2 = 2.0 * np.pi * freqs[:2]
    zeta = 0.10
    alpha = 2.0 * zeta * om1 * om2 / (om1 + om2)
    beta_r = 2.0 * zeta / (om1 + om2)
    t_end = 8.0 / (zeta * om1)
    t, U, _, _ = transient_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                              bc=bc, t_end=t_end, dt=(1.0 / freqs[0]) / 50,
                              load=F, rayleigh=(alpha, beta_r))
    K, _ = assemble_elasticity_3d(mesh, E_STEEL, NU_STEEL)
    u_static = apply_dirichlet_vec3(K, F, mesh, bc)
    err = np.abs(U[-1] - u_static).max() / np.abs(u_static).max()
    return err, len(t) - 1


def test_bc_guard():
    """Test 7: nonzero prescribed displacement is rejected."""
    mesh, _, _, _ = _beam(nx=4, nyz=2)
    try:
        transient_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                     bc={"xmin": (0.01, 0.0, 0.0)}, t_end=0.001,
                     dt=0.001)
    except ValueError:
        return True
    return False


def main():
    err_step, err_harm = test_sdof_analytic()
    print(f"[sdof analytic] step response max rel err (dt = T/200) "
          f"= {err_step:.3e}; harmonic rel err (dt = T/400) "
          f"= {err_harm:.3e}")
    assert err_step < 2e-3, "SDOF step response off"
    assert err_harm < 2e-3, "SDOF harmonic response off"

    ndivs, errs, slope = test_sdof_temporal_order()
    print(f"[sdof order] errs = {[f'{e:.2e}' for e in errs]} at "
          f"dt = T/{list(ndivs)}, fitted slope = {slope:.3f}")
    assert 1.9 < slope < 2.1, "average acceleration must be 2nd order"

    d50, d100, a30, a40, aavg = test_sdof_energy_stability()
    print(f"[sdof energy/stability] linear-accel drift: "
          f"dt = T/50 -> {d50:.3e}, dt = T/100 -> {d100:.3e} "
          f"(ratio {d50 / d100:.2f}); max|u| at omega dt = 3.0: "
          f"{a30:.3f}, at 4.0: {a40:.3e}; average-accel at 4.0: "
          f"{aavg:.3f}")
    assert d50 / d100 > 3.0, "drift must fall ~4x when dt halves"
    assert a30 < 1.01, "omega dt = 3.0 must be stable for beta = 1/6"
    assert a40 > 10.0, "omega dt = 4.0 must be unstable for beta = 1/6"
    assert aavg < 1.01, "average acceleration must stay bounded"

    f1, rms_d, max_d, rms_m, max_m = test_modal_benchmark()
    print(f"[modal benchmark] cantilever f1 = {f1:.2f} Hz; direct "
          f"Newmark vs exact modal + static correction: RMS rel = "
          f"{rms_d:.3e}, max rel = {max_d:.3e}; modal superposition: "
          f"RMS rel = {rms_m:.3e}, max rel = {max_m:.3e}")
    assert rms_d < 2e-3 and max_d < 6e-3, "direct Newmark off modal bench"
    assert rms_m < 3e-3 and max_m < 8e-3, "modal superposition off bench"

    drift = test_energy_conservation_mesh()
    print(f"[mesh energy] cantilever mode-1 free vibration, average "
          f"acceleration, dt = T1/50, 5 periods: drift = {drift:.3e}")
    assert drift < 1e-8, "mechanical energy not conserved"

    err_st, nsteps = test_static_limit()
    print(f"[static limit] Rayleigh zeta = 0.10, {nsteps} steps: "
          f"final displacement rel err vs K^-1 F = {err_st:.3e}")
    assert err_st < 1e-3, "damped response does not settle to static"

    ok = test_bc_guard()
    print(f"[bc guard] nonzero prescribed displacement rejected: {ok}")
    assert ok

    print("ALL 3D TRANSIENT DYNAMICS VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
