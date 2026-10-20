"""Validation of the 3D transient heat solver + stress chain (v1.0).

All reference values below are derived independently of the solver
under test (eigenfunction series, 1D steady conduction, global
energy balances, linear-elasticity closed forms).

1. Capacity matrix: ones^T C ones = rho_cp * V exactly (the P1
   integrals int phi_i phi_j dV = V/20 (1 + delta_ij) sum to V);
   C is symmetric positive definite.
2. 1D slab transient: a thin bar (40 x 2 x 2 cells) with both end
   faces held at Ts and insulated sides is the classical slab
   problem. theta = (T - Ts)/(T0 - Ts) has the series solution
       theta(x, t) = (4/pi) sum_n sin((2n+1) pi x/L)
                          / (2n+1) * exp(-(2n+1)^2 pi^2 Fo),
   Fo = alpha t / L^2. Compared at Fo = 0.05..0.3; errors must sit
   at the P1 + time-integration level (measured: BE <= 5.2e-3,
   CN <= 8.3e-4 of the temperature jump).
3. Steady-state limit: implicit Euler with large steps on a cube
   with a uniform source must converge to the v0.7 steady solver's
   field (the fixed point of the BE iteration solves K T = F
   exactly, for any dt).
4. Robin steady state (1D): bar with T = 0 at x = 0 and convection
   h, T_inf at x = L has the exact linear profile
       T(x) = T_inf * Bi * (x/L) / (1 + Bi),   Bi = h L / k,
   which P1 reproduces nodally; the transient run to steady state
   must land on it.
5. Neumann energy balance: uniform inward flux q on all six faces
   of a unit cube adds heat at exactly q * A_total per unit time;
   the discrete total heat ones^T C T (exact for the consistent
   capacity matrix) must track E0 + q A t to roundoff, and the mean
   temperature at t = 1 is exactly T0 + q A / (rho_cp V) = 28.
6. Adiabatic relaxation: T0 = 50 + 100 sin(pi x) sin(pi y) sin(pi z)
   with no boundary conditions conserves total heat exactly (both
   integrators: 1^T K = 0 annihilates the diffusion term). Maximum
   principle: with the consistent capacity matrix the bounds
   [50, 150] hold exactly for dt >= h^2/(6 alpha) (measured: zero
   violation at dt = 0.02 with h = 0.25); below that threshold a
   small undershoot is the documented consistent-mass behaviour
   (measured <= 3.1 K at dt = 0.005; asserted tolerance 5 K, i.e.
   5 % of the temperature range). A lumped matrix would be needed
   for an unconditional guarantee.
7. Temporal convergence order: on the smooth manufactured decay
   T = exp(-3 pi^2 t) sin(pi x) sin(pi y) sin(pi z) (exact solution
   of the homogeneous equation, zero Dirichlet faces), the order is
   measured from successive step halvings on a fixed mesh,
       e(dt) = || T_dt(t_end) - T_{dt/2}(t_end) ||_inf,
   which cancels the (dt-independent) spatial error exactly and
   isolates the temporal part: BE ~ O(dt), CN ~ O(dt^2).
8. Transient chain (uniform source + adiabatic + fully
   constrained): with a uniform volumetric source Q and insulated
   boundaries the exact temperature is spatially uniform,
   T(t) = T0 + Q t / rho_cp, at every instant (K annihilates
   constants and the discrete equations admit the uniform ansatz
   exactly, for both integrators). The fully constrained stress is
   then the v0.8 closed form at each instant,
       sigma(t) = -E alpha (T(t) - T_ref) / (1 - 2 nu) * I,
   with zero von Mises (hydrostatic) and zero displacement.
"""

import numpy as np

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson
from cloudfem.fem.tet_heat import transient_heat_tet, assemble_mass_tet
from cloudfem.physics.thermomech import transient_thermal_stress_3d

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _slab_series(x, Fo, n_terms=400):
    """theta = (T - Ts)/(T0 - Ts), slab on [0, 1], faces at Ts."""
    n = np.arange(n_terms)
    m = 2 * n + 1
    coef = (4.0 / (m * np.pi)) * np.exp(-(m * np.pi) ** 2 * Fo)
    return np.sin(np.outer(x, m) * np.pi) @ coef


def _bar_mesh(nx=40):
    return box_mesh(nx, 2, 2, xlim=(0.0, 1.0),
                    ylim=(0.0, 0.05), zlim=(0.0, 0.05))


def test_capacity_matrix():
    mesh = box_mesh(4, 4, 4, xlim=(0, 2.0), ylim=(0, 1.0), zlim=(0, 0.5))
    C = assemble_mass_tet(mesh, rho_cp=3.0)
    ones = np.ones(mesh.n_nodes)
    total = float(ones @ (C @ ones))
    err_total = abs(total - 3.0 * 2.0 * 1.0 * 0.5) / 3.0
    err_sym = float(np.abs((C - C.T).toarray()).max())
    lam_min = float(np.linalg.eigvalsh(C.toarray())[0])
    return err_total, err_sym, lam_min


def test_slab_series():
    Ts, T0 = 100.0, 0.0
    mesh = _bar_mesh()
    x = mesh.nodes[:, 0]
    bc = {"xmin": Ts, "xmax": Ts}
    out = {}
    for method, dt in (("be", 0.0025), ("cn", 0.00125)):
        _, hist = transient_heat_tet(mesh, k=1.0, rho_cp=1.0, bc=bc,
                                     t_end=0.3, dt=dt, T0=T0,
                                     method=method)
        errs = []
        for Fo in (0.05, 0.1, 0.2, 0.3):
            i = int(round(Fo / dt))
            exact = Ts + (T0 - Ts) * _slab_series(x, Fo)
            errs.append(float(np.abs(hist[i] - exact).max()) / (Ts - T0))
        out[method] = errs
    return out


def test_steady_limit():
    mesh = box_mesh(5, 5, 5)
    bc = {n: 0.0 for n in FACES}
    T_ss = solve_tet_poisson(mesh, k=1.0, f=8.0, bc=bc)
    _, hist = transient_heat_tet(mesh, k=1.0, rho_cp=1.0, f=8.0, bc=bc,
                                 t_end=12.0, dt=2.0, T0=0.0, method="be")
    return float(np.abs(hist[-1] - T_ss).max() / np.abs(T_ss).max())


def test_robin_steady():
    mesh = _bar_mesh(nx=20)
    h, T_inf, k = 2.0, 100.0, 1.0
    _, hist = transient_heat_tet(mesh, k=k, rho_cp=1.0, bc={"xmin": 0.0},
                                 robin={"xmax": (h, T_inf)},
                                 t_end=20.0, dt=2.0, T0=0.0, method="be")
    Bi = h * 1.0 / k
    exact = T_inf * Bi * mesh.nodes[:, 0] / (1.0 + Bi)
    return float(np.abs(hist[-1] - exact).max() / T_inf)


def test_neumann_energy():
    mesh = box_mesh(4, 4, 4)
    C = assemble_mass_tet(mesh, rho_cp=1.0)
    ones = np.ones(mesh.n_nodes)
    neu = {n: 3.0 for n in FACES}
    out = {}
    for method in ("be", "cn"):
        times, hist = transient_heat_tet(mesh, k=1.0, rho_cp=1.0,
                                         neumann=neu, t_end=1.0, dt=0.1,
                                         T0=10.0, method=method)
        E = hist @ (C @ ones)
        E_exact = E[0] + 3.0 * 6.0 * times  # q * A_total * t
        out[method] = (float(np.abs(E - E_exact).max() / E_exact[-1]),
                       float(E[-1]))
    return out


def test_adiabatic():
    mesh = box_mesh(4, 4, 4)
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    T0f = 50.0 + 100.0 * np.sin(np.pi * x) * np.sin(np.pi * y) \
        * np.sin(np.pi * z)
    C = assemble_mass_tet(mesh, rho_cp=1.0)
    ones = np.ones(mesh.n_nodes)
    E0 = float(ones @ (C @ T0f))
    out = {}
    for method in ("be", "cn"):
        res = {}
        for dt in (0.02, 0.005):
            _, hist = transient_heat_tet(mesh, k=1.0, rho_cp=1.0,
                                         t_end=0.2, dt=dt, T0=T0f,
                                         method=method)
            E = hist @ (C @ ones)
            res[dt] = (float(np.abs(E - E0).max() / E0),
                       float(hist.max() - T0f.max()),
                       float(T0f.min() - hist.min()))
        out[method] = res
    return out


def test_temporal_order():
    mesh = box_mesh(6, 6, 6)
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    T0f = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    bc = {n: 0.0 for n in FACES}
    out = {}
    for method in ("be", "cn"):
        diffs = []
        prev = None
        for dt in (0.01, 0.005, 0.0025, 0.00125):
            _, hist = transient_heat_tet(mesh, k=1.0, rho_cp=1.0, bc=bc,
                                         t_end=0.05, dt=dt, T0=T0f,
                                         method=method)
            if prev is not None:
                diffs.append(float(np.abs(hist[-1] - prev).max()))
            prev = hist[-1]
        orders = [float(np.log(diffs[i] / diffs[i + 1]) / np.log(2.0))
                  for i in range(len(diffs) - 1)]
        out[method] = orders
    return out


def test_chain_uniform_source():
    E, NU, AL = 200e9, 0.3, 12e-6
    mesh = box_mesh(4, 4, 4)
    bc_el = {n: (0.0, 0.0, 0.0) for n in FACES}
    out = {}
    for method in ("be", "cn"):
        times, hist = transient_heat_tet(mesh, k=1.0, rho_cp=2.0, f=6.0,
                                         t_end=1.0, dt=0.25, T0=20.0,
                                         method=method)
        chain = transient_thermal_stress_3d(mesh, times, hist, E, NU, AL,
                                            T_ref=20.0, bc=bc_el)
        err_T = err_s = 0.0
        for i, t in enumerate(times):
            T_ex = 20.0 + 3.0 * t
            err_T = max(err_T, float(np.abs(hist[i] - T_ex).max()))
            sig_ex = -E * AL * (T_ex - 20.0) / (1.0 - 2.0 * NU)
            sig = chain["history"][i]["sig_elem"]
            if t > 0:
                err_s = max(err_s, float(
                    np.abs(sig[:, :3] - sig_ex).max() / abs(sig_ex)))
        out[method] = (err_T, err_s, float(chain["vm_max"].max()),
                       float(chain["u_max"].max()))
    return out


def main():
    err_total, err_sym, lam_min = test_capacity_matrix()
    print(f"capacity matrix: total err = {err_total:.3e}, "
          f"asymmetry = {err_sym:.3e}, min eig = {lam_min:.3e}")
    assert err_total < 1e-12 and err_sym < 1e-14 and lam_min > 0.0

    errs = test_slab_series()
    for method, tol in (("be", 1e-2), ("cn", 2e-3)):
        print(f"slab series [{method}]: "
              f"{[f'{e:.3e}' for e in errs[method]]} (tol {tol:.0e})")
        assert max(errs[method]) < tol, f"{method} slab error too large"

    rel = test_steady_limit()
    print(f"steady-state limit vs v0.7 solver: rel diff = {rel:.3e}")
    assert rel < 1e-9

    rel = test_robin_steady()
    print(f"Robin steady 1D profile: rel diff = {rel:.3e}")
    assert rel < 1e-9

    out = test_neumann_energy()
    for method, (drift, E_end) in out.items():
        print(f"Neumann energy [{method}]: drift = {drift:.3e}, "
              f"E(1) = {E_end:.9f} (exact 28)")
        assert drift < 1e-10 and abs(E_end - 28.0) < 1e-7

    out = test_adiabatic()
    for method, res in out.items():
        drift, over, under = res[0.02]
        print(f"adiabatic [{method}] dt=0.02: drift = {drift:.3e}, "
              f"overshoot = {over:.3e}, undershoot = {under:.3e}")
        assert drift < 1e-10 and over <= 1e-9 and under <= 1e-9
        drift5, over5, under5 = res[0.005]
        print(f"adiabatic [{method}] dt=0.005: drift = {drift5:.3e}, "
              f"undershoot = {under5:.3e} (stated tol 5 K, see docstring)")
        assert drift5 < 1e-10 and over5 <= 1e-9 and under5 <= 5.0

    orders = test_temporal_order()
    for method, lo, hi in (("be", 0.8, 1.2), ("cn", 1.8, 2.2)):
        print(f"temporal order [{method}]: "
              f"{[f'{o:.2f}' for o in orders[method]]} (expect ~{lo + 0.2:.0f})")
        assert all(lo < o < hi for o in orders[method]), \
            f"{method} temporal order off: {orders[method]}"

    out = test_chain_uniform_source()
    for method, (err_T, err_s, vm, um) in out.items():
        print(f"chain [{method}]: T err = {err_T:.3e}, "
              f"stress rel err = {err_s:.3e}, vm_max = {vm:.3e} Pa, "
              f"u_max = {um:.3e} m")
        assert err_T < 1e-10 and err_s < 1e-9
        assert vm < 1e-5 and um < 1e-15

    print("ALL TRANSIENT 3D TESTS PASSED")


if __name__ == "__main__":
    main()
