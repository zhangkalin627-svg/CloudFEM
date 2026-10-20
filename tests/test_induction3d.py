"""3D induction-heating flagship chain (v1.4) validation.

Configuration shared by the 3D chain and the 2D reference: solid
steel cylinder R = 20 mm, L = 80 mm in a uniform axial AC field
H0 = 60 kA/m at 2.5 kHz, sigma(T) = 2e6 / (1 + 0.004 (T - 20))
(the 2D chain's law), k = 40 W/m/K, rho_cp = 3.5e6 J/m^3/K,
adiabatic workpiece, T0 = 20 C. At 20 C the skin depth is
delta = 7.1 mm (R/delta = 2.8); the 3D mesh (cylinder-in-box,
h = 0.011, as in tests/test_em3d.py) has h/delta ~ 1.55, so the
3D loss inherits the v1.3 skin-resolution + end-effect error --
the cross-check below is built around that, not around pretending
the two discretisations are the same problem.

Checks:
1. sigma(T) law and the discrete enthalpy helper (exact values).
2. Workpiece extraction: faceted volume vs analytic (O(h^2) low).
3. Energy account of the coupled run: accumulated EM energy vs the
   enthalpy gain of the final temperature field. With the source
   frozen per EM interval and an adiabatic workpiece this closes
   identically (see physics/induction3d.py docstring); the assert
   guards the Q transfer between meshes.
4. Chain physics: power positive, surface hotter than core at the
   mid-plane, zero stress at t = 0 (uniform T = T_ref), positive
   peak thermal stress later.
5. sigma(T) on/off, two regimes. Mild point (the shared run,
   H0 = 60 kA/m, 6 s): most of the volume is still cold, so the
   mean-rise difference is small (+0.09 %, measured) while the
   hottest elements' sigma drops 15 % and the power drifts *up*
   +0.26 % (surface-resistance side of the turnover; the 2D chain
   shows the same sign). Hot point (H0 = 150 kA/m, 4.5 s, separate
   short runs): the surface reaches ~340 C, sigma_min drops 44 %,
   the skin depth grows ~35 %, and the chain crosses the turnover
   -- the power now *declines* ~1 % as sigma falls, the peak
   temperature is tamed (342 vs 402 C with sigma frozen: the hot
   skin turns transparent and the deposition deepens), the coldest
   point runs warmer (80 vs 74 C), at nearly equal mean rise
   (-0.55 %). The assertions encode exactly this documented,
   regime-dependent behaviour.
6. Cross-check vs an independent 2D axisymmetric chain (library 2D
   solvers only, background-field excitation as in test_em3d.py):
   the 3D/axi mean-temperature-rise ratio must track the cold
   power ratio (end effect + skin resolution, measured by the same
   two EM solvers at the working point) corrected by the
   mesh-volume ratio. The pointwise mid-plane ratios bracket that
   power ratio -- the surface above it (the under-resolved 3D skin
   loss sits in the surface layer, and the end effect adds more),
   the core below it at early times (it heats by conduction only;
   measured at t = 3 s: surface 2.07, mean 1.81, core 1.29 against
   a power ratio of 1.75) -- and both must lie within
   [0.75, 1.45] x the power ratio.
"""

import numpy as np
from scipy.spatial import cKDTree

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.fem.timestep import (implicit_euler_heat,
                                   source_load_vector_elem)
from cloudfem.fem.tet_heat import assemble_mass_tet
from cloudfem.physics.axisym import (TWO_PI, joule_heating_density_axi,
                                     solve_eddy_current_axi,
                                     total_joule_power_axi)
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d import (extract_submesh,
                                          induction_heating_3d,
                                          steel_sigma_of_T,
                                          thermal_enthalpy)

R, L = 0.02, 0.08
SIGMA0, ALPHA, FREQ, H0 = 2e6, 0.004, 2500.0, 60e3
K_TH, RHO_CP, T0 = 40.0, 3.5e6, 20.0
T_END, DT = 6.0, 0.5


def _sigma_ref(x, y, z):
    return SIGMA0 if (x * x + y * y < R * R and abs(z) < L / 2) else 0.0


def _mesh():
    return cylinder_in_box_mesh(R, L, half=0.06, h=0.011)


def _run_chain(mesh, alpha_sig):
    return induction_heating_3d(
        mesh, _sigma_ref, omega=2 * np.pi * FREQ,
        B_applied=(0.0, 0.0, MU0 * H0), alpha_sig=alpha_sig,
        k=K_TH, rho_cp=RHO_CP, T0=T0, t_end=T_END, dt=DT,
        em_every=3, method="cn",
        E=200e9, nu_p=0.3, alpha_th=12e-6, stress_every=3)


def test_sigma_law():
    assert steel_sigma_of_T(20.0, SIGMA0, ALPHA, 20.0) == SIGMA0
    assert abs(steel_sigma_of_T(270.0, SIGMA0, ALPHA, 20.0)
               - SIGMA0 / 2) < 1e-9 * SIGMA0
    arr = steel_sigma_of_T(np.array([20.0, 520.0]), SIGMA0, ALPHA, 20.0)
    assert abs(arr[1] - SIGMA0 / 3) < 1e-9 * SIGMA0
    assert steel_sigma_of_T(500.0, SIGMA0, 0.0, 20.0) == SIGMA0


def test_enthalpy_helper():
    mesh = _mesh()
    cen = mesh.nodes[mesh.elements].mean(axis=1)
    mask = (np.linalg.norm(cen[:, :2], axis=1) < R) \
        & (np.abs(cen[:, 2]) < L / 2)
    wp, used, idx = extract_submesh(mesh, mask)
    vol = float(wp.element_volumes().sum())
    vol_exact = np.pi * R ** 2 * L
    H = thermal_enthalpy(wp, np.full(wp.n_nodes, T0 + 100.0), RHO_CP,
                         T_ref=T0)
    H_exact = RHO_CP * vol * 100.0
    return {"vol": vol, "vol_exact": vol_exact,
            "vol_rel_err": (vol - vol_exact) / vol_exact,
            "H_rel_err": (H - H_exact) / H_exact,
            "n_elem": int(mask.sum())}


def axi_chain(t_end=T_END, dt=0.5, nr=32, nz=10):
    """Independent 2D axisymmetric reference chain (test-local).

    Infinite cylinder in an axial field H0 (the v1.3 cross-check
    configuration), explicit sigma(T) update every step, adiabatic
    implicit-Euler heat; built only from the library 2D solvers,
    mirroring examples/induction_heating_axi.py's loop. Returns
    histories of total power and of volume-mean / surface / core
    temperatures plus the mesh volume.
    """
    mesh = rect_mesh(nr, nz, xlim=(0, R), ylim=(0, L))
    omega = 2 * np.pi * FREQ
    cen = mesh.nodes[mesh.elements].mean(axis=1)
    tree = cKDTree(cen)
    vols = TWO_PI * cen[:, 0] * mesh.element_areas()
    state = {"sig_e": np.full(mesh.n_elements, SIGMA0)}

    def sigma_fn(x, y):
        _, idx = tree.query([x, y])
        return float(state["sig_e"][idx])

    k_axi = lambda x, y: TWO_PI * x * K_TH          # noqa: E731
    rc_axi = lambda x, y: TWO_PI * x * RHO_CP       # noqa: E731
    r_node = mesh.nodes[:, 0]
    i_surf = int(np.argmin(np.abs(r_node - R)))
    i_core = int(np.argmin(r_node))
    T = np.full(mesh.n_nodes, T0)
    n_steps = int(round(t_end / dt))
    T_mean, T_surf, T_core, P_hist = [T0], [T0], [T0], []
    for _ in range(n_steps):
        T_e = T[mesh.elements].mean(axis=1)
        state["sig_e"] = steel_sigma_of_T(T_e, SIGMA0, ALPHA, T0)
        u = solve_eddy_current_axi(mesh, nu=1.0 / MU0, sigma=sigma_fn,
                                   omega=omega, bc={"left": 0.0},
                                   neumann={"right": TWO_PI * H0})
        Q, _ = joule_heating_density_axi(mesh, u, sigma_fn, omega)
        P_hist.append(total_joule_power_axi(mesh, u, sigma_fn, omega))
        rhs_q = source_load_vector_elem(mesh, Q,
                                        weight=lambda r: TWO_PI * r)
        _, hist = implicit_euler_heat(mesh, k_axi, rc_axi, rhs_q,
                                      None, dt, dt, T)
        T = hist[-1]
        Te = T[mesh.elements].mean(axis=1)
        T_mean.append(float(np.sum(Te * vols) / np.sum(vols)))
        T_surf.append(float(T[i_surf]))
        T_core.append(float(T[i_core]))
    return {"T_mean": np.array(T_mean), "T_surf": np.array(T_surf),
            "T_core": np.array(T_core), "P": np.array(P_hist),
            "volume": float(np.sum(vols))}


def _midplane_nodes(wp, half_width=0.006):
    """(core, surface) node indices near the mid-plane |z| <= 6 mm:
    smallest / largest radius inside that band."""
    n = wp.nodes
    r = np.linalg.norm(n[:, :2], axis=1)
    band = np.where(np.abs(n[:, 2]) <= half_width)[0]
    core = int(band[np.argmin(r[band])])
    surf = int(band[np.argmax(r[band])])
    return core, surf


if __name__ == "__main__":
    test_sigma_law()
    print("[sigma law] sigma(T0)=sigma0, sigma(270C)=sigma0/2, "
          "sigma(520C)=sigma0/3: exact")

    eh = test_enthalpy_helper()
    print(f"[extract] workpiece {eh['n_elem']} tets, volume "
          f"{eh['vol']:.4e} vs analytic {eh['vol_exact']:.4e} "
          f"({eh['vol_rel_err']:+.2%}, faceting); enthalpy of uniform "
          f"+100 K field rel err {eh['H_rel_err']:.2e}")
    assert abs(eh["vol_rel_err"]) < 0.06
    assert abs(eh["H_rel_err"]) < 1e-12

    mesh = _mesh()
    res = _run_chain(mesh, ALPHA)
    print(f"[chain] EM solves: {res['n_em_solves']}, "
          f"P_em = {np.round(res['P_em'], 1)} W, "
          f"sigma_min = {np.round(res['sigma_min'], -3)} S/m")
    print(f"[chain] T_mean: {np.round(res['T_mean'], 2)} C")
    print(f"[energy] E_in = {res['E_in']:.2f} J, enthalpy gain = "
          f"{res['enthalpy_gain']:.2f} J, rel err = "
          f"{res['energy_rel_err']:.2e}")
    assert res["n_em_solves"] == 4
    assert np.all(res["P_em"] > 0)
    assert res["energy_rel_err"] < 5e-3, "energy account does not close"

    wp = res["workpiece"]
    i_core, i_surf = _midplane_nodes(wp)
    T_core_3d = float(res["T_history"][-1, i_core])
    T_surf_3d = float(res["T_history"][-1, i_surf])
    print(f"[chain] mid-plane final: core {T_core_3d:.1f} C, "
          f"surface {T_surf_3d:.1f} C, "
          f"peak vm = {res['peak_vm_max'] / 1e6:.1f} MPa at "
          f"t = {res['peak_time']:.1f} s; vm(t=0) = "
          f"{res['vm_max'][0]:.3e} Pa")
    assert T_surf_3d > T_core_3d, "surface must lead the core"
    assert res["vm_max"][0] < 1e-6 * res["peak_vm_max"], \
        "uniform T = T_ref must be stress-free"
    assert res["peak_vm_max"] > 0

    res_off = _run_chain(mesh, 0.0)
    d_mean = (res["T_mean"][-1] - res_off["T_mean"][-1]) \
        / (res_off["T_mean"][-1] - T0)
    print(f"[sigma(T) mild] off: P = {res_off['P_em'][0]:.1f} W "
          f"(1 solve), T_mean(end) = {res_off['T_mean'][-1]:.2f} C vs "
          f"on {res['T_mean'][-1]:.2f} C (mean-rise diff {d_mean:+.2%}); "
          f"P(on) drifts {res['P_em'][0]:.0f} -> {res['P_em'][-1]:.0f} W, "
          f"sigma_min -> {res['sigma_min'][-1]:.3e} S/m")
    assert res_off["n_em_solves"] == 1
    assert res["sigma_min"][-1] < 0.90 * SIGMA0, \
        "hot elements must lose sigma"
    assert abs(d_mean) < 0.01, "mild point: mean effect must be small"

    def _hot(alpha_sig):
        return induction_heating_3d(
            mesh, _sigma_ref, omega=2 * np.pi * FREQ,
            B_applied=(0.0, 0.0, MU0 * 150e3), alpha_sig=alpha_sig,
            k=K_TH, rho_cp=RHO_CP, T0=T0, t_end=4.5, dt=DT,
            em_every=3, method="cn")

    hot_on, hot_off = _hot(ALPHA), _hot(0.0)
    dTmax_on = hot_on["T_history"][-1].max() - T0
    dTmax_off = hot_off["T_history"][-1].max() - T0
    dTmin_on = hot_on["T_history"][-1].min() - T0
    dTmin_off = hot_off["T_history"][-1].min() - T0
    d_hot = (hot_on["T_mean"][-1] - hot_off["T_mean"][-1]) \
        / (hot_off["T_mean"][-1] - T0)
    print(f"[sigma(T) hot] P(on) {hot_on['P_em'][0]:.0f} -> "
          f"{hot_on['P_em'][-1]:.0f} W (declining), sigma_min -> "
          f"{hot_on['sigma_min'][-1]:.3e}; peak dT on/off = "
          f"{dTmax_on:.0f}/{dTmax_off:.0f} K, min dT = "
          f"{dTmin_on:.0f}/{dTmin_off:.0f} K, mean diff {d_hot:+.2%}")
    assert hot_on["P_em"][-1] < hot_on["P_em"][0], \
        "hot point: power must decline as sigma falls (turnover)"
    assert dTmax_on < 0.92 * dTmax_off, \
        "sigma(T) must tame the surface peak (deeper deposition)"
    assert dTmin_on > dTmin_off, "core must run warmer with sigma(T)"
    assert abs(d_hot) < 0.02

    ref = axi_chain()
    vol_wp = float(wp.element_volumes().sum())
    ratio_P = res["P_em"][0] / ref["P"][0]
    ratio_T = (res["T_mean"][-1] - T0) / (ref["T_mean"][-1] - T0)
    expect = ratio_P * ref["volume"] / vol_wp
    print(f"[axi x-check] P cold: 3D {res['P_em'][0]:.1f} W vs axi "
          f"{ref['P'][0]:.1f} W (ratio {ratio_P:.3f}); mean-rise "
          f"ratio {ratio_T:.3f} vs power/volume expectation "
          f"{expect:.3f} (dev {(ratio_T - expect) / expect:+.2%})")
    ratio_surf = (T_surf_3d - T0) / (ref["T_surf"][-1] - T0)
    ratio_core = (T_core_3d - T0) / (ref["T_core"][-1] - T0)
    print(f"[axi x-check] final T: axi surf {ref['T_surf'][-1]:.1f} / "
          f"core {ref['T_core'][-1]:.1f} C vs 3D mid-plane surf "
          f"{T_surf_3d:.1f} / core {T_core_3d:.1f} C "
          f"(dT ratios: surf {ratio_surf:.2f}, core {ratio_core:.2f} "
          f"vs power ratio {ratio_P:.2f})")
    assert ratio_P > 1.0, "3D loss must exceed axi (end effect)"
    assert abs(ratio_T - expect) / expect < 0.25
    assert T_surf_3d > ref["T_surf"][-1] and T_core_3d > ref["T_core"][-1]
    for ratio in (ratio_surf, ratio_core):
        assert 0.75 * ratio_P < ratio < 1.45 * ratio_P, \
            "pointwise dT ratio outside the documented bracket"

    print("ALL 3D INDUCTION-CHAIN VALIDATIONS PASSED")
