"""v1.4 flagship demo: 3D induction heating of a steel billet,
full EM -> thermal -> structural chain with sigma(T).

A solid steel cylinder (R = 20 mm, L = 80 mm) in a uniform axial
AC field H0 = 180 kA/m at 2.5 kHz is heated from 20 C for 12 s,
until the surface approaches the Curie temperature (~770 C) while
the core still lags several hundred degrees behind -- the classic
induction profile, helped by sigma(T) itself: as the skin heats
its conductivity falls (2.0e6 -> ~0.6e6 S/m here), the skin depth
grows and the deposition deepens, taming the surface peak. The driver
(physics/induction3d.py) re-solves the v1.3 edge-element eddy
currents every 2 s of heat time with sigma updated from the
element temperatures, sigma(T) = 2e6 / (1 + 0.004 (T - 20)) S/m --
the 2D axisymmetric chain's law -- feeds the Joule field into the
v1.0 transient heat solver on the billet (adiabatic surface), and
samples the v0.8 thermal-stress chain along the way (free body,
3-2-1 rigid pins).

Model honesty (see physics/induction3d.py docstring): mu = mu0
throughout, so "approaching Curie" is a statement about the
temperature level only -- the magnetic Curie transition is not
modelled (same convention as the 2D reference chain). The mesh is
the v1.3 cylinder-in-a-box mesh at h = 0.011 (h/delta ~ 1.55 at
20 C), whose total loss runs ~75 % above the axisymmetric value at
this working point (end effect + skin resolution, measured in
tests/test_induction3d.py); absolute powers inherit that error,
the temperature *shapes* and the sigma(T) feedback do not depend
on it. The discrete energy account (EM energy in vs enthalpy
gain) closes at round-off level and is printed.

Runtime: 6 EM solves of ~40 s each on 2 cores; expect ~5 min on
an idle VM (longer under load).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0, skin_depth
from cloudfem.physics.induction3d import induction_heating_3d

R, L = 0.02, 0.08
SIGMA0, ALPHA, FREQ, H0 = 2e6, 0.004, 2500.0, 180e3
K_TH, RHO_CP, T0 = 40.0, 3.5e6, 20.0
T_END, DT, EM_EVERY = 12.0, 0.5, 4


def main(h=0.011):
    delta0 = skin_depth(SIGMA0, 2 * np.pi * FREQ)
    print(f"skin depth at 20 C: {delta0 * 1e3:.1f} mm "
          f"(R/delta = {R / delta0:.1f}, h/delta = {h / delta0:.2f})")
    mesh = cylinder_in_box_mesh(R, L, half=0.06, h=h)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")

    def sigma_ref(x, y, z):
        return SIGMA0 if (x * x + y * y < R * R
                          and abs(z) < L / 2) else 0.0

    res = induction_heating_3d(
        mesh, sigma_ref, omega=2 * np.pi * FREQ,
        B_applied=(0.0, 0.0, MU0 * H0), alpha_sig=ALPHA,
        k=K_TH, rho_cp=RHO_CP, T0=T0, t_end=T_END, dt=DT,
        em_every=EM_EVERY, method="cn",
        E=200e9, nu_p=0.3, alpha_th=12e-6, stress_every=4)
    wp = res["workpiece"]
    times, hist = res["times"], res["T_history"]
    print(f"EM solves: {res['n_em_solves']} at t = {res['em_times']} s")
    print(f"P: {res['P_em'][0]:.0f} -> {res['P_em'][-1]:.0f} W; "
          f"sigma_min: {res['sigma_min'][0]:.3e} -> "
          f"{res['sigma_min'][-1]:.3e} S/m")
    print(f"energy account: E_in = {res['E_in']:.1f} J, enthalpy "
          f"gain = {res['enthalpy_gain']:.1f} J, rel err = "
          f"{res['energy_rel_err']:.1e}")
    T_end = hist[-1]
    print(f"final T: mean {res['T_mean'][-1]:.0f} C, "
          f"range [{T_end.min():.0f}, {T_end.max():.0f}] C")
    print(f"peak von Mises: {res['peak_vm_max'] / 1e6:.0f} MPa at "
          f"t = {res['peak_time']:.0f} s")
    assert res["energy_rel_err"] < 1e-6
    assert T_end.max() > 600.0, "surface should approach Curie range"
    assert T_end.min() < res["T_mean"][-1] < T_end.max()

    # --- plots ----------------------------------------------------
    n = wp.nodes
    r = np.linalg.norm(n[:, :2], axis=1)
    band = np.where(np.abs(n[:, 2]) <= 0.006)[0]
    i_core = int(band[np.argmin(r[band])])
    i_surf = int(band[np.argmax(r[band])])

    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_subplot(131)
    ax.plot(times, hist[:, i_surf], label="surface (mid-plane)")
    ax.plot(times, hist[:, i_core], label="core (axis, mid-plane)")
    ax.plot(times, res["T_mean"], label="volume mean")
    ax.plot(times, hist.max(axis=1), "--", label="hottest node "
            "(end rim)")
    ax.axhline(770.0, color="grey", lw=0.8, ls=":")
    ax.text(0.4, 782, "Curie ~770 C", color="grey", fontsize=8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (C)")
    ax.set_title("billet temperatures, sigma(T) chain")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(132)
    ax.step(res["em_times"], res["P_em"] / 1e3, where="post",
            label="total Joule power")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P (kW)")
    ax2 = ax.twinx()
    ax2.plot(res["em_times"], res["sigma_min"] / 1e6, "o-",
             color="tab:red", ms=3, label="sigma_min")
    ax2.set_ylabel("sigma_min (MS/m)", color="tab:red")
    ax.set_title("power and hottest-element conductivity")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(133)
    cen = wp.nodes[wp.elements].mean(axis=1)
    pk = res["peak"]
    sc = ax.scatter(cen[:, 2] * 1e3, cen[:, 0] * 1e3,
                    c=pk["vm_elem"] / 1e6, s=6, cmap="inferno")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title(f"von Mises at peak (t = {res['peak_time']:.0f} s, "
                 f"{res['peak_vm_max'] / 1e6:.0f} MPa)")
    plt.colorbar(sc, ax=ax, label="MPa")

    fig.suptitle("CloudFEM v1.4: 3D induction heating, "
                 "EM -> thermal -> structural with sigma(T)")
    fig.tight_layout()
    fig.savefig("induction_heating_3d.png", dpi=110)
    print("saved induction_heating_3d.png")


if __name__ == "__main__":
    main()
