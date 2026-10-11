"""Axisymmetric induction heating: steel billet in a solenoid coil.

Geometry (r-z plane, metres):
  domain  : r in [0, 0.06], z in [0, 0.12]
  billet  : r in [0, 0.02], z in [0.02, 0.10], steel
  coil    : r in [0.025, 0.035], z in [0.03, 0.09], J_phi = 8e6 A/m^2
  air     : elsewhere

Coupling: sigma(T) = sigma0/(1+alpha*(T-T0)) updated explicitly each
time step (quasi-iterative). Thermal BC is adiabatic, so the energy
check is  integral(rho*cp*dT) == integral(P(t)*dt).

Outputs: PNG plots + numbers.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, joule_heating_density_axi,
    total_joule_power_axi, transient_heat_axi, MU0, TWO_PI,
    source_load_vector_axi)
from cloudfem.fem.timestep import element_to_nodal
from cloudfem.physics.em import skin_depth

FREQ = 10e3
OMEGA = 2 * np.pi * FREQ
SIGMA0 = 2e6
ALPHA_SIG = 0.004
J_COIL = 8e6
T0 = 20.0


def is_billet(x, y):
    return x <= 0.02 and 0.02 <= y <= 0.10


def is_coil(x, y):
    return 0.025 <= x <= 0.035 and 0.03 <= y <= 0.09


def main():
    t_start = time.time()
    print(f"frequency  : {FREQ/1e3:.0f} kHz, "
          f"skin depth : {skin_depth(SIGMA0, OMEGA)*1e3:.2f} mm (at 20 C)")

    mesh = rect_mesh(120, 240, xlim=(0, 0.06), ylim=(0, 0.12))
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    J_fn = lambda x, y: J_COIL if is_coil(x, y) else 0.0          # noqa: E731
    k_fn = lambda x, y: 40.0 if is_billet(x, y) else (           # noqa: E731
        400.0 if is_coil(x, y) else 0.026)
    rc_fn = lambda x, y: (3.5e6 if is_billet(x, y) else                # noqa: E731
                         (3.44e6 if is_coil(x, y) else 1200.0))

    bc_em = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    T = np.full(mesh.n_nodes, T0)
    dt, t_end = 0.5, 10.0
    n_steps = int(t_end / dt)

    # axisymmetric-weighted thermal operators (built once)
    from cloudfem.fem.timestep import implicit_euler_heat
    k_axi = lambda x, y: TWO_PI * x * k_fn(x, y)                 # noqa: E731
    rc_axi = lambda x, y: TWO_PI * x * rc_fn(x, y)               # noqa: E731

    times, T_hist = [0.0], [T.copy()]
    P_hist = []

    for step in range(n_steps):
        Te = T[mesh.elements].mean(axis=1)  # element temperatures
        sig_elem = np.zeros(mesh.n_elements)
        billet_elem = np.array([
            is_billet(*mesh.nodes[el].mean(axis=0))
            for el in mesh.elements])
        sig_elem[billet_elem] = (
            SIGMA0 / (1 + ALPHA_SIG * (Te[billet_elem] - T0)))

        def sigma_call(x, y):
            # piecewise-constant per element via nearest centroid search
            # is too slow; instead exploit structured grid:
            ir = min(int(x / 0.06 * 120), 119)
            iz = min(int(y / 0.12 * 240), 239)
            e = (iz * 120 + ir) * 2  # first triangle of the cell
            return sig_elem[e] if e < len(sig_elem) else 0.0

        u = solve_eddy_current_axi(
            mesh, nu=1.0 / MU0, sigma=sigma_call, omega=OMEGA,
            J_phi=J_fn, bc=bc_em)
        Q_elem, _ = joule_heating_density_axi(mesh, u, sigma_call, OMEGA)
        P = total_joule_power_axi(mesh, u, sigma_call, OMEGA)
        P_hist.append(P)
        Q_nodal = element_to_nodal(mesh, Q_elem)
        rhs_q = source_load_vector_axi(mesh, Q_nodal)

        _, hist = implicit_euler_heat(mesh, k_axi, rc_axi, rhs_q,
                                      None, dt, dt, T)
        T = hist[-1]
        times.append((step + 1) * dt)
        T_hist.append(T.copy())
        print(f"  step {step+1:2d}/{n_steps}: P={P:7.1f} W, "
              f"Tmax={T.max():6.1f} C, sigma_min={sig_elem[billet_elem].min():.3e}",
              flush=True)

    times = np.array(times)
    T_hist = np.array(T_hist)
    P_hist = np.array(P_hist)

    # ---- energy check: dE_thermal vs integral(P dt) ----
    areas = mesh.element_areas()
    rc = np.array([mesh.nodes[el][:, 0].mean() for el in mesh.elements])
    dTe = (T_hist[-1] - T0)[mesh.elements].mean(axis=1)
    rho = np.array([rc_fn(*mesh.nodes[el].mean(axis=0))
                    for el in mesh.elements])
    E_gain = float(np.sum(rho * dTe * TWO_PI * rc * areas))
    E_in = float(np.sum(P_hist) * dt)
    print(f"thermal energy gain : {E_gain:.1f} J")
    print(f"integral(P dt)      : {E_in:.1f} J")
    print(f"energy error        : {abs(E_gain-E_in)/E_in*100:.2f} %")
    print(f"wall time           : {time.time()-t_start:.0f} s")

    # ---- plots ----
    nr, nz = 121, 241
    rr = mesh.nodes[:, 0].reshape(nz, nr)
    zz = mesh.nodes[:, 1].reshape(nz, nr)

    Q_plot = element_to_nodal(mesh, Q_elem).reshape(nz, nr)
    T_plot = T_hist[-1].reshape(nz, nr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    im0 = axes[0].pcolormesh(rr, zz, np.log10(Q_plot + 1e-12),
                             shading="auto", cmap="inferno")
    axes[0].set_title("log10 Joule heating [W/m^3]")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(rr, zz, T_plot, shading="auto", cmap="hot")
    axes[1].set_title(f"billet temperature after {t_end:.0f} s [C]")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
    fig.tight_layout()
    fig.savefig("induction_axi_fields.png", dpi=110)
    print("saved induction_axi_fields.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    billet_nodes = np.array([is_billet(*n) for n in mesh.nodes])
    ax2.plot(times, T_hist[:, billet_nodes].max(axis=1), label="billet max")
    ax2.plot(times, T_hist[:, billet_nodes].mean(axis=1), label="billet mean")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("T [C]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("induction_axi_temp.png", dpi=110)
    print("saved induction_axi_temp.png")


if __name__ == "__main__":
    main()
