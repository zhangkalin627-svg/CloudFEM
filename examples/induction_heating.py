"""Induction heating showcase: steel billet inside a coil (2D cross-section).

Geometry (all in metres), domain [0, 0.12] x [0, 0.12]:
  - steel workpiece : [0.04, 0.08] x [0.04, 0.08], sigma = 2e6 S/m
  - coil bars       : [0.020, 0.030] x [0.04, 0.08], J = +J0
                      [0.090, 0.100] x [0.04, 0.08], J = -J0
  - air             : everywhere else, sigma = 0

EM: time-harmonic A-formulation at 10 kHz, A = 0 on outer boundary.
Thermal: transient, insulated outer boundary (adiabatic) so that the
energy check  integral(rho*cp*dT) == P_total * t_end  must hold.

Outputs: PNG plots + numbers printed to stdout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.em import (
    solve_eddy_current, joule_heating_density, total_joule_power,
    skin_depth, MU0)
from cloudfem.physics.coupled import induction_heating
from cloudfem.fem.timestep import element_to_nodal

# ---------------- parameters ----------------
LX = LY = 0.12
FREQ = 10e3
OMEGA = 2 * np.pi * FREQ
SIGMA_STEEL = 2e6
J0 = 5e6                     # A/m^2 in coil bars
K_STEEL, K_CU, K_AIR = 40.0, 400.0, 0.026
RHO_CP_STEEL = 3.5e6
RHO_CP_AIR = 1200.0
T0 = 20.0                    # degC


def in_box(x, y, x0, x1, y0, y1):
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def is_steel(x, y):
    return in_box(x, y, 0.04, 0.08, 0.04, 0.08)


def sigma_fn(x, y):
    return SIGMA_STEEL if is_steel(x, y) else 0.0


def js_fn(x, y):
    if in_box(x, y, 0.020, 0.030, 0.04, 0.08):
        return J0
    if in_box(x, y, 0.090, 0.100, 0.04, 0.08):
        return -J0
    return 0.0


def k_fn(x, y):
    if is_steel(x, y):
        return K_STEEL
    if in_box(x, y, 0.020, 0.030, 0.04, 0.08) or \
       in_box(x, y, 0.090, 0.100, 0.04, 0.08):
        return K_CU
    return K_AIR


def rho_cp_fn(x, y):
    return RHO_CP_STEEL if is_steel(x, y) else RHO_CP_AIR


def main():
    delta = skin_depth(SIGMA_STEEL, OMEGA)
    print(f"frequency      : {FREQ/1e3:.1f} kHz")
    print(f"skin depth     : {delta*1e3:.2f} mm")

    nx = ny = 150
    mesh = rect_mesh(nx, ny, xlim=(0, LX), ylim=(0, LY))
    print(f"mesh           : {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
          f"dx = {LX/nx*1e3:.2f} mm ({delta/(LX/nx):.1f} cells/skin-depth)")

    res = induction_heating(
        mesh, sigma=sigma_fn, omega=OMEGA, J_s=js_fn,
        k_th=k_fn, rho_cp=rho_cp_fn,
        bc_em={"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0},
        bc_th=None,                       # adiabatic -> energy check
        t_end=10.0, dt=0.2, T0=T0, depth=1.0,
    )
    P = res["P_total"]
    print(f"total Joule power: {P:.1f} W (per metre of billet length)")

    # ---- energy conservation check ----
    T_end = res["T_history"][-1]
    dT = T_end - T0
    areas = mesh.element_areas()
    rc = np.array([rho_cp_fn(*mesh.nodes[el].mean(axis=0))
                   for el in mesh.elements])
    dTe = dT[mesh.elements].mean(axis=1)
    E_gain = float(np.sum(rc * dTe * areas))       # J per metre
    E_expected = P * res["times"][-1]
    print(f"thermal energy gain : {E_gain:.1f} J/m")
    print(f"P * t               : {E_expected:.1f} J/m")
    print(f"energy error        : {abs(E_gain-E_expected)/E_expected*100:.2f} %")

    Q_nodal = res["Q_nodal"]
    steel = np.array([is_steel(*n) for n in mesh.nodes])
    print(f"max Q in steel      : {Q_nodal[steel].max():.3e} W/m^3")
    print(f"max temperature     : {T_end.max():.1f} C "
          f"(rise {T_end.max()-T0:.1f} K in {res['times'][-1]:.0f} s)")

    # ---- plots ----
    x = mesh.nodes[:, 0].reshape(ny + 1, nx + 1)
    y = mesh.nodes[:, 1].reshape(ny + 1, nx + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    Aabs = np.abs(res["A"]).reshape(ny + 1, nx + 1)
    im0 = axes[0].pcolormesh(x, y, Aabs, shading="auto")
    axes[0].set_title("|A| vector potential [Wb/m]")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    Qp = Q_nodal.reshape(ny + 1, nx + 1)
    im1 = axes[1].pcolormesh(x, y, np.log10(Qp + 1e-12), shading="auto",
                             cmap="inferno")
    axes[1].set_title("log10 Joule heating [W/m^3]")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    Tp = T_end.reshape(ny + 1, nx + 1)
    im2 = axes[2].pcolormesh(x, y, Tp, shading="auto", cmap="hot")
    axes[2].set_title(f"temperature after {res['times'][-1]:.0f} s [C]")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig("induction_fields.png", dpi=110)
    print("saved induction_fields.png")

    # temperature history at billet centre
    cx = np.argmin(np.abs(mesh.nodes[:, 0] - 0.06)
                   + np.abs(mesh.nodes[:, 1] - 0.06))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(res["times"], res["T_history"][:, cx], label="billet centre")
    ax2.plot(res["times"], res["T_history"].max(axis=1), label="max in domain")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("T [C]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("induction_temp_history.png", dpi=110)
    print("saved induction_temp_history.png")


if __name__ == "__main__":
    main()
