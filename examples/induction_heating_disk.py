"""Planar induction heating of a circular steel billet (unstructured mesh).

Geometry: disk domain radius 60 mm.
  - billet : r < 20 mm, steel, sigma = 2e6 S/m
  - coil   : two copper bars, x in [25,35] mm and [-35,-25] mm,
             |y| < 20 mm, Jz = +/-5e6 A/m^2
  - air    : elsewhere

EM: time-harmonic A-formulation at 10 kHz, A = 0 on outer boundary.
Thermal: transient, adiabatic outer boundary (energy check).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from cloudfem.mesh.unstructured import disk_mesh
from cloudfem.physics.em import (
    solve_eddy_current, joule_heating_density, total_joule_power,
    skin_depth, MU0)
from cloudfem.physics.coupled import induction_heating

FREQ = 10e3
OMEGA = 2 * np.pi * FREQ
SIGMA_STEEL = 2e6
J0 = 5e6
T0 = 20.0


def r_of(x, y):
    return np.hypot(x, y)


def is_billet(x, y):
    return r_of(x, y) < 0.02


def is_coil_pos(x, y):
    return (0.025 <= x <= 0.035) and abs(y) < 0.02


def is_coil_neg(x, y):
    return (-0.035 <= x <= -0.025) and abs(y) < 0.02


def sigma_fn(x, y):
    return SIGMA_STEEL if is_billet(x, y) else 0.0


def js_fn(x, y):
    if is_coil_pos(x, y):
        return J0
    if is_coil_neg(x, y):
        return -J0
    return 0.0


def k_fn(x, y):
    if is_billet(x, y):
        return 40.0
    if is_coil_pos(x, y) or is_coil_neg(x, y):
        return 400.0
    return 0.026


def rho_cp_fn(x, y):
    if is_billet(x, y):
        return 3.5e6
    if is_coil_pos(x, y) or is_coil_neg(x, y):
        return 3.44e6
    return 1200.0


def main():
    print(f"skin depth: {skin_depth(SIGMA_STEEL, OMEGA)*1e3:.2f} mm")
    mesh = disk_mesh(0.06, 0.002)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    res = induction_heating(
        mesh, sigma=sigma_fn, omega=OMEGA, J_s=js_fn,
        k_th=k_fn, rho_cp=rho_cp_fn,
        bc_em={"boundary": 0.0}, bc_th=None,
        t_end=10.0, dt=0.25, T0=T0, depth=1.0,
    )
    P = res["P_total"]
    print(f"total Joule power: {P:.1f} W/m")

    T_end = res["T_history"][-1]
    areas = mesh.element_areas()
    rc = np.array([rho_cp_fn(*mesh.nodes[el].mean(axis=0))
                   for el in mesh.elements])
    dTe = (T_end - T0)[mesh.elements].mean(axis=1)
    E_gain = float(np.sum(rc * dTe * areas))
    E_expected = P * res["times"][-1]
    print(f"energy error: {abs(E_gain-E_expected)/E_expected*100:.2f} %")
    print(f"max temperature: {T_end.max():.1f} C")

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    Qp = res["Q_nodal"]
    im0 = axes[0].tripcolor(tri, np.log10(Qp + 1e-12), cmap="inferno",
                            shading="gouraud")
    axes[0].set_title("log10 Joule heating [W/m^3]")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].tripcolor(tri, T_end, cmap="hot", shading="gouraud")
    axes[1].set_title(f"temperature after {res['times'][-1]:.0f} s [C]")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig("induction_disk_fields.png", dpi=110)
    print("saved induction_disk_fields.png")


if __name__ == "__main__":
    main()
