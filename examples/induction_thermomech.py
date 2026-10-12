"""Induction heating -> thermal stress of a circular steel billet.

Full chain on an unstructured disk mesh (billet + coil + air):
  1. time-harmonic eddy currents at 10 kHz -> Joule heating,
  2. transient heat conduction (10 s),
  3. thermal stress of the billet from the final temperature field.

The mechanical solve runs on a billet-only sub-mesh (r < 20 mm) with
steel properties, plane stress, free outer boundary; rigid-body motion
is removed by pinning the centre node and one node on the rim
(reactions are ~0 because the thermal load is self-equilibrated).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator

from cloudfem.mesh.unstructured import disk_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.coupled import induction_heating
from cloudfem.physics.thermomech import thermal_stress

FREQ = 10e3
OMEGA = 2 * np.pi * FREQ
SIGMA_STEEL = 2e6
J0 = 5e6
T0 = 20.0
R_BILLET = 0.02

# steel mechanical properties
E_STEEL = 200e9
NU_STEEL = 0.3
ALPHA_STEEL = 12e-6


def r_of(x, y):
    return np.hypot(x, y)


def is_billet(x, y):
    return r_of(x, y) < R_BILLET


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
    mesh = disk_mesh(0.06, 0.002)
    print(f"EM/thermal mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

    res = induction_heating(
        mesh, sigma=sigma_fn, omega=OMEGA, J_s=js_fn,
        k_th=k_fn, rho_cp=rho_cp_fn,
        bc_em={"boundary": 0.0}, bc_th=None,
        t_end=10.0, dt=0.25, T0=T0, depth=1.0,
    )
    T_end = res["T_history"][-1]
    print(f"max temperature after 10 s: {T_end.max():.1f} C")

    # billet-only mechanical mesh; interpolate the temperature onto it
    bmesh = disk_mesh(R_BILLET, 0.001)
    print(f"billet mesh: {bmesh.n_nodes} nodes, {bmesh.n_elements} elements")
    interp = LinearNDInterpolator(mesh.nodes, T_end)
    Tb = interp(bmesh.nodes[:, 0], bmesh.nodes[:, 1])

    centre = int(np.argmin(np.hypot(bmesh.nodes[:, 0], bmesh.nodes[:, 1])))
    rim = int(np.argmin(np.abs(np.hypot(bmesh.nodes[:, 0], bmesh.nodes[:, 1])
                               - R_BILLET) + np.abs(bmesh.nodes[:, 1])))
    bmesh.boundaries["pin_centre"] = np.array([centre])
    bmesh.boundaries["pin_rim"] = np.array([rim])
    mech = thermal_stress(
        bmesh, Tb, E_STEEL, NU_STEEL, ALPHA_STEEL, T_ref=T0,
        plane="stress",
        bc={"pin_centre": (0.0, 0.0), "pin_rim": (None, 0.0)},
    )
    print(f"max von Mises stress: {mech['vm_max']/1e6:.2f} MPa")
    print(f"max displacement: {mech['u_max']*1e6:.2f} um")

    tri = Triangulation(bmesh.nodes[:, 0], bmesh.nodes[:, 1], bmesh.elements)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    im0 = axes[0].tripcolor(tri, Tb, cmap="hot", shading="gouraud")
    axes[0].set_title("billet temperature after 10 s [C]")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].tripcolor(tri, mech["vm_nodal"] / 1e6, cmap="jet",
                            shading="gouraud")
    axes[1].set_title("von Mises stress [MPa]")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig("induction_thermomech_vm.png", dpi=110)
    print("saved induction_thermomech_vm.png")


if __name__ == "__main__":
    main()
