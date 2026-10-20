"""v1.3 demo: 3D eddy currents in a steel billet + EM -> heat chain.

A solid steel cylinder (radius 20 mm, length 80 mm, sigma = 1e6 S/m)
sits in a uniform axial AC field H0 = 20 kA/m at 2.5 kHz (skin depth
delta = 10.1 mm, R/delta = 2.0), solved with the v1.3 Nedelec
edge-element A-V formulation on a cylinder-in-a-box mesh. The
element Joule-heating field Q = 0.5*sigma*|E|^2 is then fed into the
v1.0 transient heat solver on the extracted cylinder submesh
(insulated surface, Crank-Nicolson), closing the 3D EM -> thermal
half of the flagship chain (the temperature history can drive the
v0.8/v1.0 thermal-stress chain exactly as in
examples/transient_thermomech_3d.py).

Sanity: the heating must peak in the lateral skin layer and at the
two end faces (the classic billet end effect), the total Joule power
must match the energy that appears as heat, and the surface must run
hotter than the axis throughout the transient.

The finite cylinder in an applied field is also the v1.3
cross-validation configuration against the v0.2 axisymmetric solver;
see tests/test_em3d.py for the measured agreement and its
end-effect discussion.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from cloudfem.mesh.tet_mesh import TetMesh
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.fem.tet_edge import build_edges
from cloudfem.fem.tet_heat import transient_heat_tet
from cloudfem.physics.em import MU0, skin_depth
from cloudfem.physics.em3d import (solve_eddy_current_3d,
                                   joule_heating_density_3d)


def _submesh(mesh, mask):
    """Extract the elements selected by `mask` as a standalone TetMesh."""
    els = mesh.elements[mask]
    used, inv = np.unique(els, return_inverse=True)
    return TetMesh(mesh.nodes[used], inv.reshape(-1, 4)), used


def main(h=0.011):
    R, L = 0.02, 0.08            # cylinder radius / length [m]
    sigma = 1e6                  # S/m (hot-steel order of magnitude)
    H0 = 20e3                    # A/m, applied axial field amplitude
    freq = 2500.0                # Hz
    omega = 2 * np.pi * freq
    delta = skin_depth(sigma, omega)
    print(f"skin depth = {delta * 1e3:.2f} mm (R/delta = {R / delta:.2f})")

    mesh = cylinder_in_box_mesh(R, L, half=0.06, h=h)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")

    def sigma_fn(x, y, z):
        return sigma if (x * x + y * y < R * R and abs(z) < L / 2) else 0.0

    ed = build_edges(mesh)
    a, v = solve_eddy_current_3d(mesh, nu=1.0 / MU0, sigma=sigma_fn,
                                 omega=omega,
                                 B_applied=(0.0, 0.0, MU0 * H0),
                                 edge_data=ed)
    Q, vols = joule_heating_density_3d(mesh, a, v, sigma_fn, omega,
                                       edge_data=ed)
    P = float(np.sum(Q * vols))
    print(f"total Joule power = {P:.1f} W")

    # --- EM -> heat: temperature history of the billet alone -------
    cen = mesh.nodes[mesh.elements].mean(axis=1)
    rr = np.linalg.norm(cen[:, :2], axis=1)
    in_cyl = (rr < R) & (np.abs(cen[:, 2]) < L / 2)
    billet, _ = _submesh(mesh, in_cyl)
    Q_b = Q[in_cyl]
    cen_b = billet.nodes[billet.elements].mean(axis=1)
    tree = cKDTree(cen_b)

    def heat_source(x, y, z):
        _, idx = tree.query([x, y, z])
        return float(Q_b[idx])

    k_th, rho_cp, T0 = 40.0, 7850.0 * 500.0, 20.0
    times, hist = transient_heat_tet(billet, k=k_th, rho_cp=rho_cp,
                                     f=heat_source, t_end=30.0, dt=1.0,
                                     T0=T0, method="cn")
    T_end = hist[-1]
    dT_mean = P * times[-1] / (rho_cp * np.pi * R ** 2 * L)
    print(f"after {times[-1]:.0f} s: T in [{T_end.min():.1f}, "
          f"{T_end.max():.1f}] C; adiabatic mean rise = {dT_mean:.1f} K")

    # --- plots ------------------------------------------------------
    fig = plt.figure(figsize=(13, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    sc = ax.scatter(cen_b[:, 2] * 1e3, cen_b[:, 0] * 1e3,
                    c=Q_b / 1e6, s=6, cmap="inferno")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title("Joule heating Q (MW/m$^3$), element centroids")
    plt.colorbar(sc, ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    Tn = billet.nodes
    sc = ax.scatter(Tn[:, 2] * 1e3, Tn[:, 0] * 1e3, c=T_end, s=6,
                    cmap="inferno")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title(f"temperature after {times[-1]:.0f} s (C)")
    plt.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig("eddy_current_3d.png", dpi=110)
    print("wrote eddy_current_3d.png")


if __name__ == "__main__":
    main()
