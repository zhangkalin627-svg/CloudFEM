"""v0.8 demo: 3D thermal stress in a heated steel cube, base fixed.

Sequential heat -> structural coupling on tetrahedral P1 elements:
steady conduction with a uniform volumetric source (all faces held at
20 C) gives a non-uniform temperature field; that field drives a
thermal-stress solve with the bottom face (z = 0) fully clamped.

Sanity: the peak von Mises stress must stay below the fully
constrained scale E*alpha*dT_max (a free surface relieves the
hydrostatic part), and the top face must bulge upward/outward.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson
from cloudfem.physics.thermomech import thermal_stress_3d


def main(nx=12):
    L = 0.1            # m
    k = 50.0           # W/m/K, steel
    Q = 2e7            # W/m^3
    T0 = 20.0          # C, all faces / stress-free temperature
    E = 200e9          # Pa
    nu = 0.3
    alpha = 12e-6      # 1/K

    mesh = box_mesh(nx, nx, nx,
                    xlim=(0, L), ylim=(0, L), zlim=(0, L))

    # --- thermal: steady conduction, uniform source, faces at T0 ---
    faces = {n: T0 for n in
             ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    T = solve_tet_poisson(mesh, k=k, f=Q, bc=faces)
    dT_max = float(T.max() - T0)
    slab_bound = Q * L ** 2 / (8 * k)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")
    print(f"peak temperature rise: {dT_max:.2f} C "
          f"(1D-slab upper bound: {slab_bound:.2f} C)")
    assert 0.0 < dT_max < slab_bound, "unphysical temperature field"

    # --- structural: thermal stress, bottom face fully clamped ---
    res = thermal_stress_3d(mesh, T, E, nu, alpha, T_ref=T0,
                            bc={"zmin": (0.0, 0.0, 0.0)})
    vm_max = res["vm_max"]
    u_max = res["u_max"]
    bound = E * alpha * dT_max  # fully constrained stress scale
    print(f"max von Mises stress: {vm_max/1e6:.1f} MPa "
          f"(E*alpha*dT_max scale: {bound/1e6:.1f} MPa)")
    print(f"max displacement: {u_max*1e3:.4f} mm "
          f"(free-expansion scale alpha*dT_max*L: "
          f"{alpha*dT_max*L*1e3:.4f} mm)")
    assert 0.0 < vm_max < bound, "von Mises stress out of sanity range"
    z = mesh.nodes[:, 2]
    top = z > L - 1e-12
    assert res["uz"][top].mean() > 0.0, "top face must expand upward"

    # --- plots: T slice, von Mises slice, 3D nodal von Mises scatter ---
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    cent = mesh.nodes[mesh.elements].mean(axis=1)
    fig = plt.figure(figsize=(14, 4))

    ax = fig.add_subplot(131)
    m = np.abs(z - L / 2) < 0.51 * (L / nx)
    sc = ax.tricontourf(x[m], y[m], T[m] - T0, levels=20)
    ax.set_aspect("equal")
    ax.set_title("temperature rise, z = L/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(sc, ax=ax, label="T rise (C)")

    ax = fig.add_subplot(132)
    mc = np.abs(cent[:, 2] - L / 2) < 0.51 * (L / nx)
    sc = ax.tricontourf(cent[mc, 0], cent[mc, 1],
                        res["vm_elem"][mc] / 1e6, levels=20)
    ax.set_aspect("equal")
    ax.set_title("von Mises stress, z = L/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(sc, ax=ax, label="von Mises (MPa)")

    ax = fig.add_subplot(133, projection="3d")
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 4000)]
    sc = ax.scatter(x[sel], y[sel], z[sel],
                    c=res["vm_nodal"][sel] / 1e6, s=3, cmap="hot")
    ax.set_title("3D von Mises field (nodal)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label="von Mises (MPa)", shrink=0.6)

    fig.suptitle("CloudFEM v0.8: 3D thermal stress, heated steel cube, "
                 "base clamped")
    fig.tight_layout()
    fig.savefig("thermal_stress_3d.png", dpi=110)
    print("saved thermal_stress_3d.png")


if __name__ == "__main__":
    main()
