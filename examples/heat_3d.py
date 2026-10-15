"""v0.7 demo: steady 3D heat conduction in a steel cube.

Unit steel cube (k = 50 W/m/K) with uniform volumetric heat source,
all six faces held at 20 C. Reports the centre temperature and saves
two mid-plane contour slices plus a 3D node-coloured scatter.

Sanity: for a 1D slab of the same thickness the centre rise would be
Q*L^2/(8k); the cube must stay below that (extra side losses).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson


def main(nx=16):
    L = 0.1          # m
    k = 50.0         # W/m/K, steel
    Q = 1e6          # W/m^3
    T0 = 20.0        # C, all faces

    mesh = box_mesh(nx, nx, nx,
                    xlim=(0, L), ylim=(0, L), zlim=(0, L))
    faces = {n: T0 for n in
             ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    T = solve_tet_poisson(mesh, k=k, f=Q, bc=faces)

    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    c = np.argmin((x - L / 2) ** 2 + (y - L / 2) ** 2 + (z - L / 2) ** 2)
    T_centre = T[c] - T0
    slab_bound = Q * L ** 2 / (8 * k)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")
    print(f"centre temperature rise: {T_centre:.3f} C "
          f"(1D-slab upper bound: {slab_bound:.3f} C)")
    assert 0.0 < T_centre < slab_bound, "unphysical centre temperature"
    assert T.min() >= T0 - 1e-9 and T.max() <= T0 + slab_bound + 1e-9

    fig = plt.figure(figsize=(14, 4))

    # slice z = L/2
    ax = fig.add_subplot(131)
    m = np.abs(z - L / 2) < 0.51 * (L / nx)
    sc = ax.tricontourf(x[m], y[m], T[m] - T0, levels=20)
    ax.set_aspect("equal")
    ax.set_title("mid-plane z = L/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(sc, ax=ax, label="T rise (C)")

    # slice y = L/2
    ax = fig.add_subplot(132)
    m = np.abs(y - L / 2) < 0.51 * (L / nx)
    sc = ax.tricontourf(x[m], z[m], T[m] - T0, levels=20)
    ax.set_aspect("equal")
    ax.set_title("mid-plane y = L/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plt.colorbar(sc, ax=ax, label="T rise (C)")

    # 3D scatter of a decimated node set
    ax = fig.add_subplot(133, projection="3d")
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 4000)]
    sc = ax.scatter(x[sel], y[sel], z[sel], c=T[sel] - T0,
                    s=3, cmap="hot")
    ax.set_title("3D temperature field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label="T rise (C)", shrink=0.6)

    fig.suptitle("CloudFEM v0.7: 3D steady heat, steel cube, uniform source")
    fig.tight_layout()
    fig.savefig("heat_3d.png", dpi=110)
    print("saved heat_3d.png")


if __name__ == "__main__":
    main()
