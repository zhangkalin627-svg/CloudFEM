"""v1.1 demo: steady 3D heat conduction in a steel cylinder billet.

The billet (R = 50 mm, H = 100 mm) is meshed with the unstructured
tetrahedral mesher (cloudfem.mesh.unstructured3d) and solved with the
unmodified v0.7 steady solver -- the point of the demo is that the
new meshes feed the existing solvers through the plain TetMesh
interface. Uniform volumetric heat source, all surfaces held at 20 C.

Sanity: for an infinitely long cylinder the centre rise would be
Q*R^2/(4k); the finite billet also loses heat through its end caps,
so its centre rise must stay below that bound.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

from cloudfem.mesh.unstructured3d import cylinder_mesh, mesh_quality3d
from cloudfem.fem.tet_poisson import solve_tet_poisson


def main(h=0.01):
    R, H = 0.05, 0.10   # m
    k = 50.0            # W/m/K, steel
    Q = 5e6             # W/m^3
    T0 = 20.0           # C, all surfaces

    mesh = cylinder_mesh(R, H, h)
    qmin, qmed, _ = mesh_quality3d(mesh)
    T = solve_tet_poisson(mesh, k=k, f=Q, bc={"boundary": T0})

    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    c = np.argmin(x ** 2 + y ** 2 + (z - H / 2) ** 2)
    T_centre = T[c] - T0
    cyl_bound = Q * R ** 2 / (4 * k)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets, "
          f"quality qmin = {qmin:.3f}, median = {qmed:.3f}")
    print(f"centre temperature rise: {T_centre:.3f} C "
          f"(infinite-cylinder bound: {cyl_bound:.3f} C)")
    assert 0.0 < T_centre < cyl_bound, "unphysical centre temperature"
    assert T.min() >= T0 - 1e-9

    fig = plt.figure(figsize=(14, 4))

    # mesh nodes (decimated), 3D
    ax = fig.add_subplot(131, projection="3d")
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 3000)]
    ax.scatter(x[sel], y[sel], z[sel], s=2, c="steelblue")
    ax.set_title("unstructured tet mesh (nodes)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # mid-plane slice y = 0: contour in (x, z)
    ax = fig.add_subplot(132)
    m = np.abs(y) < 0.3 * h
    sc = ax.tricontourf(x[m], z[m], T[m] - T0, levels=20)
    ax.set_aspect("equal")
    ax.set_title("mid-plane y = 0")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plt.colorbar(sc, ax=ax, label="T rise (C)")

    # 3D scatter coloured by temperature
    ax = fig.add_subplot(133, projection="3d")
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 4000)]
    sc = ax.scatter(x[sel], y[sel], z[sel], c=T[sel] - T0,
                    s=3, cmap="hot")
    ax.set_title("3D temperature field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label="T rise (C)", shrink=0.6)

    fig.suptitle("CloudFEM v1.1: 3D steady heat, steel cylinder billet, "
                 "unstructured mesh")
    fig.tight_layout()
    fig.savefig("unstructured_cylinder_3d.png", dpi=110)
    print("saved unstructured_cylinder_3d.png")


if __name__ == "__main__":
    main()
