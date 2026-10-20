"""v0.9 demo: modal analysis of a 3D steel cantilever beam.

Free-vibration analysis on tetrahedral P1 elements (consistent mass
matrix, lumped values printed for comparison). The beam is clamped at
x = 0; the first bending frequency is checked against Euler-Bernoulli
beam theory, f1 = (beta1^2 / (2 pi L^2)) * sqrt(E I / (rho A)) with
beta1*L = 1.8751040687. Constant-strain tetrahedra are overstiff in
bending, so the FEM frequency sits a few percent above the
beam-theory value (mesh sweep: +29 % on 20x2x2, +10 % on 32x4x4,
+6.6 % on the 40x5x5 mesh used here).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.physics.modal3d import solve_modes_3d


def main(nx=40, nyz=5):
    L, h = 1.0, 0.1          # m
    E = 200e9                # Pa
    nu = 0.3
    rho = 7850.0             # kg/m^3
    A = h * h
    I = h * h ** 3 / 12.0
    f_eb = 1.8751040687 ** 2 / (2.0 * np.pi * L ** 2) \
        * np.sqrt(E * I / (rho * A))

    mesh = box_mesh(nx, nyz, nyz, xlim=(0, L), ylim=(0, h), zlim=(0, h))
    bc = {"xmin": (0.0, 0.0, 0.0)}
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")

    freqs, modes = solve_modes_3d(mesh, E, nu, rho, bc=bc, n_modes=6)
    freqs_l, _ = solve_modes_3d(mesh, E, nu, rho, bc=bc, n_modes=6,
                                 lumped=True)
    print(" mode   f_consistent (Hz)   f_lumped (Hz)")
    for j in range(6):
        print(f"  {j + 1:d}        {freqs[j]:10.3f}      {freqs_l[j]:10.3f}")
    print(f"Euler-Bernoulli f1 = {f_eb:.3f} Hz; FEM/EB = "
          f"{freqs[0] / f_eb:.4f} (+{100.0 * (freqs[0] / f_eb - 1.0):.2f} %)")
    assert 1.0 < freqs[0] / f_eb < 1.15, \
        "first bending frequency outside the expected overstiff band"

    # --- plots: mode-1 magnitude on the mid-plane + deformed shape ---
    u1 = modes[:, 0].reshape(-1, 3)
    umag = np.linalg.norm(u1, axis=1)
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(121)
    m = np.abs(y - h / 2) < 0.51 * (h / nyz)
    sc = ax.tricontourf(x[m], z[m], umag[m], levels=20)
    ax.set_aspect("equal")
    ax.set_title("mode 1: |u| on mid-plane y = h/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plt.colorbar(sc, ax=ax, label="|u| (M-normalised)")

    ax = fig.add_subplot(122, projection="3d")
    scale = 0.25 * L / umag.max()  # exaggerated, mode amplitude is free
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 4000)]
    xd = x[sel] + scale * u1[sel, 0]
    yd = y[sel] + scale * u1[sel, 1]
    zd = z[sel] + scale * u1[sel, 2]
    sc = ax.scatter(xd, yd, zd, c=umag[sel], s=3, cmap="viridis")
    ax.set_title("mode 1: deformed shape (exaggerated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.colorbar(sc, ax=ax, label="|u|", shrink=0.6)

    fig.suptitle(f"CloudFEM v0.9: cantilever modal analysis, "
                 f"f1 = {freqs[0]:.1f} Hz (EB {f_eb:.1f} Hz)")
    fig.tight_layout()
    fig.savefig("modal_beam_3d.png", dpi=110)
    print("saved modal_beam_3d.png")


if __name__ == "__main__":
    main()
