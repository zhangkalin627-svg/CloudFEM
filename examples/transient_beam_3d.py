"""v1.2 demo: transient dynamics of a 3D steel cantilever beam.

A 1 kN step load is applied in -z over the tip face of a cantilever
beam (clamped at x = 0) and the response is integrated with the
direct Newmark solver (average-acceleration parameters, undamped,
consistent mass). With no damping the tip deflection oscillates
between 0 and twice the static deflection at the first bending
frequency; the static value from the v0.8 solver is drawn as the
centre line and the measured first period is compared with the
v0.9 modal f1.

Plots: tip-deflection time history, and deformed shapes (nodal
scatter, exaggerated) at three instants -- near the first peak, back
through equilibrium, and near the first trough.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.physics.elastic3d import (assemble_elasticity_3d,
                                        apply_dirichlet_vec3)
from cloudfem.physics.modal3d import solve_modes_3d
from cloudfem.physics.dynamics3d import transient_3d, peak_response


def main(nx=32, nyz=4):
    L, h = 1.0, 0.1          # m
    E = 200e9                # Pa
    nu = 0.3
    rho = 7850.0             # kg/m^3
    Ftot = 1000.0            # N, step load in -z on the tip face

    mesh = box_mesh(nx, nyz, nyz, xlim=(0, L), ylim=(0, h),
                    zlim=(0, h))
    bc = {"xmin": (0.0, 0.0, 0.0)}
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets")

    tip = np.asarray(mesh.boundaries["xmax"])
    F = np.zeros(3 * mesh.n_nodes)
    F[3 * tip + 2] = -Ftot / len(tip)

    freqs, _ = solve_modes_3d(mesh, E, nu, rho, bc=bc, n_modes=3)
    f1 = freqs[0]
    T1 = 1.0 / f1
    print(f"modal f1 = {f1:.2f} Hz (T1 = {1e3 * T1:.2f} ms), "
          f"f2 = {freqs[1]:.2f} Hz, f3 = {freqs[2]:.2f} Hz")

    # static reference from the v0.8 solver
    K, _ = assemble_elasticity_3d(mesh, E, nu)
    u_st = apply_dirichlet_vec3(K, F, mesh, bc)
    tip_st = u_st[2::3][tip].mean()
    print(f"static tip deflection = {1e3 * tip_st:.4f} mm")

    dt = T1 / 100
    t, U, V, A = transient_3d(mesh, E, nu, rho, bc=bc, t_end=2.5 * T1,
                              dt=dt, load=F)
    tip_u = U[:, 2::3][:, tip].mean(axis=1)
    peak, t_peak, _ = peak_response(t, U)
    print(f"{len(t) - 1} steps of dt = {1e3 * dt:.3f} ms; peak |u| = "
          f"{1e3 * peak.max():.4f} mm at t = {1e3 * t_peak:.2f} ms")
    print(f"tip deflection range: [{1e3 * tip_u.min():.4f}, "
          f"{1e3 * tip_u.max():.4f}] mm "
          f"(undamped step response: 0 to 2x static = "
          f"{2e3 * tip_st:.4f} mm)")
    ratio = tip_u.min() / tip_st
    assert 1.85 < ratio < 2.05, \
        "undamped step overshoot must approach 2x the static value"
    # measured period from the first two minima of the tip history
    imin = [i for i in range(1, len(t) - 1)
            if tip_u[i] <= tip_u[i - 1] and tip_u[i] <= tip_u[i + 1]]
    T_meas = t[imin[1]] - t[imin[0]]
    print(f"measured period from tip history = {1e3 * T_meas:.2f} ms "
          f"(modal T1 = {1e3 * T1:.2f} ms, "
          f"{100.0 * (T_meas / T1 - 1.0):+.2f} % -- minima are located "
          "on the dt grid (+-1 step = +-1 % of T1) and shifted by "
          "the second bending mode; the Newmark period elongation "
          "itself is only (omega dt)^2/12 = 0.03 % at dt = T1/100)")
    assert abs(T_meas / T1 - 1.0) < 0.03

    # --- plots: tip history + deformed shapes at three instants ---
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    fig = plt.figure(figsize=(13, 4.2))

    ax = fig.add_subplot(141)
    ax.plot(1e3 * t, 1e3 * tip_u, lw=1.2)
    ax.axhline(1e3 * tip_st, color="k", ls="--", lw=0.9,
               label="static deflection")
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("tip deflection (mm)")
    ax.set_title(f"tip response to 1 kN step load (f1 = {f1:.1f} Hz)")
    ax.legend(loc="lower right", fontsize=8)

    # instants: first trough (~T1/2), equilibrium return (~T1),
    # second trough (~3T1/2)
    targets = [0.5 * T1, 1.0 * T1, 1.5 * T1]
    labels = ["t = T1/2 (first trough)", "t = T1 (equilibrium)",
              "t = 3T1/2 (second trough)"]
    umax = np.abs(U).max()
    scale = 0.25 * L / umax
    sel = np.arange(mesh.n_nodes)[:: max(1, mesh.n_nodes // 3000)]
    for k, (tt, lab) in enumerate(zip(targets, labels)):
        i = int(np.argmin(np.abs(t - tt)))
        ud = U[i].reshape(-1, 3)
        umag = np.linalg.norm(ud, axis=1)
        ax = fig.add_subplot(1, 4, 2 + k, projection="3d")
        sc = ax.scatter(x[sel] + scale * ud[sel, 0],
                        y[sel] + scale * ud[sel, 1],
                        z[sel] + scale * ud[sel, 2],
                        c=umag[sel], s=3, cmap="viridis")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.colorbar(sc, ax=ax, label="|u| (m)", shrink=0.55)

    fig.suptitle("CloudFEM v1.2: cantilever transient dynamics "
                 "(Newmark average acceleration, undamped)")
    fig.tight_layout()
    fig.savefig("transient_beam_3d.png", dpi=110)
    print("saved transient_beam_3d.png")


if __name__ == "__main__":
    main()
