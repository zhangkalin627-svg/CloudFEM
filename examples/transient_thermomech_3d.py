"""v1.0 demo: transient quench of a steel cube + thermal-stress chain.

A 50 mm steel cube at a uniform 850 C is oil-quenched by convection
(h = 1000 W/m^2/K into 25 C oil, Robin condition on all faces). The
transient temperature field (Crank-Nicolson, tet P1, consistent
capacity matrix) is chained through the 3D thermal-stress solver
with only rigid-body pins (3-2-1), so the stress is the
self-equilibrating gradient stress: zero at t = 0 (uniform field),
peaking while the centre-surface gap is largest, then decaying as
the cube homogenises.

Sanity: the surface must cool faster than the centre, the peak von
Mises stress must occur at an intermediate time and stay below the
fully constrained scale E*alpha*(T0 - T_inf).

Known artefact (documented, not hidden): the consistent capacity
matrix overshoots the initial temperature in the first instants
near the surface, where the initial condition is incompatible with
the boundary temperature. Measured here: max T exceeds 850 C by
~12 K (1.4 % of the 825 K jump) in a thin surface layer; the total
heat still decreases monotonically and the centre/surface curves
are unaffected at plot accuracy. A lumped capacity matrix would
remove the overshoot (see fem/tet_heat.py docstring).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_heat import transient_heat_tet
from cloudfem.physics.thermomech import transient_thermal_stress_3d


def _pin(mesh, name, point):
    """Register the node nearest to point as a one-node boundary."""
    d2 = np.sum((mesh.nodes - np.asarray(point, dtype=float)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    mesh.boundaries[name] = np.array([idx])
    return idx


def _nearest_node(mesh, point):
    d2 = np.sum((mesh.nodes - np.asarray(point, dtype=float)) ** 2, axis=1)
    return int(np.argmin(d2))


def main(nx=8):
    L = 0.05           # m
    k = 50.0           # W/m/K, steel
    rho_cp = 7850.0 * 500.0   # J/m^3/K
    T0 = 850.0         # C, uniform initial temperature
    T_inf = 25.0       # C, quench oil
    h = 1000.0         # W/m^2/K, oil quench
    E = 200e9          # Pa
    nu = 0.3
    alpha = 12e-6      # 1/K
    t_end, dt = 60.0, 0.3
    stride = 10        # stress snapshots every 10 heat steps

    mesh = box_mesh(nx, nx, nx,
                    xlim=(0, L), ylim=(0, L), zlim=(0, L))
    faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    robin = {n: (h, T_inf) for n in faces}

    # --- transient heat: convective quench on all faces ---
    times, hist = transient_heat_tet(mesh, k=k, rho_cp=rho_cp, robin=robin,
                                     t_end=t_end, dt=dt, T0=T0, method="cn")
    i_cen = _nearest_node(mesh, (L / 2, L / 2, L / 2))
    i_sur = _nearest_node(mesh, (L, L / 2, L / 2))
    T_cen, T_sur = hist[:, i_cen], hist[:, i_sur]
    overshoot = float(hist.max() - T0)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} tets; "
          f"{len(times) - 1} CN steps of {times[1] - times[0]:.2f} s")
    print(f"surface node: {T_sur[0]:.0f} -> {T_sur[-1]:.1f} C; "
          f"centre node: {T_cen[0]:.0f} -> {T_cen[-1]:.1f} C")
    print(f"consistent-mass surface overshoot: {overshoot:.1f} K "
          f"(see docstring)")
    assert T_sur[-1] < T_cen[-1], "surface must lead the centre"
    assert overshoot < 25.0, "overshoot beyond the documented tolerance"
    gap = T_cen - T_sur
    print(f"max centre-surface gap: {gap.max():.1f} C "
          f"at t = {times[int(np.argmax(gap))]:.1f} s")
    assert gap.max() > 100.0

    # --- transient thermal stress chain (rigid-body pins only) ---
    _pin(mesh, "pin_a", (0.0, 0.0, 0.0))
    _pin(mesh, "pin_b", (L, 0.0, 0.0))
    _pin(mesh, "pin_c", (0.0, L, 0.0))
    bc_el = {"pin_a": (0.0, 0.0, 0.0),
             "pin_b": (None, 0.0, 0.0),
             "pin_c": (None, None, 0.0)}
    chain = transient_thermal_stress_3d(
        mesh, times[::stride], hist[::stride], E, nu, alpha,
        T_ref=T0, bc=bc_el)
    bound = E * alpha * (T0 - T_inf)
    print(f"peak von Mises: {chain['peak_vm_max'] / 1e6:.1f} MPa "
          f"at t = {chain['peak_time']:.1f} s "
          f"(fully constrained scale: {bound / 1e6:.0f} MPa)")
    print(f"max displacement at peak: {chain['peak']['u_max'] * 1e3:.4f} mm; "
          f"von Mises at t_end: {chain['vm_max'][-1] / 1e6:.1f} MPa "
          f"(decaying as the cube homogenises)")
    assert 0.0 < chain["peak_vm_max"] < bound
    assert 0.0 < chain["peak_time"] < t_end, \
        "peak stress must sit at an intermediate time"
    assert chain["vm_max"][-1] < chain["peak_vm_max"]

    # --- plots ---
    fig = plt.figure(figsize=(14, 4))

    ax = fig.add_subplot(131)
    ax.plot(times, T_sur, label="surface (face centre)")
    ax.plot(times, T_cen, label="cube centre")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (C)")
    ax.set_title("quench temperatures")
    ax.legend()

    ax = fig.add_subplot(132)
    ax.plot(chain["times"], chain["vm_max"] / 1e6)
    ax.plot(chain["peak_time"], chain["peak_vm_max"] / 1e6, "ro")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("max von Mises (MPa)")
    ax.set_title("peak-stress history")

    ax = fig.add_subplot(133)
    cent = mesh.nodes[mesh.elements].mean(axis=1)
    mc = np.abs(cent[:, 2] - L / 2) < 0.51 * (L / nx)
    sc = ax.tricontourf(cent[mc, 0], cent[mc, 1],
                        chain["peak"]["vm_elem"][mc] / 1e6, levels=20)
    ax.set_aspect("equal")
    ax.set_title(f"von Mises at peak (t = {chain['peak_time']:.1f} s), "
                 f"z = L/2")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(sc, ax=ax, label="von Mises (MPa)")

    fig.suptitle("CloudFEM v1.0: transient quench + thermal stress, "
                 "steel cube, free body (rigid pins)")
    fig.tight_layout()
    fig.savefig("transient_thermomech_3d.png", dpi=110)
    print("saved transient_thermomech_3d.png")


if __name__ == "__main__":
    main()
