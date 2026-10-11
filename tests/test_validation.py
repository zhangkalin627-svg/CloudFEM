"""Validation against analytical solutions.

1. Steady heat: 1D slab, uniform generation, T=0 at both faces.
   Analytical: T(x) = Q/(2k) * x*(L-x).
2. Skin depth: semi-infinite conductor, A(0) = A0.
   Analytical: A(x) = A0 * exp(-(1+i)*x/delta).
3. Mesh convergence: P1 elements should show O(h^2) in L2.
"""

import numpy as np

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.heat import steady_heat
from cloudfem.physics.em import solve_eddy_current, skin_depth, MU0


def test_heat_slab(nx=50):
    L, W = 0.1, 0.02
    k, Q = 50.0, 1e6
    mesh = rect_mesh(nx, max(4, nx // 5), xlim=(0, L), ylim=(0, W))
    T = steady_heat(mesh, k=k, Q=Q, bc={"left": 0.0, "right": 0.0})
    x = mesh.nodes[:, 0]
    T_exact = Q / (2 * k) * x * (L - x)
    err = np.sqrt(np.mean((T - T_exact) ** 2)) / T_exact.max()
    return err, T_exact.max()


def test_skin_depth():
    sigma = 1e6          # S/m
    freq = 10e3          # Hz
    omega = 2 * np.pi * freq
    delta = skin_depth(sigma, omega)
    L, W = 6 * delta, 2 * delta
    mesh = rect_mesh(120, 20, xlim=(0, L), ylim=(0, W))
    A = solve_eddy_current(mesh, nu=1.0 / MU0, sigma=sigma, omega=omega,
                           bc={"left": 1.0, "right": 0.0})
    # centerline profile
    yc = W / 2
    rows = np.where(np.abs(mesh.nodes[:, 1] - yc)
                    < 0.51 * (W / 20))[0]
    order = np.argsort(mesh.nodes[rows, 0])
    rows = rows[order]
    x = mesh.nodes[rows, 0]
    A_num = A[rows]
    A_exact = np.exp(-(1 + 1j) * x / delta)
    err = np.sqrt(np.mean(np.abs(A_num - A_exact) ** 2))
    # Phase is only meaningful where the field is significant; also the
    # artificial A=0 end at 6*delta perturbs the tail. Restrict to x<4d
    # where >98% of the Joule heating occurs.
    m = x < 4 * delta
    mag_err = np.abs(np.abs(A_num[m]) - np.exp(-x[m] / delta)).max()
    phase_err = np.abs(np.unwrap(np.angle(A_num[m])) + x[m] / delta).max()
    # Joule heating profile: the quantity that actually feeds the thermal
    # solver. Q = 0.5*sigma*omega^2*|A|^2, exact: Q0*exp(-2x/delta).
    Q_num = 0.5 * sigma * omega ** 2 * np.abs(A_num[m]) ** 2
    Q_exact = 0.5 * sigma * omega ** 2 * np.exp(-2 * x[m] / delta)
    q_err = np.abs(Q_num - Q_exact).max() / Q_exact.max()
    return err, mag_err, phase_err, q_err, delta


def test_convergence():
    # manufactured solution: T = sin(pi x) sin(pi y), -laplacian T = f
    errs = []
    for nx in (20, 40, 80):
        mesh = rect_mesh(nx, nx, xlim=(0, 1), ylim=(0, 1))
        f = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        T = steady_heat(mesh, k=1.0, Q=f,
                        bc={"left": 0.0, "right": 0.0,
                            "bottom": 0.0, "top": 0.0})
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        T_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        errs.append(np.sqrt(np.mean((T - T_exact) ** 2)))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    return errs, rates


if __name__ == "__main__":
    err, Tmax = test_heat_slab()
    print(f"[heat slab]  rel L2 error = {err:.3e},  Tmax = {Tmax:.3f} K")
    assert err < 0.01, "heat slab validation failed"

    err, mag_err, phase_err, q_err, delta = test_skin_depth()
    print(f"[skin depth] delta = {delta*1e3:.3f} mm, L2 err = {err:.3e}, "
          f"max|mag| err = {mag_err:.3e}, max phase err = {phase_err:.3e} rad, "
          f"max Q-profile err = {q_err:.3e}")
    assert err < 0.02 and phase_err < 0.05 and q_err < 0.02, \
        "skin depth validation failed"

    errs, rates = test_convergence()
    print(f"[convergence] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    assert all(r > 1.7 for r in rates), "convergence rate too low"

    print("ALL VALIDATIONS PASSED")
