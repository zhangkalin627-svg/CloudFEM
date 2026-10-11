"""Axisymmetric validation.

1. Solid cylinder (radius a) in uniform axial AC field H0:
   analytical H_z(r) = H0 * J0(alpha*r)/J0(alpha*a), alpha = (1-i)/delta.
2. Axisymmetric steady heat, manufactured solution
   T = (1-(r/a)^2)*sin(pi*z/L).
"""

import numpy as np
from scipy.special import jv

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, flux_density_axi, transient_heat_axi, MU0)
from cloudfem.physics.em import skin_depth


def test_cylinder_ac_field():
    a = 0.02
    sigma = 2e6
    freq = 10e3
    omega = 2 * np.pi * freq
    delta = skin_depth(sigma, omega)
    H0 = 1000.0

    mesh = rect_mesh(100, 20, xlim=(0, a), ylim=(0, a))
    # Neumann on the cylindrical surface: (2*pi*nu/r)*d(utilde)/dr = 2*pi*H0
    # (this enforces H_z(a) = H0); Dirichlet utilde = 0 on the axis.
    utilde = solve_eddy_current_axi(
        mesh, nu=1.0 / MU0, sigma=sigma, omega=omega,
        bc={"left": 0.0}, neumann={"right": 2 * np.pi * H0})
    B = flux_density_axi(mesh, utilde)
    Hz_num = (B[:, 1] / MU0).real  # compare magnitude via complex below
    Hz_num_c = B[:, 1] / MU0
    rc = np.array([mesh.nodes[el][:, 0].mean() for el in mesh.elements])

    alpha = (1 - 1j) / delta
    Hz_exact = H0 * jv(0, alpha * rc) / jv(0, alpha * a)

    # relative complex L2 error, weighted to the region carrying current
    err = np.sqrt(np.mean(np.abs(Hz_num_c - Hz_exact) ** 2)) / H0
    # outer-layer check against the analytical profile at the same radii
    dr = a / 100
    m = rc > a - 3 * dr
    surf_err = np.abs(Hz_num_c[m] - Hz_exact[m]).max() / H0
    return err, surf_err, delta, a


def test_axi_heat_convergence():
    a, L = 0.05, 0.1
    k = 40.0
    errs = []
    for n in (20, 40, 80):
        mesh = rect_mesh(n, 2 * n, xlim=(0, a), ylim=(0, L))

        def f_fn(x, y):
            S = np.sin(np.pi * y / L)
            T = (1 - (x / a) ** 2) * S
            return k * (4 / a ** 2 * S + (np.pi / L) ** 2 * T)

        bc = {"right": 0.0, "bottom": 0.0, "top": 0.0}
        times, hist = transient_heat_axi(
            mesh, k=k, rho_cp=1.0, Q=f_fn, bc=bc,
            t_end=1e9, dt=1e9, T0=0.0)  # one giant step -> steady state
        T = hist[-1]
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        T_exact = (1 - (x / a) ** 2) * np.sin(np.pi * y / L)
        errs.append(np.sqrt(np.mean((T - T_exact) ** 2)))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    return errs, rates


if __name__ == "__main__":
    err, surf_err, delta, a = test_cylinder_ac_field()
    print(f"[axi cylinder] a/delta = {a/delta:.2f}, rel L2 err = {err:.3e}, "
          f"surface field err = {surf_err:.3e}")
    assert err < 0.02 and surf_err < 0.02, "axisymmetric EM validation failed"

    errs, rates = test_axi_heat_convergence()
    print(f"[axi heat] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    assert all(r > 1.7 for r in rates), "axisymmetric heat convergence failed"

    print("ALL AXISYMMETRIC VALIDATIONS PASSED")
