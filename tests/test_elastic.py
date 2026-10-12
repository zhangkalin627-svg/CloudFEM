"""Validation of the structural (linear elasticity) module.

1. Constrained block, uniform heating (plane strain), all u = 0:
   analytical sigma_x = sigma_y = -E*alpha*dT / ((1+nu)*(1-2*nu)),
   shear = 0.
2. Manufactured solution (plane stress), Dirichlet on all boundaries:
   P1 elements must show O(h^2) convergence of the displacement in L2.
3. Full disk with parabolic radial temperature (plane stress), pinned only
   to remove rigid-body motion (pins consistent with the free solution):
   analytical free-disk stresses
       sigma_r     = K * (1 - (r/R)^2)
       sigma_theta = K * (1 - 3*(r/R)^2),   K = alpha*E*dT / 4.
"""

import numpy as np
import sympy as sp

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.mesh.unstructured import disk_mesh
from cloudfem.physics.elastic import (
    solve_elasticity, element_stress_strain, von_mises)
from cloudfem.physics.thermomech import thermal_stress

# steel-ish properties
E_STEEL = 200e9
NU_STEEL = 0.3
ALPHA_STEEL = 12e-6


def test_constrained_block():
    """Uniform heating of a fully clamped block (plane strain)."""
    mesh = rect_mesh(20, 10, xlim=(0.0, 0.02), ylim=(0.0, 0.01))
    dT = 50.0
    T = np.full(mesh.n_nodes, dT)
    ux, uy = solve_elasticity(mesh, E_STEEL, NU_STEEL, plane="strain",
                              bc={"all": (0.0, 0.0)},
                              T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    assert np.abs(ux).max() < 1e-12 and np.abs(uy).max() < 1e-12
    sig, _ = element_stress_strain(mesh, (ux, uy), E_STEEL, NU_STEEL,
                                   plane="strain",
                                   T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    s_exact = -E_STEEL * ALPHA_STEEL * dT / ((1.0 + NU_STEEL) * (1.0 - 2.0 * NU_STEEL))
    err_x = abs(sig[:, 0].mean() - s_exact) / abs(s_exact)
    err_y = abs(sig[:, 1].mean() - s_exact) / abs(s_exact)
    shear = abs(sig[:, 2]).max() / abs(s_exact)
    return err_x, err_y, shear, s_exact


def _manufactured_body(E, nu, a):
    """Body force for u = (sin(pi x) sin(pi y), a sin(2 pi x) sin(pi y)).

    Derived symbolically: f = -div sigma(u), plane stress.
    Returns (fx, fy) callables f(x, y).
    """
    x, y = sp.symbols("x y")
    ux = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    uy = a * sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * y)
    exx = sp.diff(ux, x)
    eyy = sp.diff(uy, y)
    gxy = sp.diff(ux, y) + sp.diff(uy, x)
    fac = E / (1 - nu ** 2)
    sxx = fac * (exx + nu * eyy)
    syy = fac * (eyy + nu * exx)
    txy = fac * (1 - nu) / 2 * gxy
    fx = -(sp.diff(sxx, x) + sp.diff(txy, y))
    fy = -(sp.diff(txy, x) + sp.diff(syy, y))
    fx_fn = sp.lambdify((x, y), fx, "numpy")
    fy_fn = sp.lambdify((x, y), fy, "numpy")
    ux_fn = sp.lambdify((x, y), ux, "numpy")
    uy_fn = sp.lambdify((x, y), uy, "numpy")
    return (fx_fn, fy_fn), (ux_fn, uy_fn)


def test_convergence():
    E, nu, a = 1.0, 0.3, 0.3
    (fx_fn, fy_fn), (ux_fn, uy_fn) = _manufactured_body(E, nu, a)
    errs = []
    for nx in (16, 32, 64):
        mesh = rect_mesh(nx, nx, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        ux, uy = solve_elasticity(
            mesh, E, nu, plane="stress",
            bc={"left": (ux_fn, uy_fn), "right": (ux_fn, uy_fn),
                "bottom": (ux_fn, uy_fn), "top": (ux_fn, uy_fn)},
            body=(fx_fn, fy_fn))
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        ex = ux - ux_fn(x, y)
        ey = uy - uy_fn(x, y)
        errs.append(np.sqrt(np.mean(ex ** 2 + ey ** 2)))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    return errs, rates


def _pinned_disk_mesh(R, h):
    """Full disk mesh with centre pinned (ux=uy=0) and one rim node uy=0.

    Both pins are consistent with the analytical free-disk solution
    (u(0) = 0, u_y(R, 0) = 0), and the thermal load is self-equilibrated,
    so pin reactions vanish with mesh refinement.

    NOTE: a half-disk with symmetry BCs was tried first, but the long
    straight diameter edge makes the Delaunay mesher emit degenerate
    sliver triangles along it (all three nodes ~collinear), corrupting
    the corner stresses. Straight-edged polygons are a known mesher
    limitation; the full disk avoids it.
    """
    mesh = disk_mesh(R, h)
    centre = int(np.argmin(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])))
    rim = int(np.argmin(np.abs(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1]) - R)
                        + np.abs(mesh.nodes[:, 1])))
    mesh.boundaries["pin_centre"] = np.array([centre])
    mesh.boundaries["pin_rim"] = np.array([rim])
    return mesh


def test_disk_thermal_stress():
    """Parabolic radial temperature on a free disk (plane stress).

    Analytical (plane stress, traction-free rim):
        sigma_r     = K * (1 - (r/R)^2)
        sigma_theta = K * (1 - 3*(r/R)^2),   K = alpha*E*dT / 4.
    """
    R, h = 0.06, 0.004
    dT = 80.0
    mesh = _pinned_disk_mesh(R, h)
    r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    T = dT * (r / R) ** 2  # T_ref = 0
    res = thermal_stress(mesh, T, E_STEEL, NU_STEEL, ALPHA_STEEL,
                         T_ref=0.0, plane="stress",
                         bc={"pin_centre": (0.0, 0.0), "pin_rim": (None, 0.0)})
    sig = res["sig_elem"]
    cent = mesh.nodes[mesh.elements].mean(axis=1)
    rc = np.hypot(cent[:, 0], cent[:, 1])
    th = np.arctan2(cent[:, 1], cent[:, 0])
    c, s = np.cos(th), np.sin(th)
    sx, sy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    sr_num = sx * c ** 2 + sy * s ** 2 + 2 * txy * s * c
    st_num = sx * s ** 2 + sy * c ** 2 - 2 * txy * s * c
    K = ALPHA_STEEL * E_STEEL * dT / 4.0
    sr_ana = K * (1.0 - (rc / R) ** 2)
    st_ana = K * (1.0 - 3.0 * (rc / R) ** 2)
    err = np.sqrt(np.mean((sr_num - sr_ana) ** 2 + (st_num - st_ana) ** 2)) / K
    return err, K, res["vm_max"]


if __name__ == "__main__":
    err_x, err_y, shear, s_exact = test_constrained_block()
    print(f"[constrained block] sigma_exact = {s_exact/1e6:.1f} MPa, "
          f"rel err sx = {err_x:.3e}, sy = {err_y:.3e}, "
          f"max|shear|/exact = {shear:.3e}")
    assert err_x < 0.02 and err_y < 0.02 and shear < 1e-6, \
        "constrained block validation failed"

    errs, rates = test_convergence()
    print(f"[convergence] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    assert all(r > 1.7 for r in rates), "convergence rate too low"

    err, K, vm_max = test_disk_thermal_stress()
    print(f"[disk thermal stress] K = {K/1e6:.2f} MPa, "
          f"rel L2 stress err = {err:.3e}, max von Mises = {vm_max/1e6:.2f} MPa")
    assert err < 0.05, "disk thermal stress validation failed"

    print("ALL ELASTIC VALIDATIONS PASSED")
