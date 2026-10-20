"""Validation of the 3D unstructured tetrahedral mesher (v1.1).

All reference values are derived independently of the mesher/solver
(analytic geometry, steady conduction solutions, linear elasticity):

1. Volume conservation: mesh volume vs analytic volume for a sphere
   (4*pi*R^3/3), a solid cylinder (pi*R^2*H) and the L block
   (lx*notch_y + notch_x*(ly - notch_y), times height). Curved
   surfaces are faceted, so sphere/cylinder errors must shrink like
   O(h^2); the L block has planar faces and must be exact.
2. Hollow-cylinder radial conduction: T(ri) = 100, T(R) = 0, caps
   insulated, gives the z-invariant log profile
   T(r) = 100 * ln(R/r) / ln(R/ri). P1 on the unstructured mesh must
   converge at ~O(h^2) in the nodal RMS error.
3. Sphere with uniform volumetric source q, T = 0 on the surface:
   T(r) = q*(R^2 - r^2)/(6k) (parabolic profile).
4. Mesh quality: radius-ratio q = 3*r_in/R_circ; the meshes used
   here must have qmin > 0.025 and < 1 % of tets below q = 0.05.
5. Solver hookup (structural): solid sphere, uniform dT, surface
   fully fixed -> u = 0 and hydrostatic stress
   sigma = -E*alpha*dT/(1 - 2*nu) exactly (affine-exact for P1),
   through the unmodified v0.8 elastic solver.
"""

import numpy as np

from cloudfem.mesh.unstructured3d import (sphere_mesh, cylinder_mesh,
                                          l_block_mesh, mesh_quality3d,
                                          tet_quality)
from cloudfem.fem.tet_poisson import solve_tet_poisson
from cloudfem.physics.thermomech import thermal_stress_3d


def _vol_err(mesh, exact):
    return abs(mesh.element_volumes().sum() - exact) / exact


def test_volume_convergence():
    cases = [
        ("sphere", lambda h: sphere_mesh(1.0, h),
         4.0 * np.pi / 3.0, (0.4, 0.28, 0.2)),
        ("cylinder", lambda h: cylinder_mesh(1.0, 2.0, h),
         2.0 * np.pi, (0.4, 0.28, 0.2)),
    ]
    out = {}
    for name, gen, exact, hs in cases:
        errs = [_vol_err(gen(h), exact) for h in hs]
        rates = [np.log(errs[i] / errs[i + 1])
                 / np.log(hs[i] / hs[i + 1]) for i in range(len(hs) - 1)]
        out[name] = (errs, rates)
        assert all(e1 < e0 for e0, e1 in zip(errs, errs[1:])), \
            f"{name} volume error not decreasing: {errs}"
        assert all(1.5 < r < 2.6 for r in rates), \
            f"{name} volume rate off O(h^2): {rates}"
    # L block: planar faces -> volume exact to roundoff
    for h in (0.5, 0.35):
        m = l_block_mesh(3.0, 3.0, (1.0, 1.0), 1.0, h)
        err = _vol_err(m, 5.0)
        out.setdefault("lblock", []).append(err)
        assert err < 1e-9, f"L block volume err {err}"
    return out


def test_quality():
    meshes = {
        "sphere h=0.2": sphere_mesh(1.0, 0.2),
        "cylinder h=0.2": cylinder_mesh(1.0, 2.0, 0.2),
        "hollow cyl h=0.2": cylinder_mesh(1.0, 1.0, 0.2, r_inner=0.5),
        "L block h=0.35": l_block_mesh(3.0, 3.0, (1.0, 1.0), 1.0, 0.35),
    }
    stats = {}
    for name, m in meshes.items():
        q = tet_quality(m.nodes, m.elements)
        qmin, qmed, _ = mesh_quality3d(m)
        frac = float((q < 0.05).mean())
        stats[name] = (qmin, qmed, frac)
        assert qmin > 0.025, f"{name}: qmin {qmin}"
        assert frac < 0.01, f"{name}: {frac:.2%} tets below q=0.05"
        assert np.all(m.element_volumes() > 0.0), \
            f"{name}: degenerate tetrahedra present"
    return stats


def test_hollow_cylinder_conduction():
    ri, R, H, T_in = 0.5, 1.0, 1.0, 100.0
    exact = lambda r: T_in * np.log(R / r) / np.log(R / ri)
    errs = []
    for h in (0.2, 0.14):
        m = cylinder_mesh(R, H, h, r_inner=ri)
        T = solve_tet_poisson(m, k=1.0, f=0.0,
                              bc={"inner": T_in, "lateral": 0.0})
        r = np.linalg.norm(m.nodes[:, :2], axis=1)
        errs.append(np.sqrt(np.mean((T - exact(r)) ** 2)) / T_in)
    rate = np.log(errs[0] / errs[1]) / np.log(0.2 / 0.14)
    return errs, rate


def test_sphere_source():
    errs = []
    for h in (0.3, 0.2):
        m = sphere_mesh(1.0, h)
        T = solve_tet_poisson(m, k=1.0, f=6.0, bc={"boundary": 0.0})
        r = np.linalg.norm(m.nodes, axis=1)
        errs.append(np.sqrt(np.mean((T - (1.0 - r ** 2)) ** 2)))
    rate = np.log(errs[0] / errs[1]) / np.log(0.3 / 0.2)
    return errs, rate


def test_sphere_thermal_stress():
    E, nu, alpha, dT = 200e9, 0.3, 12e-6, 50.0
    m = sphere_mesh(1.0, 0.25)
    res = thermal_stress_3d(m, np.full(m.n_nodes, dT), E, nu, alpha,
                            T_ref=0.0, bc={"boundary": (0.0, 0.0, 0.0)})
    sig_exact = -E * alpha * dT / (1.0 - 2.0 * nu)
    sig = res["sig_elem"]
    rel = np.abs(sig[:, :3] - sig_exact).max() / abs(sig_exact)
    shear = np.abs(sig[:, 3:]).max() / abs(sig_exact)
    return rel, shear, res["u_max"], res["vm_max"]


def main():
    out = test_volume_convergence()
    for name in ("sphere", "cylinder"):
        errs, rates = out[name]
        print(f"[volume {name}] errs = {[f'{e:.2e}' for e in errs]}, "
              f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    print(f"[volume L block] errs = {[f'{e:.2e}' for e in out['lblock']]}"
          " (planar faces, expect ~0)")

    stats = test_quality()
    for name, (qmin, qmed, frac) in stats.items():
        print(f"[quality {name}] qmin = {qmin:.4f}, median = {qmed:.3f}, "
              f"frac(q<0.05) = {frac * 100:.2f}%")

    errs, rate = test_hollow_cylinder_conduction()
    print(f"[hollow cylinder log profile] rel RMS errs = "
          f"{[f'{e:.3e}' for e in errs]}, rate = {rate:.2f} (expect ~2)")
    assert errs[0] < 6e-3 and rate > 1.5, "hollow cylinder validation failed"

    errs, rate = test_sphere_source()
    print(f"[sphere uniform source] rel RMS errs = "
          f"{[f'{e:.3e}' for e in errs]}, rate = {rate:.2f}")
    assert errs[0] < 5e-3 and rate > 1.3, "sphere source validation failed"

    rel, shear, u_max, vm_max = test_sphere_thermal_stress()
    print(f"[sphere thermal stress] normal-stress rel err = {rel:.3e}, "
          f"shear ratio = {shear:.3e}, u_max = {u_max:.3e}, "
          f"vm_max = {vm_max:.3e} Pa")
    assert rel < 1e-10 and shear < 1e-10, "thermal stress hookup failed"

    print("ALL 3D UNSTRUCTURED TESTS PASSED")


if __name__ == "__main__":
    main()
