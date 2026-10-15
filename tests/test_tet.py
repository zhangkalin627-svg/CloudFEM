"""Validation of the 3D tetrahedral P1 solver (v0.7).

1. Mesh sanity: 6 tets per cell tile the box (volumes sum to the box
   volume, all positive).
2. Linear exactness: P1 reproduces affine solutions exactly.
3. Mesh convergence: manufactured solution sin(pi x)sin(pi y)sin(pi z)
   with f = 3*pi^2*T; P1 should show O(h^2) in L2.
"""

import numpy as np

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson


def test_tet_mesh_volumes():
    mesh = box_mesh(3, 4, 5, xlim=(0, 2.0), ylim=(0, 1.0), zlim=(0, 0.5))
    vols = mesh.element_volumes()
    assert np.all(vols > 0.0), "degenerate or inverted tetrahedra"
    assert abs(vols.sum() - 2.0 * 1.0 * 0.5) < 1e-10 * 2.0
    assert mesh.n_elements == 6 * 3 * 4 * 5


def test_tet_linear_exact(nx=6):
    exact = lambda x, y, z: x + 2.0 * y + 3.0 * z - 1.0
    mesh = box_mesh(nx, nx, nx)
    faces = {n: exact for n in
             ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    T = solve_tet_poisson(mesh, k=1.0, f=0.0, bc=faces)
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    err = np.sqrt(np.mean((T - exact(x, y, z)) ** 2))
    return err


def test_tet_manufactured(nx):
    exact = (lambda x, y, z:
             np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
    f = lambda x, y, z: 3.0 * np.pi ** 2 * exact(x, y, z)
    mesh = box_mesh(nx, nx, nx)
    # exact solution vanishes on all six faces of the unit cube
    faces = {n: 0.0 for n in
             ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    T = solve_tet_poisson(mesh, k=1.0, f=f, bc=faces)
    x, y, z = mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.nodes[:, 2]
    err = np.sqrt(np.mean((T - exact(x, y, z)) ** 2))
    return err


def convergence_rate(ns=(4, 8, 16)):
    errs = [test_tet_manufactured(n) for n in ns]
    hs = [1.0 / n for n in ns]
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
             for i in range(len(ns) - 1)]
    return errs, rates


def main():
    test_tet_mesh_volumes()
    print("mesh volumes: OK")

    err = test_tet_linear_exact()
    print(f"linear exactness L2 err = {err:.3e}")
    assert err < 1e-10, f"P1 must reproduce affine solutions, got {err}"

    errs, rates = convergence_rate()
    for n, e in zip((4, 8, 16), errs):
        print(f"  nx={n:2d}: L2 err = {e:.3e}")
    print(f"convergence rates: {[f'{r:.2f}' for r in rates]} (theory 2.0)")
    assert all(1.7 < r < 2.3 for r in rates), f"rate off: {rates}"
    print("ALL 3D TESTS PASSED")


if __name__ == "__main__":
    main()
