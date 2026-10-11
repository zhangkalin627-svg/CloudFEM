"""Unstructured mesh validation.

1. Steady heat in a disk with uniform source, T=0 on boundary:
   analytical T(r) = Q/(4k)*(R^2 - r^2). Checks O(h^2) convergence.
2. Mesh quality sanity: no degenerate triangles (qmin > 0.2).
"""

import numpy as np

from cloudfem.mesh.unstructured import disk_mesh, mesh_quality
from cloudfem.physics.heat import steady_heat


def test_disk_heat_convergence():
    R, k, Q = 0.05, 40.0, 1e6
    errs = []
    for h in (0.008, 0.004, 0.002):
        m = disk_mesh(R, h)
        qmin, _ = mesh_quality(m)
        assert qmin > 0.2, f"degenerate triangles at h={h}: qmin={qmin}"
        T = steady_heat(m, k=k, Q=Q, bc={"boundary": 0.0})
        r = np.linalg.norm(m.nodes, axis=1)
        T_exact = Q / (4 * k) * (R ** 2 - r ** 2)
        errs.append(np.sqrt(np.mean((T - T_exact) ** 2)) / T_exact.max())
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    return errs, rates


if __name__ == "__main__":
    errs, rates = test_disk_heat_convergence()
    print(f"[disk heat] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    assert all(r > 1.7 for r in rates), "unstructured convergence failed"
    print("UNSTRUCTURED MESH VALIDATION PASSED")
