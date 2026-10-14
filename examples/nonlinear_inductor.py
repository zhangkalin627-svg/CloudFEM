"""C-core inductor: B-H saturation demo (nonlinear magnetostatics).

A laminated-steel C-core with a 2 mm air gap, coil wound around the
left leg. We sweep the magnetomotive force NI and record the average
flux density in the back leg. The curve is linear at low excitation
and bends over as the iron saturates (~1 T knee) -- the classic
saturation signature that a linear model cannot reproduce.

Geometry (metres), domain [0, 0.12]^2:
  back leg   x in [0.02, 0.10], y in [0.02, 0.04]
  left leg   x in [0.02, 0.04], y in [0.02, 0.10]
  right leg  x in [0.08, 0.10], y in [0.02, 0.10]
  top yoke   x in [0.02, 0.059] U [0.061, 0.10], y in [0.10, 0.12]
  air gap    x in [0.059, 0.061], y in [0.10, 0.12]
  coil +J    x in [0.008, 0.020], y in [0.02, 0.10]
  coil -J    x in [0.040, 0.052], y in [0.02, 0.10]
A = 0 on the outer boundary (flux containment).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.nonlinear_em import (
    solve_nonlinear_magnetostatics, bh_saturation, flux_density,
)

L = 0.12
GAP = (0.059, 0.061)


def in_rect(x, y, x0, x1, y0, y1):
    return (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)


def is_iron(x, y):
    x = np.asarray(x); y = np.asarray(y)
    back = in_rect(x, y, 0.02, 0.10, 0.02, 0.04)
    left = in_rect(x, y, 0.02, 0.04, 0.02, 0.10)
    right = in_rect(x, y, 0.08, 0.10, 0.02, 0.10)
    yoke_l = in_rect(x, y, 0.02, GAP[0], 0.10, 0.12)
    yoke_r = in_rect(x, y, GAP[1], 0.10, 0.10, 0.12)
    return back | left | right | yoke_l | yoke_r


def is_coil_pos(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return in_rect(x, y, 0.008, 0.020, 0.02, 0.10)


def is_coil_neg(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return in_rect(x, y, 0.040, 0.052, 0.02, 0.10)


COIL_AREA = 0.012 * 0.08   # m^2 per side


def main(nx=120):
    mesh = rect_mesh(nx, nx, xlim=(0, L), ylim=(0, L))
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    nu_model = bh_saturation(mu_r0=2000.0, B_knee=1.0, iron=is_iron)
    bc = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}

    # back-leg element mask for the B average
    p = mesh.nodes[mesh.elements]
    xc, yc = p[:, :, 0].mean(axis=1), p[:, :, 1].mean(axis=1)
    back = in_rect(xc, yc, 0.02, 0.10, 0.02, 0.04)
    iron_e = is_iron(xc, yc)

    NI_vals = np.logspace(np.log10(20), np.log10(30000), 12)
    B_back, B_iron_max, iters = [], [], []
    A = None
    print(f"{'NI [A.t]':>10} {'B_back [T]':>10} {'max|B|_iron':>12} {'iters':>6}")
    for NI in NI_vals:
        J0 = NI / (2 * COIL_AREA)
        J_s = lambda x, y, J0=J0: np.where(is_coil_pos(x, y), J0,
                                          np.where(is_coil_neg(x, y), -J0, 0.0))
        A, info = solve_nonlinear_magnetostatics(
            mesh, nu_model, J_s=J_s, bc=bc, tol=1e-7, A_init=A)
        assert info["converged"], f"picard did not converge at NI={NI:.0f}"
        B = flux_density(mesh, A)
        Bmag = np.hypot(B[:, 0], B[:, 1])
        B_back.append(float(Bmag[back].mean()))
        B_iron_max.append(float(Bmag[iron_e].max()))
        iters.append(info["iterations"])
        print(f"{NI:10.0f} {B_back[-1]:10.3f} {B_iron_max[-1]:12.3f} "
              f"{iters[-1]:6d}")
    B_back = np.array(B_back)

    # --- saturation signature checks -------------------------------------
    # 1) linear regime: doubling a small NI doubles B
    lin_ratio = B_back[1] / B_back[0]
    print(f"\nlinearity check: B({NI_vals[1]:.0f})/B({NI_vals[0]:.0f}) = "
          f"{lin_ratio:.4f} (expect ~{NI_vals[1]/NI_vals[0]:.2f})")
    assert abs(lin_ratio - NI_vals[1] / NI_vals[0]) < 0.05 * NI_vals[1] / NI_vals[0]
    # 2) saturation: top point far below the linear extrapolation
    slope = B_back[1] / NI_vals[1]
    sat_frac = B_back[-1] / (slope * NI_vals[-1])
    print(f"saturation check: B({NI_vals[-1]:.0f}) is {sat_frac:.1%} of the "
          f"linear extrapolation")
    assert sat_frac < 0.55, "no saturation knee observed"
    assert B_iron_max[-1] > 0.9, "iron never entered the knee transition"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(NI_vals, B_back, "o-", label="nonlinear (saturation)")
    ax.loglog(NI_vals, slope * NI_vals, "--", label="linear extrapolation")
    ax.set_xlabel("magnetomotive force NI [A.turns]")
    ax.set_ylabel("average |B| in back leg [T]")
    ax.set_title("C-core inductor: saturation curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("nonlinear_inductor_saturation.png", dpi=110)
    print("saved nonlinear_inductor_saturation.png")


if __name__ == "__main__":
    main()
