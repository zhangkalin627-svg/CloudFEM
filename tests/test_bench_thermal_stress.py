"""BIG-M7: CloudFEM thermal-structural coupling benchmark.

1-D rod thermal stress with a closed-form (textbook) reference
solution, computed with the real CloudFEM chain (heat solve ->
thermal-stress solve on the same 3D tet mesh). Nothing is fed from
the analytic temperature profile into the FEM except through the
heat solver's boundary data.

Problem
-------
Slender steel rod, axis along x:
    L = 0.2 m, cross-section 0.01 m x 0.01 m (L/h = 20).

Thermal problem (steady conduction, CloudFEM solve_tet_poisson):
    T = 220 C on the xmin face, T = 20 C on the xmax face, lateral
    faces insulated, constant k. Exact solution is the linear
    profile  T(x) = 220 - 1000 x  [C],  mean temperature 120 C.

Structural problem (CloudFEM solve_elasticity_3d): both end faces
fully clamped (u = 0), no body force / traction, thermal strain
from the nodal temperature field of the heat solve, T_ref = 20 C.

Analytic 1-D rod solution (independent reference)
-------------------------------------------------
With dT = T(x) - T_ref, uniaxial stress sigma(x) and total strain
eps = sigma/E + alpha*dT:
  * equilibrium without body force  ->  sigma = const,
  * clamped ends: int_0^L eps dx = u(L) - u(0) = 0
        ->  sigma = -E * alpha * <dT>,
    <dT> = mean temperature rise = 120 - 20 = 100 K.
    sigma_ref = -200e9 * 12e-6 * 100 = -240.00 MPa (uniform).
  * mid-span displacement:
        u(L/2) = alpha * int_0^{L/2} (dT - <dT>) dx
               = 12e-6 * 5 K*m = +60 um.
  * end reactions: |R| = |sigma_ref| * A = 240e6 * 1e-4 = 24.0 kN
    at each end, opposite signs (self-equilibrated).

Sampling: element sigma_xx is averaged over the central slab
x/L in [0.45, 0.55] (>= 9 section widths from either end, clear of
the clamped-end boundary layers). sigma_yy / sigma_zz in the slab
should vanish (uniaxial state) and are reported as a physics check.

Run:  python3 tests/test_bench_thermal_stress.py   (from the repo root)
Integrated into the CloudFEM suite by BIG-M11; physics unchanged
from the standalone BIG-M7 script.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson
from cloudfem.physics.elastic3d import (assemble_elasticity_3d,
                                        apply_dirichlet_vec3,
                                        element_stress_strain_3d)

# ---------------- parameters (steel) ----------------
L = 0.2            # rod length [m]
H = 0.01           # cross-section width/height [m]
E = 200e9          # Young's modulus [Pa]
NU = 0.3           # Poisson ratio
ALPHA = 12e-6      # thermal expansion [1/K]
K_THERM = 50.0     # thermal conductivity [W/(m K)]
T_HOT = 220.0      # xmin face temperature [C]
T_COLD = 20.0      # xmax face temperature [C]
T_REF = 20.0       # stress-free reference temperature [C]

# ---------------- analytic reference ----------------
DT_MEAN = 0.5 * (T_HOT + T_COLD) - T_REF          # 100 K
SIG_REF = -E * ALPHA * DT_MEAN                    # -240 MPa
U_MID_REF = ALPHA * 5.0                           # +60 um (see docstring)
R_REF = abs(SIG_REF) * H * H                      # 24.0 kN
SLAB = (0.45 * L, 0.55 * L)                       # sampling slab [m]


def run_level(nx, ny, nz):
    """One mesh level of the full heat -> thermal-stress chain."""
    mesh = box_mesh(nx, ny, nz, xlim=(0.0, L), ylim=(0.0, H),
                    zlim=(0.0, H))
    cen = mesh.nodes[mesh.elements].mean(axis=1)  # element centroids

    # ---- 1) heat solve (steady conduction on the tet mesh) ----
    Tn = solve_tet_poisson(mesh, k=K_THERM, f=0.0,
                           bc={"xmin": T_HOT, "xmax": T_COLD})
    T_exact_nodes = T_HOT + (T_COLD - T_HOT) * mesh.nodes[:, 0] / L
    err_T = float(np.max(np.abs(Tn - T_exact_nodes)))
    T_mean_vol = float(np.average(Tn[mesh.elements].mean(axis=1),
                                  weights=mesh.element_volumes()))

    # ---- 2) thermal-stress solve with the heat-solved T field ----
    bc3 = {"xmin": (0.0, 0.0, 0.0), "xmax": (0.0, 0.0, 0.0)}
    A_mat, rhs = assemble_elasticity_3d(mesh, E, NU, T=Tn,
                                        alpha=ALPHA, T_ref=T_REF)
    u = apply_dirichlet_vec3(A_mat, rhs, mesh, bc3)
    sig, _ = element_stress_strain_3d(mesh, u, E, NU, T=Tn,
                                      alpha=ALPHA, T_ref=T_REF)

    # ---- 3) sampling: central slab element stresses ----
    in_slab = (cen[:, 0] >= SLAB[0]) & (cen[:, 0] <= SLAB[1])
    sxx = sig[in_slab, 0]
    syy = sig[in_slab, 1]
    szz = sig[in_slab, 2]
    sig_mean = float(sxx.mean())
    err_sig = abs(sig_mean - SIG_REF) / abs(SIG_REF) * 100.0
    spread = float((sxx.max() - sxx.min()) / abs(SIG_REF) * 100.0)

    # ---- 4) mid-span axial displacement (nodes on x = L/2 plane) --
    mid = np.isclose(mesh.nodes[:, 0], 0.5 * L)
    u_mid = float(u[0::3][mid].mean())
    err_u = abs(u_mid - U_MID_REF) / U_MID_REF * 100.0

    # ---- 5) end reactions from the residual A u - rhs ----
    r = A_mat @ u - rhs
    rx_min = float(r[0::3][mesh.boundaries["xmin"]].sum())
    rx_max = float(r[0::3][mesh.boundaries["xmax"]].sum())
    err_R = abs(abs(rx_min) - R_REF) / R_REF * 100.0

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "n_elem": mesh.n_elements, "n_node": mesh.n_nodes,
        "err_T": err_T, "T_mean_vol": T_mean_vol,
        "sig_mean_MPa": sig_mean / 1e6, "err_sig_pct": err_sig,
        "spread_pct": spread,
        "syy_MPa": float(syy.mean()) / 1e6,
        "szz_MPa": float(szz.mean()) / 1e6,
        "u_mid_um": u_mid * 1e6, "err_u_pct": err_u,
        "R_min_kN": rx_min / 1e3, "R_max_kN": rx_max / 1e3,
        "err_R_pct": err_R,
    }


def main():
    print("BIG-M7: 1-D rod thermal-stress benchmark "
          "(CloudFEM heat -> thermal stress chain)")
    print(f"rod L={L} m, section {H}x{H} m, E={E:.3g} Pa, nu={NU}, "
          f"alpha={ALPHA:.3g} 1/K, k={K_THERM} W/(m K)")
    print(f"thermal BC: xmin {T_HOT} C, xmax {T_COLD} C, "
          f"T_ref {T_REF} C")
    print(f"analytic: <dT> = {DT_MEAN:.1f} K, "
          f"sigma_ref = {SIG_REF / 1e6:.2f} MPa, "
          f"u_mid_ref = {U_MID_REF * 1e6:.1f} um, "
          f"|R|_ref = {R_REF / 1e3:.1f} kN")
    print("-" * 100)
    hdr = (f"{'level':>14} {'elems':>7} {'max|dT|err':>11} "
           f"{'sig_xx slab':>12} {'err%':>7} {'spread%':>8} "
           f"{'syy':>8} {'szz':>8} {'u_mid um':>9} {'err%':>7} "
           f"{'|R| err%':>9}")
    print(hdr)
    results = []
    for (nx, ny, nz) in [(40, 2, 2), (80, 3, 3), (100, 4, 4),
                         (160, 6, 6)]:
        r = run_level(nx, ny, nz)
        results.append(r)
        print(f"{f'{nx}x{ny}x{nz}':>14} {r['n_elem']:>7} "
              f"{r['err_T']:>11.2e} {r['sig_mean_MPa']:>12.3f} "
              f"{r['err_sig_pct']:>7.3f} {r['spread_pct']:>8.3f} "
              f"{r['syy_MPa']:>8.3f} {r['szz_MPa']:>8.3f} "
              f"{r['u_mid_um']:>9.2f} {r['err_u_pct']:>7.3f} "
              f"{r['err_R_pct']:>9.3f}")
    print("-" * 100)
    fin = results[-1]
    print(f"volume-mean T (finest) = {fin['T_mean_vol']:.6f} C "
          f"(analytic 120 C)")
    print(f"reactions (finest): xmin {fin['R_min_kN']:+.2f} kN, "
          f"xmax {fin['R_max_kN']:+.2f} kN (analytic |R| = 24.00 kN)")
    ok = fin["err_sig_pct"] < 2.0
    print(f"VERDICT: finest-level slab-mean sigma_xx error "
          f"= {fin['err_sig_pct']:.3f}%  ->  "
          f"{'PASS (< 2%)' if ok else 'FAIL (>= 2%)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
