"""BIG-D2: cylinder 50 kHz induction heating -- FEM vs classical skin-depth formula.

FEM path: CloudFEM axisymmetric eddy-current solver (M8 route:
Neumann g = 2*pi*H0 on r = a, Dirichlet on axis, z-invariant slab).

Parameters: a = 20 mm (diameter 40 mm), sigma = 2.0e6 S/m (M8 cold-state
value), mu_r = 1, f = 50 kHz, H0 = 5000 A/m, constant sigma (cold-state
electromagnetic snapshot, no thermal feedback).

Analytic references (Bessel exact solution, phasor e^{+iwt}):
  H_z(r) = H0 * J0(k r)/J0(k a),   k = (1-i)/delta,
  q(r) = (sigma/2) |E_phi(r)|^2 = (sigma/2) |H0*k/sigma * J1(k r)/J0(k a)|^2,
  P/L: volume integral / Poynting flux / Kelvin ber-bei closed form.
Classical skin-depth formula (planar half-space):
  q_planar(x) = q_surf * exp(-2x/delta),  x = a - r = depth from surface.

Comparison: total P/L, delta_eff (1/e of surface power density ->
delta_eff = 2 * depth_at_1/e), L2/max deviation of normalized profiles.
Extra: linear mu_r = 200 contrast case.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jv, ber, bei, berp, beip

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from cloudfem.mesh.tri_mesh import TriMesh
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, total_joule_power_axi,
    joule_heating_density_axi, MU0)

# ------------------------------------------------------------ parameters
A = 0.02              # cylinder radius [m]  (diameter 40 mm)
SIGMA = 2.0e6         # conductivity [S/m], constant (M8 cold-state value)
FREQ = 50.0e3         # [Hz]
H0 = 5000.0           # solenoid field amplitude [A/m] (constant H0 drive)
LZ = A                # r-z slab height [m]
NZ = 4                # axial divisions
GROWTH = 1.18         # radial geometric growth ratio
LEVELS_HS = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 24.0,
             1.0 / 48.0)
N_BINS = 48           # radial bins for the Q(r) profile (outer 5*delta)

MU_R_MAIN = 1.0
MU_R_HIGH = 200.0


def omega_of(freq):
    return 2.0 * np.pi * freq


def delta_of(freq, mu_r):
    return np.sqrt(2.0 / (omega_of(freq) * MU0 * mu_r * SIGMA))


# ---------------------------------------------------------------- mesh
def graded_slab_mesh(h_surf):
    q = GROWTH
    m = 1
    while h_surf * (q ** m - 1.0) / (q - 1.0) < A:
        m += 1
    h0 = A * (q - 1.0) / (q ** m - 1.0)
    thick = h0 * q ** np.arange(m)
    edges_from_surf = np.concatenate(([0.0], np.cumsum(thick)))
    r_nodes = A - edges_from_surf[::-1]
    nr = len(r_nodes) - 1
    z_nodes = np.linspace(0.0, LZ, NZ + 1)
    nodes = np.array([(r, z) for z in z_nodes for r in r_nodes])

    def idx(i, j):
        return j * (nr + 1) + i

    elements = []
    for j in range(NZ):
        for i in range(nr):
            n0, n1 = idx(i, j), idx(i + 1, j)
            n2, n3 = idx(i, j + 1), idx(i + 1, j + 1)
            elements.append((n0, n1, n3))
            elements.append((n0, n3, n2))
    boundaries = {
        "left": np.array([idx(0, j) for j in range(NZ + 1)]),
        "right": np.array([idx(nr, j) for j in range(NZ + 1)]),
        "bottom": np.array([idx(i, 0) for i in range(nr + 1)]),
        "top": np.array([idx(i, NZ) for i in range(nr + 1)]),
    }
    return TriMesh(nodes, np.array(elements), boundaries), nr, float(thick[0])


# ------------------------------------------------------------------ FEM
def fem_solve(h_surf, mu_r):
    mesh, nr, h_min = graded_slab_mesh(h_surf)
    omega = omega_of(FREQ)
    utilde = solve_eddy_current_axi(
        mesh, nu=1.0 / (MU0 * mu_r), sigma=SIGMA, omega=omega,
        bc={"left": 0.0}, neumann={"right": 2.0 * np.pi * H0})
    p_total = total_joule_power_axi(mesh, utilde, SIGMA, omega)
    q_elem, areas = joule_heating_density_axi(mesh, utilde, SIGMA, omega)
    rc = np.array([mesh.nodes[el][:, 0].mean() for el in mesh.elements])
    return mesh, p_total / LZ, q_elem, rc, areas, nr, h_min


# ----------------------------------------------------------- analytics
def analytic_power_routes(mu_r):
    omega = omega_of(FREQ)
    delta = delta_of(FREQ, mu_r)
    k = (1.0 - 1.0j) / delta

    def e_phi(r):
        return (H0 * k / SIGMA) * jv(1, k * r) / jv(0, k * A)

    xg, wg = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, A, 1201)
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        total += np.sum(w * 0.5 * SIGMA * np.abs(e_phi(r)) ** 2
                        * 2.0 * np.pi * r)
    p_vol = float(total)
    p_poy = float(-np.pi * A * np.real(e_phi(A) * np.conj(H0)))
    x = np.sqrt(2.0) * A / delta
    p_kel = float(np.pi * A * H0 ** 2 * np.sqrt(2.0) / (SIGMA * delta)
                  * (ber(x) * berp(x) + bei(x) * beip(x))
                  / (ber(x) ** 2 + bei(x) ** 2))
    return p_vol, p_poy, p_kel, delta, k


def q_bessel_norm(r, k):
    """Normalized power density profile q(r)/q(a) from Bessel solution."""
    j1r = np.abs(jv(1, k * r)) ** 2
    j1a = float(np.abs(jv(1, k * A)) ** 2)
    out = np.zeros_like(r)
    m = j1a > 0
    out = j1r / j1a
    return out


# ------------------------------------------------- radial profile from FEM
def radial_profile(q_elem, rc, areas, delta, mu_r):
    """Volume-weighted Q(r) binned over the outer 5*delta."""
    r0 = max(A - 5.0 * delta, 0.0)
    edges = np.linspace(r0, A, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vol = 2.0 * np.pi * rc * areas          # element volumes (per Lz=1 slice? )
    # note: vol per element = 2*pi*rc*area over the whole slab; per-unit-
    # length radial density needs division by LZ as well.
    qv = np.zeros(N_BINS)
    for b in range(N_BINS):
        sel = (rc >= edges[b]) & (rc < edges[b + 1] if b < N_BINS - 1
                                  else rc <= edges[b + 1])
        if np.any(sel):
            qv[b] = np.sum(q_elem[sel] * vol[sel]) / np.sum(vol[sel])
    # surface value: linear extrapolation of the two outermost bins
    q_surf = qv[-1] + (qv[-1] - qv[-2]) * 0.5
    qn = qv / q_surf
    # 1/e crossing by linear interpolation on the normalized profile
    depth_1e = np.nan
    for b in range(N_BINS - 1, 0, -1):
        if (qn[b] - 1.0 / np.e) * (qn[b - 1] - 1.0 / np.e) <= 0 \
                and qn[b - 1] != qn[b]:
            t = ((1.0 / np.e) - qn[b]) / (qn[b - 1] - qn[b])
            rcross = centers[b] + t * (centers[b - 1] - centers[b])
            depth_1e = A - rcross
            break
    delta_eff = 2.0 * depth_1e if np.isfinite(depth_1e) else np.nan
    # per-unit-length radial power density d(P/L)/dr [W/m per m of radius]
    dpl_dr = np.zeros(N_BINS)
    dr = edges[1] - edges[0]
    for b in range(N_BINS):
        sel = (rc >= edges[b]) & (rc < edges[b + 1] if b < N_BINS - 1
                                  else rc <= edges[b + 1])
        dpl_dr[b] = np.sum(q_elem[sel] * 2.0 * np.pi * rc[sel]
                           * areas[sel]) / LZ / dr
    return centers, qn, q_surf, depth_1e, delta_eff, dpl_dr, edges


# ------------------------------------------------------------------ main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("BIG-D2  cylinder 50 kHz: FEM vs classical skin-depth formula")
    print("  [all numbers below are simulation outputs]")
    print("  a = 20.0 mm (diameter 40 mm), sigma = 2.0e6 S/m (constant, "
          "cold-state EM snapshot), f = 50 kHz, H0 = 5000 A/m (constant "
          "drive), mu_r = 1 (main) / 200 (contrast)")

    omega = omega_of(FREQ)

    # ---------------- analytic references, mu_r = 1
    p_vol, p_poy, p_kel, delta, k = analytic_power_routes(MU_R_MAIN)
    d1 = abs(p_poy - p_vol) / p_vol
    d2 = abs(p_kel - p_vol) / p_vol
    print(f"\n[mu_r = 1] delta = {delta*1e3:.4f} mm, a/delta = {A/delta:.2f}")
    print(f"  analytic P/L: vol-int = {p_vol:.4f} W/m, "
          f"Poynting = {p_poy:.4f} (reldiff {d1:.2e}), "
          f"Kelvin = {p_kel:.4f} (reldiff {d2:.2e})")
    assert d1 < 1e-8 and d2 < 1e-8, "analytic routes disagree"
    p_ref = p_vol

    # ---------------- FEM convergence, mu_r = 1
    print("  level  h_surf/delta   nr  elements   nodes   "
          "P_FEM [W/m]   rel err")
    rows = []
    best = None
    for lev, frac in enumerate(LEVELS_HS, start=1):
        mesh, p_fem, q_elem, rc, areas, nr, h_min = \
            fem_solve(frac * delta, MU_R_MAIN)
        err = (p_fem - p_ref) / p_ref
        rows.append((lev, h_min / delta, nr, mesh.n_elements,
                     mesh.n_nodes, p_fem, err))
        print(f"  {lev:5d}  {h_min/delta:12.4f}  {nr:3d}  "
              f"{mesh.n_elements:8d}  {mesh.n_nodes:6d}   "
              f"{p_fem:11.3f}   {err*100:+.3f}%")
        best = (mesh, p_fem, q_elem, rc, areas, nr, h_min)
    errs = [abs(r[6]) for r in rows]
    rates = [np.log(errs[i] / errs[i + 1]) /
             np.log(rows[i][1] / rows[i + 1][1]) for i in range(len(rows) - 1)]
    print(f"  convergence rates (in h_surf): "
          f"{[f'{r:.2f}' for r in rates]}")
    print(f"  finest error = {rows[-1][6]*100:+.3f}%")

    # ---------------- radial profile, mu_r = 1
    mesh, p_fem, q_elem, rc, areas, nr, h_min = best
    centers, qn, q_surf, depth_1e, delta_eff, dpl_dr, edges = \
        radial_profile(q_elem, rc, areas, delta, MU_R_MAIN)
    x_skin = (A - centers) / delta                       # depth in delta
    q_bes = q_bessel_norm(centers, k)
    q_pla = np.exp(-2.0 * x_skin)
    sel = x_skin <= 4.0
    l2_bes = float(np.sqrt(np.mean((qn[sel] - q_bes[sel]) ** 2)))
    max_bes = float(np.max(np.abs(qn[sel] - q_bes[sel])))
    l2_pla = float(np.sqrt(np.mean((qn[sel] - q_pla[sel]) ** 2)))
    max_pla = float(np.max(np.abs(qn[sel] - q_pla[sel])))
    err_delta = (delta_eff - delta) / delta
    err_pl = (p_fem - p_ref) / p_ref
    print(f"\n[mu_r = 1, finest mesh] profile comparison (outer 4*delta):")
    print(f"  q_surf (FEM, extrapolated) = {q_surf:.4e} W/m^3")
    print(f"  depth at q/q_surf = 1/e: {depth_1e*1e3:.4f} mm -> "
          f"delta_eff = {delta_eff*1e3:.4f} mm, "
          f"vs analytic delta = {delta*1e3:.4f} mm "
          f"(err {err_delta*100:+.2f}%)")
    print(f"  P/L: FEM = {p_fem:.3f} W/m vs Kelvin = {p_ref:.3f} W/m "
          f"(err {err_pl*100:+.3f}%)")
    print(f"  normalized profile vs Bessel exact: L2 = {l2_bes:.3e}, "
          f"max dev = {max_bes:.3e}")
    print(f"  normalized profile vs planar e^(-2x/delta): L2 = {l2_pla:.3e}, "
          f"max dev = {max_pla:.3e}")

    np.savetxt(os.path.join(here, "big_d2_profile_mu1.csv"),
               np.column_stack([centers * 1e3, x_skin, qn, q_bes, q_pla,
                                dpl_dr]),
               delimiter=",",
               header="r_mm,depth_in_delta,q_FEM_norm,q_Bessel_norm,"
                      "q_planar_norm,dPL_dr_W_per_m_per_m",
               comments="")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(x_skin, qn, "o-", ms=3, label="FEM (volume-weighted bins)")
    ax.semilogy(x_skin, q_bes, "-", label="Bessel exact (cylinder)")
    ax.semilogy(x_skin, q_pla, "--", label="planar e^(-2x/delta)")
    ax.axhline(1.0 / np.e, color="gray", lw=0.8, ls=":")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("depth from surface / delta")
    ax.set_ylabel("q(r) / q_surf  [normalized power density]")
    ax.set_title("BIG-D2: radial power-density profile, mu_r = 1, f = 50 kHz\n"
                 "(simulation outputs)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "big_d2_profile_mu1.png"), dpi=120)
    plt.close(fig)

    # ---------------- contrast: linear mu_r = 200
    p_vol2, p_poy2, p_kel2, delta2, k2 = analytic_power_routes(MU_R_HIGH)
    d1b = abs(p_poy2 - p_vol2) / p_vol2
    d2b = abs(p_kel2 - p_vol2) / p_vol2
    print(f"\n[mu_r = 200] delta = {delta2*1e6:.2f} um, "
          f"a/delta = {A/delta2:.1f}")
    print(f"  analytic P/L: vol-int = {p_vol2:.3f} W/m, "
          f"Poynting reldiff {d1b:.2e}, Kelvin reldiff {d2b:.2e}")
    assert d1b < 1e-8 and d2b < 1e-8, "analytic routes disagree (mu_r=200)"
    mesh2, p_fem2, q_elem2, rc2, areas2, nr2, h_min2 = \
        fem_solve(delta2 / 24.0, MU_R_HIGH)
    err2 = (p_fem2 - p_kel2) / p_kel2
    centers2, qn2, q_surf2, depth2, deff2, dpl2, edges2 = \
        radial_profile(q_elem2, rc2, areas2, delta2, MU_R_HIGH)
    x2 = (A - centers2) / delta2
    q_bes2 = q_bessel_norm(centers2, k2)
    q_pla2 = np.exp(-2.0 * x2)
    sel2 = x2 <= 4.0
    l2_pla2 = float(np.sqrt(np.mean((qn2[sel2] - q_pla2[sel2]) ** 2)))
    max_pla2 = float(np.max(np.abs(qn2[sel2] - q_pla2[sel2])))
    err_d2 = (deff2 - delta2) / delta2
    print(f"  FEM (h_surf=delta/24, {mesh2.n_elements} elem): "
          f"P/L = {p_fem2:.3f} W/m vs Kelvin = {p_kel2:.3f} W/m "
          f"(err {err2*100:+.3f}%)")
    print(f"  delta_eff = {deff2*1e6:.2f} um vs delta = {delta2*1e6:.2f} um "
          f"(err {err_d2*100:+.2f}%)")
    print(f"  profile vs planar e^(-2x/delta): L2 = {l2_pla2:.3e}, "
          f"max dev = {max_pla2:.3e}")
    print(f"  P/L ratio (mu_r=200 / mu_r=1): FEM {p_fem2/p_fem:.2f}, "
          f"analytic {p_kel2/p_ref:.2f} (planar scaling ~ sqrt(200) = 14.14)")
    b_surf = MU0 * MU_R_HIGH * H0
    print(f"  surface B (linear) = {b_surf:.3f} T (< Bsat 1.9 T: "
          f"linear regime self-consistent at H0 = {H0:.0f} A/m)")

    np.savetxt(os.path.join(here, "big_d2_profile_mu200.csv"),
               np.column_stack([centers2 * 1e3, x2, qn2, q_bes2, q_pla2]),
               delimiter=",",
               header="r_mm,depth_in_delta,q_FEM_norm,q_Bessel_norm,"
                      "q_planar_norm",
               comments="")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(x2, qn2, "o-", ms=3, label="FEM (volume-weighted bins)")
    ax.semilogy(x2, q_bes2, "-", label="Bessel exact (cylinder)")
    ax.semilogy(x2, q_pla2, "--", label="planar e^(-2x/delta)")
    ax.axhline(1.0 / np.e, color="gray", lw=0.8, ls=":")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("depth from surface / delta")
    ax.set_ylabel("q(r) / q_surf  [normalized power density]")
    ax.set_title("BIG-D2: radial power-density profile, mu_r = 200, "
                 "f = 50 kHz\n(simulation outputs)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "big_d2_profile_mu200.png"), dpi=120)
    plt.close(fig)

    # ---------------- summary line for the report
    summary = dict(
        delta_mm=delta * 1e3, a_over_delta=A / delta,
        p_kelvin=p_ref, p_fem_finest=p_fem, err_pl_pct=err_pl * 100,
        delta_eff_mm=delta_eff * 1e3, err_delta_pct=err_delta * 100,
        l2_bessel=l2_bes, max_bessel=max_bes,
        l2_planar=l2_pla, max_planar=max_pla,
        delta2_um=delta2 * 1e6, a_over_delta2=A / delta2,
        p_kelvin2=p_kel2, p_fem2=p_fem2, err_pl2_pct=err2 * 100,
        delta_eff2_um=deff2 * 1e6, err_delta2_pct=err_d2 * 100,
        l2_planar2=l2_pla2, max_planar2=max_pla2,
        conv_rates=rates, finest_levels=[r[6] * 100 for r in rows],
    )
    np.save(os.path.join(here, "big_d2_summary.npy"),
            np.array([0]))  # placeholder; real table below
    with open(os.path.join(here, "big_d2_summary.txt"), "w") as f:
        for kk, vv in summary.items():
            f.write(f"{kk} = {vv}\n")
    # ---- BIG-M11 benchmark criterion: finest-mesh P/L error < 1 %
    # in both the mu_r = 1 main case and the mu_r = 200 contrast case
    # (errors computed above, unchanged).
    ok_pl = abs(err_pl) < 0.01 and abs(err2) < 0.01
    print(f"VERDICT: P/L errors mu_r=1 {err_pl * 100:+.3f}%, "
          f"mu_r=200 {err2 * 100:+.3f}% (criterion < 1%) -> "
          f"{'PASS' if ok_pl else 'FAIL'}")
    assert ok_pl, "BIG-D2 P/L criterion failed"
    print("BIG-D2 BENCHMARK PASSED")


if __name__ == "__main__":
    main()
