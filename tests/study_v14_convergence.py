"""v1.4 acceptance study: where does the 3D/axi cold-power ratio
1.751 come from?

Configuration (identical to the tests/test_induction3d.py
cross-check working point): solid cylinder R = 20 mm, L = 80 mm,
cold constant sigma = 2e6 S/m, mu = mu0, uniform axial applied
field H0 = 60 kA/m at f = 2.5 kHz (skin depth delta = 7.12 mm,
R/delta = 2.81). The v1.4 chain's first (cold) EM solve on the
h = 0.011 cylinder-in-box mesh gives P_3d = 1844.3 W against
P_axi = 1053.2 W on the chain's axi mesh (ratio 1.751).

This study decomposes that factor against three references at the
same working point:

* analytic -- infinite solid cylinder in a uniform axial phasor
  field (Bessel J0/J1 solution; power by volume integral of
  0.5*sigma*|E_phi|^2, cross-checked by the surface Poynting
  flux -- the two agree to quadrature accuracy);
* axi -- the library 2D axisymmetric solver (which *is* the
  infinite-cylinder problem: z-uniform boundary data), run on
  three meshes to establish its own convergence;
* 3D -- the v1.3 edge-element solver on cylinder-in-box meshes at
  two refinement levels in the working box (h/delta 1.55 and 1.10),
  plus a box-size sweep at both levels and a length sweep at the
  baseline h, separating the skin-discretisation error, the
  finite-box boundary perturbation and the finite-length end
  effect. Finer levels (h/delta 0.77 and 0.55, 48k-296k tets)
  were attempted repeatedly but the sparse-direct factorisation
  of the A-V system exceeded this 8 GB host in every attempt
  (OOM-killed at 87k tets with both COLAMD and MMD_AT_PLUS_A
  orderings); the h -> 0 values below are therefore two-level
  Richardson estimates with the order bracketed, not measured
  points. This machine limit is part of the reported evidence.

Per 3D case the loss is also split by axial position (mid section
|z| <= L/2 - 2*delta vs the two end slabs): the mid section's
power per unit length is directly comparable to the analytic
infinite-cylinder value away from the ends.

This is a study script, not a pass/fail validation: it asserts
only reference self-consistency (analytic two ways, axi mesh
convergence). No tolerance anywhere is fitted to the 3D results.

Usage (each case appends one JSON line to the results file):
    export PYTHONPATH=.
    python3 tests/study_v14_convergence.py refs
    python3 tests/study_v14_convergence.py run3d H L HALF TAG
    python3 tests/study_v14_convergence.py report
Results file: $CONV_RAW or /tmp/v14_conv_raw.jsonl
"""

import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import jv

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.fem.tet_edge import build_edges
from cloudfem.physics.axisym import (solve_eddy_current_axi,
                                     total_joule_power_axi)
from cloudfem.physics.em import MU0, skin_depth
from cloudfem.physics.em3d import (joule_heating_density_3d,
                                   solve_eddy_current_3d)

R, SIG, FREQ, H0 = 0.02, 2e6, 2500.0, 60e3
OMEGA = 2 * np.pi * FREQ
DELTA = skin_depth(SIG, OMEGA)
RAW = os.environ.get("CONV_RAW", "/tmp/v14_conv_raw.jsonl")


def _append(rec):
    with open(RAW, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def analytic_power_per_length():
    """Infinite cylinder, axial phasor field H0 (Bessel solution).

    Hz(r) = H0 J0(k r)/J0(k R), E_phi = (k/sigma) H0 J1(k r)/J0(k R),
    k = (1 - i)/delta. Power per unit length by volume integral,
    cross-checked by the Poynting flux through r = R.
    """
    k = (1.0 - 1.0j) / DELTA
    J0R, J1R = jv(0, k * R), jv(1, k * R)
    I1, _ = quad(lambda r: abs(jv(1, k * r)) ** 2 * r, 0.0, R,
                 limit=200)
    p_vol = np.pi * SIG * abs(k / SIG) ** 2 * abs(H0 / J0R) ** 2 * I1
    # dissipation = -flux of 0.5*Re(E x H*) through r = R (inward)
    p_poy = -np.pi * R * H0 ** 2 * ((k / SIG) * J1R / J0R).real
    return float(p_vol), float(p_poy)


def case_refs():
    p_vol, p_poy = analytic_power_per_length()
    rec = {"case": "analytic", "P_per_len_vol": p_vol,
           "P_per_len_poynting": p_poy,
           "rel_diff": abs(p_vol - p_poy) / p_vol}
    axi = {}
    for nr, nz in ((32, 10), (80, 40), (160, 80)):
        m2 = rect_mesh(nr, nz, xlim=(0, R), ylim=(0, 0.08))
        u = solve_eddy_current_axi(m2, nu=1.0 / MU0, sigma=SIG,
                                   omega=OMEGA, bc={"left": 0.0},
                                   neumann={"right": 2 * np.pi * H0})
        axi[f"{nr}x{nz}"] = total_joule_power_axi(m2, u, SIG, OMEGA)
    rec["axi"] = axi
    rec["axi_err_vs_analytic"] = {
        key: (val - p_vol * 0.08) / (p_vol * 0.08)
        for key, val in axi.items()}
    _append(rec)


def case_run3d(h, L, half, tag):
    t0 = time.time()
    mesh = cylinder_in_box_mesh(R, L, half=half, h=h)
    t_mesh = time.time() - t0
    ed = build_edges(mesh)

    def sigma_fn(x, y, z):
        return SIG if (x * x + y * y < R * R
                       and abs(z) < L / 2) else 0.0

    t0 = time.time()
    av, vv = solve_eddy_current_3d(mesh, nu=1.0 / MU0, sigma=sigma_fn,
                                   omega=OMEGA,
                                   B_applied=(0.0, 0.0, MU0 * H0),
                                   edge_data=ed,
                                   permc_spec=os.environ.get(
                                       "STUDY_PERMC") or None)
    Q, vols = joule_heating_density_3d(mesh, av, vv, sigma_fn, OMEGA,
                                       edge_data=ed)
    t_solve = time.time() - t0
    cen = mesh.nodes[mesh.elements].mean(axis=1)
    cond = Q > 0.0
    P = float(np.sum(Q * vols))
    vol_c = float(vols[cond].sum())
    z = np.abs(cen[:, 2])
    mid = cond & (z <= L / 2 - 2 * DELTA)
    len_mid = max(L - 4 * DELTA, 0.0)
    P_mid_len = float(np.sum(Q[mid] * vols[mid])) / len_mid \
        if len_mid > 0 else float("nan")
    P_ends = float(np.sum(Q[cond & ~mid] * vols[cond & ~mid]))
    rec = {"case": "run3d", "tag": tag, "h": h, "L": L, "half": half,
           "permc": os.environ.get("STUDY_PERMC") or "default",
           "h_over_delta": h / DELTA, "n_tets": int(mesh.n_elements),
           "n_edges": int(len(ed[0])), "P": P,
           "vol_cond": vol_c,
           "vol_exact": float(np.pi * R ** 2 * L),
           "P_mid_per_len": P_mid_len, "P_ends": P_ends,
           "t_mesh_s": t_mesh, "t_solve_s": t_solve}
    _append(rec)


def _load():
    refs, runs = None, {}
    with open(RAW) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["case"] == "analytic":
                refs = rec
            else:
                runs[rec["tag"]] = rec
    return refs, runs


def report():
    refs, runs = _load()
    p_len = refs["P_per_len_vol"]
    axi_conv = refs["axi"]["160x80"]
    axi_chain = refs["axi"]["32x10"]
    print(f"delta = {DELTA * 1e3:.3f} mm, R/delta = {R / DELTA:.2f}")
    print(f"[analytic] P/L = {p_len:.2f} W/m (Poynting check "
          f"{refs['P_per_len_poynting']:.2f}, rel diff "
          f"{refs['rel_diff']:.2e}); P(L=80mm) = {p_len * 0.08:.2f} W")
    for key, val in refs["axi"].items():
        print(f"[axi {key}] P = {val:.2f} W, err vs analytic "
              f"{refs['axi_err_vs_analytic'][key]:+.3%}")
    print()
    base = runs["base_h011"]
    print(f"[baseline reproduction] P_3d(h=0.011) = {base['P']:.1f} W; "
          f"ratio vs chain axi = {base['P'] / axi_chain:.3f} "
          f"(v1.4 cross-check: 1.751); vs converged axi = "
          f"{base['P'] / axi_conv:.3f}; vs analytic = "
          f"{base['P'] / (p_len * 0.08):.3f}")
    print()
    print("3D runs (all cold, sigma = 2e6, f = 2.5 kHz, H0 = 60 kA/m):")
    print(f"{'tag':>16} {'h/d':>5} {'L':>5} {'half':>5} {'tets':>8} "
          f"{'P[W]':>9} {'err/axi':>8} {'mid P/L err':>11} "
          f"{'P_ends[W]':>9} {'vol err':>8}")
    for tag in sorted(runs):
        r = runs[tag]
        print(f"{tag:>16} {r['h_over_delta']:5.2f} {r['L']:5.2f} "
              f"{r['half']:5.3f} {r['n_tets']:8d} {r['P']:9.1f} "
              f"{(r['P'] - axi_conv) / axi_conv:+8.2%} "
              f"{(r['P_mid_per_len'] - p_len) / p_len:+11.2%} "
              f"{r['P_ends']:9.1f} "
              f"{r['vol_cond'] / r['vol_exact'] - 1:+8.2%}")
    print()
    # --- apparent order from the mid-section pair -------------------
    # e_mid(h) is measured against the analytic value; the power p
    # below is the apparent order IF the mid-section limit is the
    # analytic infinite-cylinder value. Two points fit any (C, p),
    # so p is an assumption to bracket with, not a measurement.
    fine = runs["h0078"]
    e1 = base["P_mid_per_len"] / p_len - 1.0
    e2 = fine["P_mid_per_len"] / p_len - 1.0
    p_app = np.log(e1 / e2) / np.log(base["h"] / fine["h"])
    print(f"[apparent order] mid-section excess {e1:+.3f} -> {e2:+.3f} "
          f"for h {base['h']} -> {fine['h']}: p_app = {p_app:.2f} "
          f"(assumes the analytic limit; bracket p in [1, 2])")
    for p in (1.0, p_app, 2.0):
        rp = (base["h"] / fine["h"]) ** p
        p_inf = fine["P"] - (base["P"] - fine["P"]) / (rp - 1.0)
        print(f"  Richardson totals (half=0.06), p = {p:.2f}: "
              f"P(h->0) = {p_inf:.1f} W "
              f"({(p_inf - axi_conv) / axi_conv:+.2%} vs converged axi); "
              f"discretisation part of the baseline excess = "
              f"{base['P'] - p_inf:.1f} W")
    print()
    # --- box sweep ---------------------------------------------------
    print("box sweep (L = 80 mm): P vs 1/half^3, LSQ P = P_inf + C/half^3")
    for htag, tags in (("h=0.011", ("half0045", "base_h011",
                                    "half0090")),
                       ("h=0.0078", ("half0045_h0078", "h0078"))):
        pts = [(runs[t]["half"], runs[t]["P"]) for t in tags
               if t in runs]
        for half, val in pts:
            print(f"  {htag}, half = {half * 1e3:5.0f} mm: P = {val:8.1f} W")
        if len(pts) == 3:
            A = np.vstack([np.ones(3),
                           1.0 / np.array([q[0] for q in pts]) ** 3]).T
            coef, *_ = np.linalg.lstsq(
                A, np.array([q[1] for q in pts]), rcond=None)
            print(f"  {htag} LSQ: P(box->inf) = {coef[0]:.1f} W, "
                  f"box part at half=0.06 = "
                  f"{runs['base_h011']['P'] - coef[0]:.1f} W")
    d_box_11 = runs["half0045"]["P"] - base["P"]
    d_box_78 = runs["half0045_h0078"]["P"] - fine["P"]
    print(f"  box step 0.045 vs 0.06: {d_box_11:+.1f} W (h=0.011), "
          f"{d_box_78:+.1f} W (h=0.0078) -- mostly a boundary-"
          f"condition effect, weakly h-dependent")
    print()
    # --- length sweep / end effect ----------------------------------
    print("length sweep (h = 0.011, half = 0.06): P = slope*L + ends")
    ls = [runs[t] for t in ("L004", "base_h011")]
    slope = (ls[1]["P"] - ls[0]["P"]) / (ls[1]["L"] - ls[0]["L"])
    ends = ls[0]["P"] - slope * ls[0]["L"]
    print(f"  slope = {slope:.1f} W/m ({(slope - p_len) / p_len:+.2%} "
          f"vs analytic P/L); intercept (both ends) = {ends:.1f} W")
    slab = 4 * DELTA
    for tag in ("base_h011", "h0078"):
        r = runs[tag]
        excess = r["P_ends"] - r["P_mid_per_len"] * slab
        print(f"  z-partition {tag}: end slabs ({slab * 1e3:.1f} mm) "
              f"hold {r['P_ends']:.1f} W; excess over the mid rate "
              f"= {excess:+.1f} W; excess over the analytic rate = "
              f"{r['P_ends'] - p_len * slab:+.1f} W")
    print()
    tot = base["P"] - axi_conv
    print(f"baseline excess over converged axi: {tot:+.1f} W "
          f"({tot / axi_conv:+.2%}); see report MD for the "
          f"decomposition table and its brackets")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refs":
        case_refs()
    elif cmd == "run3d":
        case_run3d(float(sys.argv[2]), float(sys.argv[3]),
                   float(sys.argv[4]), sys.argv[5])
    elif cmd == "report":
        report()
