"""v2.0 convergence study (DT + EM-update-interval refinement).

NOT part of the repo test suite. One invocation runs ONE case:
  python3 dt_em_conv.py --dt 0.02 --em-every 5 --case A1
Setup identical to tests/test_tet10_em_thermal_coupled.py::_coupled_run:
  iron cylinder R=15mm L=60mm in air box, B_applied H0=1e6 A/m,
  f=2kHz, t_end=0.4s, adiabatic, em_tol=1e-3, em_max_iter=60,
  em_relax=0.5, BE heat integration, stagger_iter=1.

Records per case (all simulation outputs):
  Tmax_end_C (degC)      : max iron-element temperature at t_end
  E_in_J                : accumulated EM input energy over the run
  mu_r_surf_mean_last   : surface-band mean secant mu_r at last EM solve
  P_last_W              : total Joule power at last EM solve
  T_surf_peak_C         : peak surface-band element temperature over the
                          whole run, sampled at EM interval ends
                          (same sampling as the v2.0 test's own crossing
                          assertion via T_elem_hist)
  t_cross_Tc_s          : first EM-interval-end time (in res["times"])
                          where the surface-band mean temperature > 770 C
                          (Tc); "未穿越" if never crossed within t_end
  P_total_last_W        : total Joule power at last EM interval (= P_last_W)
Result JSON -> results/<case>.json (idempotent: skips if present).
Schema v2 (2026-10-22 14:45 CST): added T_surf_peak_C, t_cross_Tc_s,
P_total_last_W.  stagger_iter=1 (butler-confirmed OK); analytic M10
laws unchanged (no lookup-table study needed).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d_tet10 import induction_heating_3d_tet10

GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
TC = 770.0


def build_mesh():
    m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                              h=GEO["H"])
    return tet4_to_tet10(m4)


def iron_mask_of(mesh):
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    mask = np.array([(x * x + y * y < GEO["R"] ** 2)
                     and abs(z) < GEO["L"] / 2 for x, y, z in cen])
    return mask, cen


def surface_band(mesh, iron_mask, cen):
    cen_i = cen[iron_mask]
    r_c = np.hypot(cen_i[:, 0], cen_i[:, 1])
    return (r_c > 0.75 * GEO["R"]) & (np.abs(cen_i[:, 2]) < 0.01)


def run_case(case_id, dt, em_every, t_end=0.4):
    t0 = time.time()
    mesh = build_mesh()
    iron_mask, cen = iron_mask_of(mesh)
    surf = surface_band(mesh, iron_mask, cen)
    omega = 2 * np.pi * GEO["FREQ"]

    def in_iron(x, y, z):
        return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2

    res = induction_heating_3d_tet10(
        mesh, omega, B_applied=(0.0, 0.0, MU0 * 1.0e6),
        iron=in_iron, T0=20.0, t_end=t_end, dt=dt, em_every=em_every,
        em_tol=1e-3, em_max_iter=60, em_relax=0.5)
    hist = res["history"]
    assert all(h["em_converged"] for h in hist), "EM must converge"
    last = hist[-1]
    Te_hist = np.asarray(res["T_elem_hist"])   # rows: EM interval ends
    times = np.asarray(res["times"])
    assert Te_hist.shape[0] == len(times) == len(hist) + 1
    surf_peak = float(Te_hist[:, surf].max())
    surf_mean = Te_hist[:, surf].mean(axis=1)
    cross = [i for i in range(1, len(surf_mean)) if surf_mean[i] > TC]
    t_cross = float(times[cross[0]]) if cross else "未穿越"
    out = {
        "case": case_id,
        "dt": dt,
        "em_every": em_every,
        "t_end": t_end,
        "n_em_intervals": len(hist),
        "n_heat_steps": int(round(t_end / dt)),
        "Tmax_end_C": float(last["T_elem_max"]),
        "E_in_J": float(res["E_in"]),
        "mu_r_surf_mean_last": float(last["mu_r_iron"][surf].mean()),
        "P_last_W": float(last["P"]),
        "P_total_last_W": float(last["P"]),
        "T_surf_peak_C": surf_peak,
        "t_cross_Tc_s": t_cross,
        "em_iterations": [int(h["em_iterations"]) for h in hist],
        "P_first_W": float(hist[0]["P"]),
        "balance_rel": float(res["balance_rel"]),
        "wall_s": float(time.time() - t0),
        "rc": 0,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--em-every", type=int, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser(
        "~/workspace/v20_conv/results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.case}.json")
    if os.path.exists(path):
        print(f"SKIP {args.case}: {path} already exists", flush=True)
        return
    print(f"START case={args.case} dt={args.dt} em_every={args.em_every}",
          flush=True)
    out = run_case(args.case, args.dt, args.em_every)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE case={args.case} wall={out['wall_s']:.0f}s "
          f"Tmax={out['Tmax_end_C']:.1f}C E_in={out['E_in_J']:.6e}J "
          f"mu_surf={out['mu_r_surf_mean_last']:.3f} "
          f"P_last={out['P_last_W']:.4e}W "
          f"T_surf_peak={out['T_surf_peak_C']:.1f}C "
          f"t_cross_Tc={out['t_cross_Tc_s']}", flush=True)


if __name__ == "__main__":
    main()
