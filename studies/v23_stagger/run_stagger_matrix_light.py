#!/usr/bin/env python3
"""v2.3 (B) stagger 交错耦合判据研究 —— 轻算例冒烟矩阵。

Scope: ~/workspace/cloudfem_v23_plan/V23_SCOPE.md
基线：main@7f57cc78 的 induction_heating_3d_tet10（物理代码不改）；
本工作树 ~/workspace/cloudfem 当前检出 v2.1-cp-enthalpy@1b7d26f，
经 git diff 确认 1b7d26f..7f57cc7 仅差 README 一行，物理代码逐位一致。

轻算例：S4 级加热短算例（t_end=0.1, dt=0.02, em_every=5 → 1 个 EM 区间），
几何沿用 main 分支 tests/test_tet10_em_thermal_coupled.py 的同款
（铁柱 R=15mm/L=60mm，气盒 half=0.045，h=0.016，f=2kHz，B_applied 驱动
H0=1e6 A/m，T0=20°C，cp(T) 律默认关闭=常 rho_cp，绝热驱动）。

矩阵：stagger_iter ∈ {1,2,3} × stagger_tol ∈ {None, 1e-2, 1e-3}（9 组）。

记录（全部为仿真输出）：
- 每区间 stagger_passes / stagger_dT（驱动 history 原样）
- 收缩分析：history 只存每区间最终 dT（s≥1 轮间的最后一次
  max|T_trial(s)-T_trial(s-1)|）。iter=2 存 d1=|T2−T1|，iter=3（tol=None）
  存 d2=|T3−T2|；跨跑比值 d2/d1 ≈ 同区间定点迭代收缩因子（R1 要求）。
- Tmax/E_in 相对 iter=1 基线的变化、EM 求解次数、wall 时间。

输出：JSON 到 ~/workspace/cloudfem_v23_plan/data/stagger_light_matrix.json。
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.expanduser("~/workspace/cloudfem")
sys.path.insert(0, REPO)

from cloudfem.mesh.tet10_mesh import tet4_to_tet10          # noqa: E402
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh  # noqa: E402
from cloudfem.physics.em import MU0                         # noqa: E402
from cloudfem.physics.induction3d_tet10 import (            # noqa: E402
    induction_heating_3d_tet10)

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
CASE = dict(t_end=0.1, dt=0.02, em_every=5, T0=20.0)

_ITERS = (1, 2, 3)
_TOLS = (None, 1e-2, 1e-3)

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def run_case(stagger_iter, stagger_tol):
    omega = 2 * PI * GEO["FREQ"]
    kw = dict(mesh=_mesh(), omega=omega, iron=_in_iron, T0=CASE["T0"],
              t_end=CASE["t_end"], dt=CASE["dt"], em_every=CASE["em_every"],
              stagger_iter=stagger_iter, stagger_tol=stagger_tol,
              B_applied=(0.0, 0.0, MU0 * 1.0e6))
    t0 = time.perf_counter()
    res = induction_heating_3d_tet10(**kw)
    wall = time.perf_counter() - t0
    hist = res["history"]
    Tmax = float(np.max(res["T_hist"]))
    return {
        "stagger_iter": stagger_iter,
        "stagger_tol": stagger_tol,
        "n_intervals": len(hist),
        "stagger_passes": [h["stagger_passes"] for h in hist],
        "stagger_dT": [h["stagger_dT"] for h in hist],
        "P_per_interval": [h["P"] for h in hist],
        "em_picard_iters": [h["em_iterations"] for h in hist],
        "em_converged": [bool(h["em_converged"]) for h in hist],
        "total_em_solves": int(sum(h["stagger_passes"] for h in hist)),
        "total_picard_iters": int(sum(h["em_iterations"] for h in hist)),
        "Tmax_degC": Tmax,
        "E_in_J": float(res["E_in"]),
        "balance_rel": float(res["balance_rel"]),
        "wall_s": wall,
        "unit": "simulation outputs (仿真输出)",
    }


def main():
    out_dir = os.path.expanduser("~/workspace/cloudfem_v23_plan/data")
    os.makedirs(out_dir, exist_ok=True)
    cases = []
    for it in _ITERS:
        for tol in _TOLS:
            print(f"[run] stagger_iter={it} stagger_tol={tol} ...", flush=True)
            c = run_case(it, tol)
            cases.append(c)
            print(f"      Tmax={c['Tmax_degC']:.2f} C E_in={c['E_in_J']:.1f} J "
                  f"passes={c['stagger_passes']} dT={c['stagger_dT']} "
                  f"solves={c['total_em_solves']} wall={c['wall_s']:.0f}s",
                  flush=True)

    base = next(c for c in cases
                if c["stagger_iter"] == 1 and c["stagger_tol"] is None)
    T0b, E0b = base["Tmax_degC"], base["E_in_J"]
    for c in cases:
        c["rel_Tmax_vs_iter1"] = (c["Tmax_degC"] - T0b) / abs(T0b)
        c["rel_Ein_vs_iter1"] = (c["E_in_J"] - E0b) / abs(E0b)

    # 收缩分析（跨跑比值 d2(iter=3)/d1(iter=2)，tol=None，逐区间）
    contraction = {}
    for idx in range(base["n_intervals"]):
        try:
            d2 = next(c for c in cases
                      if c["stagger_iter"] == 3 and c["stagger_tol"] is None
                      )["stagger_dT"][idx]
            d1 = next(c for c in cases
                      if c["stagger_iter"] == 2 and c["stagger_tol"] is None
                      )["stagger_dT"][idx]
            contraction[f"interval_{idx}"] = {
                "d1_iter2": d1, "d2_iter3": d2,
                "factor": (d2 / d1) if d1 else None,
                "contracts": (d2 < d1) if (d1 and d2 is not None) else None,
            }
        except (StopIteration, IndexError, TypeError):
            contraction[f"interval_{idx}"] = {"error": "missing data"}
    payload = {
        "meta": {
            "study": "v2.3 B stagger light smoke (S4-level, t_end=0.1, "
                     "dt=0.02, em_every=5)",
            "code_base": "main@7f57cc7 (checkout 1b7d26f, diff 1b7d26f.."
                         "7f57cc7 = README only; physics identical)",
            "repo": REPO,
            "geometry": "iron cylinder R=15mm L=60mm, air box half=0.045, "
                        "h=0.016; f=2kHz B_applied H0=1e6 A/m; T0=20C; "
                        "cp(T) law default-off (constant rho_cp); adiabatic",
            "numpy": np.__version__,
            "ran_at_utc": datetime.now(timezone.utc).isoformat(),
            "unit": "all numbers are simulation outputs (仿真输出)",
        },
        "cases": cases,
        "contraction_cross_run": contraction,
    }
    path = os.path.join(out_dir, "stagger_light_matrix.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
