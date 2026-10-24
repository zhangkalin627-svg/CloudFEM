#!/usr/bin/env python3
"""v2.4 (diag) 第一阶段 —— μr 可得性冒烟验证。

Scope: ~/workspace/cloudfem_v24_plan/V24_SCOPE.md §7 风险1。
基线：main@7f57cc7 的 induction_heating_3d_tet10（物理代码不改，
本脚本只读驱动返回做验证）。

S4 级轻算例（同 v2.3 轻算例口径）：t_end=0.1, dt=0.02, em_every=5 →
1 个 EM 区间；铁柱 R=15mm/L=60mm，气盒 half=0.045，h=0.016，
f=2kHz，B_applied 驱动 H0=1e6 A/m，T0=20°C，cp(T) 律默认关闭，
绝热驱动。

验证问题（§7.1）：割线 μr 是 EM 求解内部量，驱动侧是否现成可得？
三种可能结论：
  A) μr 可得（说明从哪拿到）；
  B) 需扩 EM 出参（列出精确改动清单）；
  C) 驱动侧后处理重算（说明方法与差异声明）。
结论写 studies/v24_diag/SMOKE_MUR.md。
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.expanduser("~/workspace/cloudfem_v20")
sys.path.insert(0, REPO)

from cloudfem.mesh.tet10_mesh import tet4_to_tet10          # noqa: E402
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh  # noqa: E402
from cloudfem.physics.em import MU0                         # noqa: E402
from cloudfem.physics.induction3d_tet10 import (            # noqa: E402
    induction_heating_3d_tet10)
from cloudfem.physics import induction3d_tet10 as drv       # noqa: E402

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
CASE = dict(t_end=0.1, dt=0.02, em_every=5, T0=20.0)

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def run_case(stagger_iter):
    omega = 2 * PI * GEO["FREQ"]
    kw = dict(mesh=_mesh(), omega=omega, iron=_in_iron, T0=CASE["T0"],
              t_end=CASE["t_end"], dt=CASE["dt"], em_every=CASE["em_every"],
              stagger_iter=stagger_iter, stagger_tol=None,
              B_applied=(0.0, 0.0, MU0 * 1.0e6))
    t0 = time.perf_counter()
    res = induction_heating_3d_tet10(**kw)
    wall = time.perf_counter() - t0
    return res, wall


def main():
    out_dir = os.path.expanduser("~/workspace/cloudfem_v20/studies/v24_diag")
    os.makedirs(out_dir, exist_ok=True)

    t_curie = float(drv.TC_CURIE)
    print(f"t_curie = {t_curie} degC (driver law param)", flush=True)

    # iter=3: 让末 pass 的 μr 与区间末温度场必然不同态，检验取样点差异
    res3, wall3 = run_case(3)
    res1, wall1 = run_case(1)

    checks = {}
    h = res3["history"][0]
    iron_idx = res3["iron_idx"]
    sub = res3["submesh"]
    vols = np.asarray(sub.element_volumes())

    # 1) 驱动侧 history 是否现成有 μr
    for key in ("mu_r_iron", "mu_r_iron_mean", "mu_r_iron_min",
                "frac_above_Tc", "Q_iron", "stagger_passes", "stagger_dT"):
        checks[f"history_has_{key}"] = key in h
    checks["mu_r_iron_is_ndarray"] = isinstance(h["mu_r_iron"], np.ndarray)
    checks["mu_r_iron_len"] = int(len(h["mu_r_iron"]))
    checks["iron_idx_len"] = int(len(iron_idx))
    checks["vols_len"] = int(len(vols))

    # 2) history 的 mu_r_iron 是否逐位等于末 pass EM 解的 mu_r_elem
    em_mur = np.asarray(res3["em_info"]["mu_r_elem"])[iron_idx]
    d = np.abs(em_mur - np.asarray(h["mu_r_iron"]))
    checks["mu_iron_vs_em_info_maxabsdiff"] = float(d.max())
    checks["mu_iron_bitwise_equal_em_info"] = bool(d.max() == 0.0)

    # 3) μr 聚合量（scope §3.1 定义：min / 体积加权均值 / max）可直接算
    mur = np.asarray(h["mu_r_iron"], dtype=float)
    vw_mean = float(np.sum(mur * vols) / np.sum(vols))
    checks["mu_r_agg_min"] = float(mur.min())
    checks["mu_r_agg_volwmean"] = vw_mean
    checks["mu_r_agg_max"] = float(mur.max())
    checks["mu_r_agg_countmean_vs_volwmean"] = [
        float(h["mu_r_iron_mean"]), vw_mean]
    checks["order_ok"] = bool(mur.min() <= vw_mean <= mur.max())

    # 4) frac_above_Tc 口径：现有字段为单元计数分数；体积分数可直接算
    Te = np.asarray(res3["T_elem_hist"][-1])  # 区间末单元温度
    count_frac = float(np.mean(Te > t_curie))
    vol_frac = float(np.sum(vols[Te > t_curie]) / np.sum(vols))
    checks["frac_above_Tc_history_countfrac"] = float(h["frac_above_Tc"])
    checks["frac_count_matches_history"] = bool(count_frac == h["frac_above_Tc"])
    checks["frac_above_Tc_volfrac"] = vol_frac
    checks["Tmax_degC"] = float(np.max(res3["T_hist"]))

    # 5) 不同态取样点的实证：iter=1（μr 取自 T0 场）vs iter=3（μr 取自
    #    末 pass 输入场），两者 μr_agg 不同 → μr 绑定 EM 解输入场
    mur1 = np.asarray(res1["history"][0]["mu_r_iron"], dtype=float)
    checks["iter1_mu_r_volwmean"] = float(np.sum(mur1 * vols)
                                          / np.sum(vols))
    checks["iter3_mu_r_volwmean"] = vw_mean
    checks["mur_differs_between_iter1_and_iter3"] = bool(
        not np.array_equal(mur1, mur))

    # 6) 未动求解路径的旁证：本脚本只读 res，不改任何驱动/求解器代码
    checks["wall_s_iter3"] = wall3
    checks["wall_s_iter1"] = wall1

    payload = {
        "meta": {
            "study": "v2.4 diag phase-1: secant mu_r availability smoke "
                     "(S4-level)",
            "code_base": "main@7f57cc7 (branch v2.4-diag cut from "
                         "origin/main); physics code NOT modified",
            "repo": REPO,
            "geometry": "iron cylinder R=15mm L=60mm, air box half=0.045, "
                        "h=0.016; f=2kHz B_applied H0=1e6 A/m; T0=20C; "
                        "cp(T) law default-off; adiabatic",
            "numpy": np.__version__,
            "ran_at_utc": datetime.now(timezone.utc).isoformat(),
            "unit": "all numbers are simulation outputs (仿真输出); "
                    "wall times are local-machine measurements (本机实测)",
        },
        "checks": checks,
    }
    path = os.path.join(out_dir, "mur_smoke.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("wrote", path)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
