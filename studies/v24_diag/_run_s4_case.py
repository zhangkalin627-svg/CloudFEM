#!/usr/bin/env python3
"""v2.4 diag 阶段二 —— S4 级单算例运行器（供 D1/D2 验证）。

用法：CFEM_REPO=<仓库检出路径> python3 _run_s4_case.py <stagger_iter> <diag 0/1> <stagger_tol|none> <out_json>

S4 级轻算例（同 v2.3 轻算例口径与 v2.4 第一阶段冒烟）：
t_end=0.1, dt=0.02, em_every=5 → 1 个 EM 区间；
铁柱 R=15mm/L=60mm，气盒 half=0.045，h=0.016，f=2kHz，
B_applied 驱动 H0=1e6 A/m，T0=20°C，cp(T) 律默认关闭，绝热驱动。

输出 JSON：Tmax / E_in / balance_rel / T_hist / history（数组转 list，
float64 经 repr 精确往返；缺失 dT 记 null）。
全部数为仿真输出。
"""

import json
import os
import sys

CFEM_REPO = os.environ.get("CFEM_REPO")
if not CFEM_REPO:
    raise SystemExit("CFEM_REPO env var required")
sys.path.insert(0, CFEM_REPO)

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d_tet10 import induction_heating_3d_tet10
import inspect as _inspect

_HAS_DIAG = "diag" in _inspect.signature(induction_heating_3d_tet10).parameters

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
CASE = dict(t_end=0.1, dt=0.02, em_every=5, T0=20.0)


def _mesh():
    m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                              h=GEO["H"])
    return tet4_to_tet10(m4)


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def _conv(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def main():
    stagger_iter = int(sys.argv[1])
    diag = bool(int(sys.argv[2]))
    tol_arg = sys.argv[3]
    stagger_tol = None if tol_arg == "none" else float(tol_arg)
    out_path = sys.argv[4]

    omega = 2 * PI * GEO["FREQ"]
    kw = dict(mesh=_mesh(), omega=omega, iron=_in_iron, T0=CASE["T0"],
              t_end=CASE["t_end"], dt=CASE["dt"], em_every=CASE["em_every"],
              stagger_iter=stagger_iter, stagger_tol=stagger_tol,
              B_applied=(0.0, 0.0, MU0 * 1.0e6))
    if _HAS_DIAG:
        kw["diag"] = diag
    elif diag:
        raise SystemExit("driver has no diag parameter but diag=1 "
                         "requested")
    res = induction_heating_3d_tet10(**kw)

    hist = []
    for rec in res["history"]:
        conv = {}
        for k, v in rec.items():
            if isinstance(v, (list, tuple)):
                conv[k] = [_conv(x) if x is not None else None for x in v]
            else:
                conv[k] = _conv(v)
        hist.append(conv)

    payload = {
        "meta": {
            "repo": CFEM_REPO,
            "stagger_iter": stagger_iter,
            "stagger_tol": stagger_tol,
            "diag": diag,
            "driver_has_diag": _HAS_DIAG,
            "unit": "simulation outputs (仿真输出)",
        },
        "Tmax": float(np.max(res["T_hist"])),
        "E_in": float(res["E_in"]),
        "balance_rel": float(res["balance_rel"]),
        "T_hist": np.asarray(res["T_hist"]).tolist(),
        "history": hist,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {out_path} Tmax={payload['Tmax']:.6f} "
          f"E_in={payload['E_in']:.6f}")


if __name__ == "__main__":
    main()
