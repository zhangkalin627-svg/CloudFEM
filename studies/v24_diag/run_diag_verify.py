#!/usr/bin/env python3
"""v2.4 diag 阶段二 —— D1/D2 验证（本地冒烟，S4 级算例）。

D1（硬判据）：(a) v2.4-diag 诊断关 vs main@7f57cc7 基线；
             (b) v2.4-diag 诊断开 vs 诊断关。
  Tmax / E_in / history 既有浮点键相对差 <= 1e-12；
  整数/布尔键精确相等；balance_rel 维持 <= 4.4e-16 量级。
D2（结构）：stagger_dT_seq 长度 = stagger_passes；首 pass 缺失显式
  标记（None）；frac_above_Tc_vol ∈ [0,1]；mu_r_agg 三元组有序；
  标量 stagger_dT = 序列末值（与现状语义一致）。

main 基线跑在临时 worktree（main@7f57cc7，跑完删除）。
全部数为仿真输出；耗时为本机实测。
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORKTREE = os.path.expanduser("~/workspace/cloudfem_v24diag")
STUDIES = os.path.join(WORKTREE, "studies", "v24_diag")
RUNNER = os.path.join(STUDIES, "_run_s4_case.py")
TMP_MAIN = os.path.expanduser("~/workspace/cloudfem_main_tmp")
OUT_JSON = os.path.join(STUDIES, "diag_verify.json")

THRESH = 1e-12
BAL_THRESH = 4.4e-16

BASE_KEYS_FLOAT = ["P", "T_elem_mean", "T_elem_max", "frac_above_Tc",
                   "mu_r_iron_mean", "mu_r_iron_min", "t_end"]
BASE_KEYS_INTBOOL = ["em_iterations", "em_converged", "stagger_passes"]
BASE_KEYS_ARRAY = ["mu_r_iron", "Q_iron"]


def run_case(repo, stagger_iter, diag, stagger_tol, tag):
    out = os.path.join("/tmp", f"v24_{tag}.json")
    tol_arg = "none" if stagger_tol is None else str(stagger_tol)
    env = dict(os.environ, CFEM_REPO=repo)
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, RUNNER, str(stagger_iter), str(int(diag)),
         tol_arg, out],
        env=env, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"case {tag} failed rc={r.returncode}\n"
                           f"STDOUT:\n{r.stdout[-2000:]}\n"
                           f"STDERR:\n{r.stderr[-2000:]}")
    with open(out) as f:
        payload = json.load(f)
    payload["wall_s"] = wall
    print(f"  [{tag}] Tmax={payload['Tmax']:.4f} E_in={payload['E_in']:.6f} "
          f"wall={wall:.1f}s", flush=True)
    return payload


def rel_diff(a, b):
    if a == b:
        return 0.0
    d = abs(b if b != 0 else 1.0)
    return abs(a - b) / d


def compare_base(a, b, label):
    """比较两份 baseline history（诊断键除外），返回最大相对差与明细。"""
    res = {"label": label, "ok": True, "max_rel_diff": 0.0, "details": {}}
    for key, va, vb in [("Tmax", a["Tmax"], b["Tmax"]),
                        ("E_in", a["E_in"], b["E_in"]),
                        ("balance_rel", a["balance_rel"], b["balance_rel"])]:
        rd = rel_diff(va, vb)
        res["details"][key] = {"rel_diff": rd, "a": va, "b": vb}
        res["max_rel_diff"] = max(res["max_rel_diff"], rd)
    ha, hb = a["history"][0], b["history"][0]
    for k in BASE_KEYS_FLOAT:
        rd = rel_diff(ha[k], hb[k])
        res["details"][f"history.{k}"] = {"rel_diff": rd}
        res["max_rel_diff"] = max(res["max_rel_diff"], rd)
    for k in BASE_KEYS_INTBOOL:
        eq = ha[k] == hb[k]
        res["details"][f"history.{k}"] = {"equal": eq}
        if not eq:
            res["ok"] = False
    for k in BASE_KEYS_ARRAY:
        eq = ha[k] == hb[k]
        res["details"][f"history.{k}"] = {"bitwise_equal": eq}
        if not eq:
            res["ok"] = False
    eq = a["T_hist"] == b["T_hist"]
    res["details"]["T_hist"] = {"bitwise_equal": eq}
    if not eq:
        res["ok"] = False
    bal_ok = (abs(a["balance_rel"]) <= BAL_THRESH
              and abs(b["balance_rel"]) <= BAL_THRESH)
    res["details"]["balance_rel_magnitude_ok"] = bal_ok
    if res["max_rel_diff"] > THRESH:
        res["ok"] = False
    return res


def check_d2(payload, label):
    """D2 结构性检查（诊断开运行）。"""
    h = payload["history"][0]
    checks = {}
    seq = h.get("stagger_dT_seq")
    checks["has_keys"] = all(k in h for k in
                             ("stagger_dT_seq", "frac_above_Tc_vol",
                              "mu_r_agg"))
    checks["seq_len_eq_passes"] = (seq is not None
                                   and len(seq) == h["stagger_passes"])
    checks["seq_first_missing"] = (seq is not None and len(seq) > 0
                                   and seq[0] is None)
    checks["seq_last_eq_scalar"] = (
        seq is not None and len(seq) > 0
        and ((h["stagger_passes"] == 1 and h["stagger_dT"] is None
              and seq == [None])
             or (h["stagger_passes"] > 1
                 and seq[-1] == h["stagger_dT"])))
    checks["seq_rest_floats"] = (
        seq is not None and all(x is None or isinstance(x, float)
                                for x in seq))
    fv = h.get("frac_above_Tc_vol")
    checks["frac_vol_in_01"] = (isinstance(fv, float) and 0.0 <= fv <= 1.0)
    agg = h.get("mu_r_agg")
    checks["mu_r_agg_ordered"] = (
        agg is not None and len(agg) == 3
        and agg[0] <= agg[1] <= agg[2])
    # 现有计数口径字段未被改动：仍为计数分数语义（与体积分数可能不同）
    checks["legacy_count_frac_key_intact"] = "frac_above_Tc" in h
    ok = all(checks.values())
    return {"label": label, "ok": ok, "checks": checks,
            "seq": seq, "frac_above_Tc_vol": fv, "mu_r_agg": agg,
            "stagger_passes": h["stagger_passes"],
            "stagger_dT": h["stagger_dT"]}


def main():
    print("== v2.4 diag D1/D2 verification (S4-level, local) ==", flush=True)
    t_start = time.perf_counter()

    # 临时 main 基线 worktree
    subprocess.run(["git", "-C", WORKTREE, "worktree", "add", "--detach",
                    TMP_MAIN, "7f57cc7"], check=True,
                   capture_output=True, text=True)

    results = {"d1a": [], "d1b": [], "d2": []}
    try:
        # 14 个单算例，2 并发（VM 2 核、内存紧，不再加并发）
        jobs = []
        for it in (1, 2, 3, 4, 5):
            jobs.append(("v24", it, False, None, f"v24_off_iter{it}"))
            jobs.append(("main", it, False, None, f"main_off_iter{it}"))
            jobs.append(("v24", it, True, None, f"v24_on_iter{it}"))
        jobs.append(("v24", 5, True, 1e6, "v24_on_earlybreak"))
        repo_of = {"v24": WORKTREE, "main": TMP_MAIN}
        payloads = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(run_case, repo_of[kind], it, dg, tol, tag): tag
                    for (kind, it, dg, tol, tag) in jobs}
            for fut in futs:
                tag = futs[fut]
                payloads[tag] = fut.result()
        cases_v24_off = {it: payloads[f"v24_off_iter{it}"]
                         for it in (1, 2, 3, 4, 5)}
        cases_main = {it: payloads[f"main_off_iter{it}"]
                      for it in (1, 2, 3, 4, 5)}
        cases_v24_on = {it: payloads[f"v24_on_iter{it}"]
                        for it in (1, 2, 3, 4, 5)}
        eb = payloads["v24_on_earlybreak"]

        print("-- D1a: v2.4 diag-off vs main@7f57cc7 baseline --", flush=True)
        for it in (1, 2, 3, 4, 5):
            c = compare_base(cases_v24_off[it], cases_main[it],
                             f"iter={it} diag-off vs main")
            results["d1a"].append(c)
            print(f"  iter={it}: ok={c['ok']} "
                  f"max_rel_diff={c['max_rel_diff']:.3e}", flush=True)

        print("-- D1b: v2.4 diag-on vs diag-off --", flush=True)
        for it in (1, 2, 3, 4, 5):
            c = compare_base(cases_v24_on[it], cases_v24_off[it],
                             f"iter={it} diag-on vs diag-off")
            results["d1b"].append(c)
            print(f"  iter={it}: ok={c['ok']} "
                  f"max_rel_diff={c['max_rel_diff']:.3e}", flush=True)

        print("-- D2: diagnostic record structure (diag-on) --",
              flush=True)
        for it in (1, 2, 3, 4, 5):
            c = check_d2(cases_v24_on[it], f"iter={it}")
            results["d2"].append(c)
            print(f"  iter={it}: ok={c['ok']} seq={c['seq']} "
                  f"frac_vol={c['frac_above_Tc_vol']:.6f} "
                  f"mu_r_agg={[f'{x:.3f}' for x in c['mu_r_agg']]}",
                  flush=True)
        c = check_d2(eb, "early-break(iter=5,tol=1e6)")
        results["d2"].append(c)
        print(f"  early-break: ok={c['ok']} seq={c['seq']} "
              f"passes={c['stagger_passes']}", flush=True)
    finally:
        subprocess.run(["git", "-C", WORKTREE, "worktree", "remove",
                        "--force", TMP_MAIN], capture_output=True)

    total = time.perf_counter() - t_start
    all_ok = (all(c["ok"] for c in results["d1a"])
              and all(c["ok"] for c in results["d1b"])
              and all(c["ok"] for c in results["d2"]))
    payload = {
        "meta": {
            "study": "v2.4 diag phase-2 D1/D2 verification (S4-level, "
                     "local smoke)",
            "threshold": THRESH,
            "balance_rel_threshold": BAL_THRESH,
            "unit": "all numbers are simulation outputs (仿真输出); "
                    "wall times are local-machine measurements (本机实测)",
            "total_wall_s": total,
        },
        "results": results,
        "all_ok": all_ok,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT_JSON}; all_ok={all_ok}; "
          f"total_wall={total:.1f}s", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
