"""BIG-M11: run-all driver for the CloudFEM benchmark suite.

Runs the BIG-M6..M10 + BIG-D2 benchmarks in sequence as subprocesses
(each is a standalone script in this directory with its own asserts),
parses the headline error of each from its stdout, and prints a
summary table with PASS/FAIL per benchmark.  Exit status is non-zero
if any benchmark fails.

BIG-D3 (realworld_sim induction-heating API adapter) is NOT part of
this suite: it depends on the external realworld_sim package, so it
is listed in the summary as skipped.

Run from the repo root:  python3 tests/run_all_benchmarks.py
"""

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def last_float(pattern, text, flags=0):
    m = re.findall(pattern, text, flags)
    return float(m[-1]) if m else None


def all_floats(pattern, text, flags=0):
    return [float(x) for x in re.findall(pattern, text, flags)]


# Each entry: (id, title, script, criterion text, metric extractor)
def metric_m6(out):
    v = last_float(r"err EB = ([+-]?[\d.]+) %", out)
    return f"EB err {v:+.2f}%" if v is not None else "n/a"


def metric_m7(out):
    v = last_float(r"VERDICT: finest-level slab-mean sigma_xx error "
                   r"= ([\d.]+)%", out)
    return f"sigma err {v:.3f}%" if v is not None else "n/a"


def metric_m8(out):
    v = last_float(r"finest-level error = ([+-][\d.]+)%", out)
    return f"P/L err {v:+.3f}%" if v is not None else "n/a"


def metric_m9(out):
    cold = last_float(r"analytic \(Kelvin, cold\) = [\d.]+ W/m  "
                      r"\(([+-][\d.]+)%\)", out)
    bal = None
    for line in out.splitlines():
        if "[main] energy per m:" in line:
            m = re.search(r"residual = \S+ J \(([+-][\d.]+)%\)", line)
            if m:
                bal = float(m.group(1))
    parts = []
    if cold is not None:
        parts.append(f"cold-P err {cold:+.2f}%")
    if bal is not None:
        parts.append(f"balance {bal:+.2f}%")
    return ", ".join(parts) if parts else "n/a"


def metric_m10(out):
    errs = all_floats(r"FEM err ([+-][\d.]+)%", out)
    frz = last_float(r"frozen-profile check: .*?rel diff = ([\d.eE+-]+)",
                     out)
    parts = []
    if errs:
        parts.append("lin err " + "/".join(f"{e:+.2f}%" for e in errs))
    if frz is not None:
        parts.append(f"frozen {frz:.1e}")
    return ", ".join(parts) if parts else "n/a"


def metric_d2(out):
    v = last_float(r"finest error = ([+-][\d.]+)%", out)
    return f"P/L err {v:+.3f}%" if v is not None else "n/a"


BENCHMARKS = [
    ("BIG-M6", "cantilever beam tip deflection (P1 tet, half model)",
     "test_cantilever_benchmark.py", "|err vs EB| < 1% at L4",
     metric_m6),
    ("BIG-M7", "1-D rod thermal stress (heat -> stress chain)",
     "test_bench_thermal_stress.py", "slab sigma err < 2%",
     metric_m7),
    ("BIG-M8", "axisymmetric eddy current, cylinder in axial AC field",
     "test_bench_axi_eddy.py", "P/L err < 5% at finest mesh",
     metric_m8),
    ("BIG-M9", "full induction-heating chain (axi EM -> transient heat)",
     "test_bench_induction_chain.py",
     "energy balance < 1%, cold P err < 2%", metric_m9),
    ("BIG-M10", "nonlinear B-H + Curie point (axi eddy, Picard)",
     "test_bench_nonlinear_bh.py",
     "lin refs < 5%, Picard converges, frozen profile < 0.1%",
     metric_m10),
    ("BIG-D2", "cylinder 50 kHz skin effect vs classical formula",
     "test_bench_skin50k.py", "P/L err < 1%", metric_d2),
]


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    t_all = time.time()
    for bid, title, script, criterion, metric_fn in BENCHMARKS:
        path = os.path.join(HERE, script)
        print("=" * 78)
        print(f"{bid}: {title}")
        print(f"  script: tests/{script}   criterion: {criterion}")
        print("-" * 78)
        t0 = time.time()
        proc = subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        dt = time.time() - t0
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stdout.write(proc.stderr)
        ok = proc.returncode == 0
        metric = metric_fn(proc.stdout)
        results.append((bid, metric, criterion, ok, dt))
        print(f"--> {bid}: {'PASS' if ok else 'FAIL'} "
              f"(rc={proc.returncode}, {dt:.0f} s)")
    print("=" * 78)
    print("BENCHMARK SUMMARY (BIG-M11 run_all)")
    print(f"{'id':<8} {'key error (measured)':<44} {'criterion':<46} "
          f"{'result':<6} {'time':>6}")
    print("-" * 118)
    for bid, metric, criterion, ok, dt in results:
        print(f"{bid:<8} {metric:<44} {criterion:<46} "
              f"{'PASS' if ok else 'FAIL':<6} {dt:>5.0f}s")
    print(f"{'BIG-D3':<8} {'n/a':<44} "
          f"{'external dependency (realworld_sim)':<46} "
          f"{'SKIP':<6} {'-':>6}")
    n_fail = sum(1 for r in results if not r[3])
    print("-" * 118)
    print(f"total: {len(results) - n_fail}/{len(results)} passed, "
          f"1 skipped (BIG-D3), wall time {time.time() - t_all:.0f} s")
    if n_fail:
        print("RUN_ALL: FAIL")
        return 1
    print("RUN_ALL: ALL BENCHMARKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
