#!/usr/bin/env python3
"""Run the CloudFEM test suite in light or full mode.

Usage:
    python3 tests/run_all_tests.py --light [--list] [--timeout SECONDS]
    python3 tests/run_all_tests.py --full  [--list] [--timeout SECONDS]

``--full`` is the default.  The runner dynamically discovers every
``tests/test_*.py`` file; it does not rely on a checked-in list of test
filenames.  Each selected test is run as a separate Python subprocess
with the repository root as its working directory and with
``PYTHONPATH=.`` (any caller-provided PYTHONPATH entries are preserved
after that leading dot).  A failing or timed-out
test does not stop later tests.  The process exit status is the number
of failed tests (launch errors and timeouts count as failures).

CLOUDFEM_SMOKE convention
--------------------------
Light mode exports ``CLOUDFEM_SMOKE=1`` to every selected child
process.  Tests assigned to the reduced tier may use that variable to
choose a coarser grid, fewer steps, or a smaller subset; native-light
tests may simply ignore it.  This runner only establishes the entry
point side of the convention: whether an individual existing test
responds to the variable is outside the scope of this runner, and no
existing test file is modified by it.  Full mode explicitly removes
``CLOUDFEM_SMOKE`` from child environments so an inherited value cannot
silently turn a purported full run into a reduced run.

Tier assignments follow section 5 of
``~/workspace/cloudfem_perf/TEST_MATRIX.md``.  A dynamically discovered
file with no entry in the classification table defaults to the full
tier and is labelled "未分类（全量）" in the summary.

``run_all_benchmarks.py`` is intentionally not scheduled as an extra
item: it is not a ``test_*.py`` file, and the benchmark programs it
drives are themselves discovered here as individual ``test_*.py``
files, so scheduling it as well would duplicate them.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DEFAULT_TIMEOUT_SECONDS = 3600.0


class Tier(str, Enum):
    LIGHT = "light"
    REDUCED = "reduced"
    FULL = "full"


@dataclass(frozen=True)
class Classification:
    tier: Tier
    note: str
    classified: bool = True

    @property
    def label(self) -> str:
        if not self.classified:
            return "未分类（全量）"
        if self.tier is Tier.LIGHT:
            return "轻量（原样）"
        if self.tier is Tier.REDUCED:
            return "轻量缩减（全量原样）"
        return "全量"


# Tier table from TEST_MATRIX.md section 5.  Discovery remains dynamic;
# this table only assigns a tier to a filename.  In particular, the
# v2.0 coupled test is classified here even when running from a baseline
# that predates that file, so the same runner remains correct if the
# file is later added to tests/.
CLASSIFICATIONS: dict[str, Classification] = {
    # Native-light tier in TEST_MATRIX.md section 5.
    "test_validation.py": Classification(Tier.LIGHT, "2D 小网格，原样轻量"),
    "test_axi_validation.py": Classification(Tier.LIGHT, "轴对称小网格，原样轻量"),
    "test_unstructured.py": Classification(Tier.LIGHT, "三档网格仍便宜，原样轻量"),
    "test_elastic.py": Classification(Tier.LIGHT, "2D 制造解，原样轻量"),
    "test_nonlinear_em.py": Classification(Tier.LIGHT, "2D 小网格，原样轻量"),
    "test_webui.py": Classification(Tier.LIGHT, "接口/HTTP 冒烟，原样轻量"),
    "test_tet.py": Classification(Tier.LIGHT, "小盒网格，原样轻量"),
    "test_tet10_bc.py": Classification(Tier.LIGHT, "闭式积分/短瞬态，原样轻量"),
    # Reduced in light mode, original/full in full mode.
    "test_elastic3d.py": Classification(Tier.REDUCED, "轻量保留仿射、面片与最粗档"),
    "test_modal3d.py": Classification(Tier.REDUCED, "轻量保留质量矩阵与低阶模态哨兵"),
    "test_transient3d.py": Classification(Tier.REDUCED, "轻量保留级数对照与一档时间阶"),
    "test_unstructured3d.py": Classification(Tier.REDUCED, "轻量保留中档与网格质量哨兵"),
    "test_dynamics3d.py": Classification(Tier.REDUCED, "轻量保留 SDOF、稳定性与 API 哨兵"),
    "test_bench_axi_eddy.py": Classification(Tier.REDUCED, "轻量保留最粗与最细两档"),
    "test_bench_skin50k.py": Classification(Tier.REDUCED, "轻量保留两档并建议关闭绘图"),
    "test_tet10.py": Classification(Tier.REDUCED, "轻量保留面片/积分锚点子集"),
    # Full-only tier in TEST_MATRIX.md section 5.
    "test_bench_thermal_stress.py": Classification(Tier.FULL, "四档收敛是判据本体"),
    "test_em3d.py": Classification(Tier.FULL, "边元 3D 双频/多档重解"),
    "test_induction3d.py": Classification(Tier.FULL, "多工况 3D 耦合链长跑"),
    "test_tet10_em.py": Classification(Tier.FULL, "MMS、球与 D1 多档重解"),
    "test_tet10_em_transient.py": Classification(Tier.FULL, "时间步减半与多工况为判据本体"),
    "test_inductance.py": Classification(Tier.FULL, "依赖完整 D1 Tet10 相量解"),
    "test_tet10_em_nonlinear.py": Classification(Tier.FULL, "非线性 Picard 长跑"),
    "test_tet10_em_thermal_coupled.py": Classification(Tier.FULL, "v2.0 热-电磁耦合全程，最重档"),
    "test_tet10_cp_enthalpy.py": Classification(Tier.FULL, "v2.1 cp(T) 焓峰，同 v2.0 小网格耦合解"),
    "test_cantilever_benchmark.py": Classification(Tier.FULL, "L4 大自由度直接解"),
    "test_bench_induction_chain.py": Classification(Tier.FULL, "60 s 瞬态且每步重解 EM"),
    "test_bench_nonlinear_bh.py": Classification(Tier.FULL, "瞬态 Picard 长跑"),
}

UNCLASSIFIED = Classification(
    Tier.FULL,
    "未在 TEST_MATRIX 分档表中归类，默认全量",
    classified=False,
)


@dataclass
class Result:
    name: str
    tier_label: str
    return_code: int
    elapsed_seconds: float
    status: str


def positive_timeout(value: str) -> float:
    """Parse a timeout; zero is the documented no-timeout value."""
    timeout = float(value)
    if timeout < 0:
        raise argparse.ArgumentTypeError("timeout must be zero or positive")
    return timeout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamically discovered CloudFEM tests by tier."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--light",
        dest="mode",
        action="store_const",
        const="light",
        help="run native-light tests and reduced-mode tests (default: full)",
    )
    mode.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="run every discovered test in its original/full form",
    )
    parser.set_defaults(mode="full")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the tests selected by the chosen mode without running them",
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "per-test timeout in seconds (default: %(default)s; "
            "0 disables the timeout)"
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TEST",
        help=(
            "restrict discovery to this test filename or stem; may be "
            "given more than once"
        ),
    )
    return parser.parse_args(argv)


def discover_tests() -> list[Path]:
    """Return all tests/test_*.py files in deterministic filename order."""
    return sorted(
        (path for path in TESTS_DIR.glob("test_*.py") if path.is_file()),
        key=lambda path: path.name,
    )


def classification_for(path: Path) -> Classification:
    return CLASSIFICATIONS.get(path.name, UNCLASSIFIED)


def normalize_only(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        name = Path(value).name
        if not name.endswith(".py"):
            name += ".py"
        names.add(name)
    return names


def selected_tests(
    discovered: list[Path], mode: str, only: set[str]
) -> tuple[list[tuple[Path, Classification]], list[str]]:
    selected: list[tuple[Path, Classification]] = []
    requested_but_unselected: list[str] = []

    for path in discovered:
        classification = classification_for(path)
        if only and path.name not in only:
            continue
        if mode == "light" and classification.tier is Tier.FULL:
            if only:
                requested_but_unselected.append(path.name)
            continue
        selected.append((path, classification))

    if only:
        discovered_names = {path.name for path in discovered}
        missing = sorted(only - discovered_names)
        requested_but_unselected.extend(missing)

    return selected, sorted(set(requested_but_unselected))


def smoke_convention_text(mode: str) -> str:
    if mode == "light":
        return (
            "CLOUDFEM_SMOKE=1 is exported to every selected child process; "
            "reduced-tier tests may honor it, native-light tests may ignore it."
        )
    return (
        "CLOUDFEM_SMOKE is removed from every child process in full mode, "
        "so an inherited value cannot silently reduce a full run."
    )


def child_environment(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [
        entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    # Use the literal "." required by the established invocation
    # convention; the child working directory is REPO_ROOT.  Preserve
    # any caller entries after it, without duplicating ".".
    pythonpath_entries = [entry for entry in pythonpath_entries if entry != "."]
    env["PYTHONPATH"] = os.pathsep.join([".", *pythonpath_entries])
    if mode == "light":
        env["CLOUDFEM_SMOKE"] = "1"
    else:
        env.pop("CLOUDFEM_SMOKE", None)
    return env


def print_selection_list(
    selected: list[tuple[Path, Classification]], mode: str
) -> None:
    print("\nSelected tests (dry run; nothing executed):")
    print(f"{'Test':40} {'Tier':24} {'Light':8} {'Full':8} Note")
    print(f"{'-' * 40} {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 24}")
    for path, classification in selected:
        light_action = (
            "run" if classification.tier in (Tier.LIGHT, Tier.REDUCED) else "-"
        )
        print(
            f"{path.name:40} {classification.label:24} "
            f"{light_action:8} {'run':8} {classification.note}"
        )
    print(f"\nConvention: {smoke_convention_text(mode)}")


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a timed-out test and, on POSIX, its process group."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        return

    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        return
    process.wait()


def run_test(
    path: Path,
    classification: Classification,
    mode: str,
    timeout: float,
) -> Result:
    print(f"\n>>> {path.name} [{classification.label}]", flush=True)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            env=child_environment(mode),
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        elapsed = time.monotonic() - started
        print(f"<<< {path.name}: failed to launch: {exc}", flush=True)
        return Result(path.name, classification.label, 127, elapsed, "ERROR")

    try:
        return_code = process.wait(timeout=timeout if timeout > 0 else None)
        status = "PASS" if return_code == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        return_code = 124
        status = "TIMEOUT"
    except KeyboardInterrupt:
        terminate_process_tree(process)
        raise

    elapsed = time.monotonic() - started
    print(
        f"<<< {path.name}: rc={return_code}, elapsed={elapsed:.2f}s, "
        f"status={status}",
        flush=True,
    )
    return Result(path.name, classification.label, return_code, elapsed, status)


def print_summary(results: list[Result], mode: str) -> int:
    failed = sum(result.status != "PASS" for result in results)
    passed = len(results) - failed

    print("\nSummary")
    print(f"Convention: {smoke_convention_text(mode)}")
    print(f"{'Test':40} {'rc':>6} {'elapsed (s)':>12} {'Tier':24} Status")
    print(f"{'-' * 40} {'-' * 6} {'-' * 12} {'-' * 24} {'-' * 8}")
    for result in results:
        print(
            f"{result.name:40} {result.return_code:>6} "
            f"{result.elapsed_seconds:>12.2f} {result.tier_label:24} "
            f"{result.status}"
        )
    print(
        f"\nSelected: {len(results)}; passed: {passed}; failed: {failed}; "
        f"exit code: {failed}"
    )
    return failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    discovered = discover_tests()
    if not discovered:
        print(f"No test_*.py files found in {TESTS_DIR}", file=sys.stderr)
        return 2

    only = normalize_only(args.only)
    selected, unselected = selected_tests(discovered, args.mode, only)
    if unselected:
        print(
            "Requested tests were not selected "
            f"({'mode=' + args.mode}): {', '.join(unselected)}",
            file=sys.stderr,
        )
        return 2
    if not selected:
        print("No tests selected.", file=sys.stderr)
        return 2

    timeout_text = (
        "disabled" if args.timeout == 0 else f"{args.timeout:g} seconds per test"
    )
    print("CloudFEM test runner")
    print(f"Repository: {REPO_ROOT}")
    print(f"Mode: {args.mode}")
    print(f"Timeout: {timeout_text}")
    print(f"Discovered: {len(discovered)}; selected: {len(selected)}")
    print(f"Convention: {smoke_convention_text(args.mode)}")

    if args.list:
        print_selection_list(selected, args.mode)
        return 0

    results: list[Result] = []
    try:
        for path, classification in selected:
            results.append(run_test(path, classification, args.mode, args.timeout))
    except KeyboardInterrupt:
        print("\nInterrupted; summarizing completed tests.", file=sys.stderr)
        if results:
            print_summary(results, args.mode)
        return 130

    return print_summary(results, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
