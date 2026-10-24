"""v2.4 diagnostic switch validation (D2 structure + D1 reproducibility).

The v2.4 driver adds a default-off diagnostic switch ``diag`` to
:func:`cloudfem.physics.induction3d_tet10.induction_heating_3d_tet10`:
when on, each per-interval history record additionally carries
``stagger_dT_seq`` (per-pass dT in pass order, first pass None =
explicit missing marker), ``frac_above_Tc_vol`` (volume fraction of
iron elements above t_curie at the interval-end temperature field)
and ``mu_r_agg`` (min / volume-weighted mean / max of the secant
mu_r of that interval's last EM pass).  Diagnostic recording is
read-only observation of solver arrays; it must not alter any
computed number (frozen criterion CLOUD_CRITERIA.md D1).

Tests (S4-level case: t_end=0.1, dt=0.02, em_every=5 -> 1 EM
interval; iron cylinder R=15mm/L=60mm, air box half=0.045,
h=0.016, f=2kHz, B_applied drive H0=1e6 A/m, T0=20C, cp(T) law
default-off, adiabatic -- the same deliberately small mesh as the
v1.9/v2.0 tests):
1. diag off (default): no diagnostic keys in history.
2. D2 structure (diag on, iters 1 and 3): seq length =
   stagger_passes; first entry None (never 0); scalar stagger_dT =
   last seq entry; frac_above_Tc_vol in [0, 1]; mu_r_agg ordered
   (min <= volume-weighted mean <= max); legacy count-fraction
   key frac_above_Tc untouched and present.
3. Early break (stagger_tol huge): seq length = actual
   stagger_passes = 2, first entry None.
4. D1 reproducibility (diag on vs off, same case): T_hist, Tmax,
   E_in, balance_rel and all pre-existing history keys bitwise
   identical; formal threshold per the frozen criterion is
   relative diff <= 1e-12 (ints/bools exact).

All numbers are simulation outputs.
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d_tet10 import induction_heating_3d_tet10

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
CASE = dict(t_end=0.1, dt=0.02, em_every=5, T0=20.0)
THRESH = 1e-12

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def _run(stagger_iter, diag, stagger_tol=None):
    key = (stagger_iter, diag, stagger_tol)
    if key not in _CACHE:
        omega = 2 * PI * GEO["FREQ"]
        _CACHE[key] = induction_heating_3d_tet10(
            _mesh(), omega, B_applied=(0.0, 0.0, MU0 * 1.0e6),
            iron=_in_iron, T0=CASE["T0"], t_end=CASE["t_end"],
            dt=CASE["dt"], em_every=CASE["em_every"],
            stagger_iter=stagger_iter, stagger_tol=stagger_tol,
            diag=diag)
    return _CACHE[key]


def _check_d2(rec, label):
    seq = rec["stagger_dT_seq"]
    assert isinstance(seq, list), f"{label}: seq must be a list"
    assert len(seq) == rec["stagger_passes"], \
        f"{label}: len(seq)={len(seq)} != passes={rec['stagger_passes']}"
    assert len(seq) > 0 and seq[0] is None, \
        f"{label}: first pass must be an explicit missing marker (None)"
    assert all(x is None or isinstance(x, float) for x in seq), \
        f"{label}: seq entries must be None or float"
    if rec["stagger_passes"] == 1:
        assert rec["stagger_dT"] is None and seq == [None], \
            f"{label}: single-pass interval"
    else:
        assert seq[-1] == rec["stagger_dT"], \
            f"{label}: scalar stagger_dT must equal last seq entry"
    fv = rec["frac_above_Tc_vol"]
    assert isinstance(fv, float) and 0.0 <= fv <= 1.0, \
        f"{label}: frac_above_Tc_vol={fv} out of [0,1]"
    agg = rec["mu_r_agg"]
    assert (isinstance(agg, tuple) and len(agg) == 3
            and agg[0] <= agg[1] <= agg[2]), \
        f"{label}: mu_r_agg={agg} must be ordered (min, vw-mean, max)"
    # legacy count-fraction key untouched and present
    assert "frac_above_Tc" in rec
    assert "mu_r_iron_mean" in rec and "mu_r_iron_min" in rec
    print(f"[{label}] seq={seq} frac_vol={fv:.6f} "
          f"mu_r_agg=({agg[0]:.3f},{agg[1]:.3f},{agg[2]:.3f}) "
          f"(simulation outputs)")


def test_diag_off_absent():
    """diag off (default): history has no diagnostic keys."""
    res = _run(3, False)
    for rec in res["history"]:
        for k in ("stagger_dT_seq", "frac_above_Tc_vol", "mu_r_agg"):
            assert k not in rec, f"diag off must not record {k}"
    print("[diag-off] no diagnostic keys present (1 interval)")


def test_diag_d2_structure():
    """D2: diagnostic record structure on iters 1 and 3."""
    for it in (1, 3):
        res = _run(it, True)
        assert len(res["history"]) == 1
        _check_d2(res["history"][0], f"iter={it}")


def test_diag_early_break_structure():
    """D2: early break -- seq length tracks actual passes."""
    res = _run(5, True, stagger_tol=1e6)
    rec = res["history"][0]
    assert rec["stagger_passes"] == 2, \
        f"expected early break at 2 passes, got {rec['stagger_passes']}"
    _check_d2(rec, "early-break")


def _rel(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / abs(b if b != 0 else 1.0)


def test_diag_d1_reproducibility():
    """D1: diag on vs off -- no computed number may change."""
    on = _run(3, True)
    off = _run(3, False)
    max_rd = 0.0
    for key in ("E_in", "balance_rel"):
        rd = _rel(on[key], off[key])
        max_rd = max(max_rd, rd)
    rd = _rel(float(np.max(on["T_hist"])), float(np.max(off["T_hist"])))
    max_rd = max(max_rd, rd)
    assert np.array_equal(np.asarray(on["T_hist"]),
                          np.asarray(off["T_hist"])), "T_hist must match"
    ho, hf = on["history"][0], off["history"][0]
    assert set(ho) - set(hf) == {"stagger_dT_seq", "frac_above_Tc_vol",
                                 "mu_r_agg"}, \
        "only diagnostic keys may be added"
    for k in ("P", "T_elem_mean", "T_elem_max", "frac_above_Tc",
              "mu_r_iron_mean", "mu_r_iron_min", "t_end"):
        max_rd = max(max_rd, _rel(ho[k], hf[k]))
    for k in ("em_iterations", "em_converged", "stagger_passes",
              "stagger_dT"):
        assert ho[k] == hf[k], f"history[{k}] must be exactly equal"
    for k in ("mu_r_iron", "Q_iron"):
        assert np.array_equal(np.asarray(ho[k]), np.asarray(hf[k])), \
            f"history[{k}] must be bitwise equal"
    print(f"[D1] diag on vs off: max float rel diff = {max_rd:.3e} "
          f"(threshold {THRESH:.0e}; simulation outputs)")
    assert max_rd <= THRESH


if __name__ == "__main__":
    test_diag_off_absent()
    test_diag_d2_structure()
    test_diag_early_break_structure()
    test_diag_d1_reproducibility()
    print("ALL V2.4 DIAG VALIDATIONS PASSED")
