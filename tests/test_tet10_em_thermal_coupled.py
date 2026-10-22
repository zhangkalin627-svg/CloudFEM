"""Validation of the v2.0 thermally coupled nonlinear B-H solver.

All numbers are simulation outputs on the same deliberately small
mesh as the v1.9 test (iron cylinder R = 15 mm, L = 60 mm in an air
box, h = 16 mm; the skin is NOT resolved, h/delta >> 1, and every
assertion is a same-mesh comparison, an exact degeneration limit or
a direction statement -- the v1.9 protocol).

Material laws: the BIG-M4/M10 analytic model laws (module docstring
of cloudfem/physics/induction3d_tet10.py) -- M(H) = Ms*tanh(H/H_REF)
with the tanh Curie switch step(T) at Tc = 770 degC (width 15 K)
and the tanh sigma(T) law (6e6 -> 1e6 S/m, centre 550 degC, width
150 K).  Documented *model assumptions*, not measured material
data.  Thermal properties are constant (k = 30 W/(m K),
rho_cp = 3.768e6 J/(m^3 K)); M10's cp(T) with its Curie enthalpy
peak is NOT carried over, so crossing times are not quantitatively
comparable to BIG-M10 -- only the mechanism is asserted.

Tests:
1. Material-law unit check: curie_step_T / bh_secant_mu_r_HT /
   steel_sigma_T against an independent transcription of the M10
   formulas (step_T / mu_r_of_HT / sigma_T of
   tests/test_bench_nonlinear_bh.py), plus exact float64 anchors
   (step(20 degC) = 1, step(770 degC) = 0.5, step(1200 degC) = 0).
2. Isothermal limit: T_elem == 20 degC reproduces the v1.9 solver
   with sigma == sigma(20 degC) held constant (the BIG-M10 case-A
   isothermal convention) -- a-field and power relative differences
   at solver precision.
3. Coupled run (B_applied drive H0 = 1e6 A/m, f = 2 kHz,
   t_end = 0.4 s, heat dt = 0.02 s, EM every 5 heat steps,
   em_tol = 1e-3): the hottest element crosses Tc by t = 0.2 s;
   when the surface-layer mean temperature first crosses Tc the
   core band is still > 100 K colder; the surface-layer mean
   secant mu_r collapses between the first and last EM intervals;
   the radial centroid of the Joule power moves inward during
   the crossing window; the total power falls below half its
   initial value; the discrete energy account (accumulated EM
   energy vs enthalpy gain) closes at solver precision; every EM
   interval's Picard iteration converges.  Thresholds were fixed
   from the measured prototype values with margin, as documented
   in ~/workspace/cloudfem_v20_report.md.  NOTE: the coupled EM
   tolerance must stay at 1e-3 -- at 5e-3 the cold first interval
   "converges" at iteration 2, before the saturation front forms
   (rel_dB rises before it falls), overestimating P by 13%.
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.em3d_tet10 import solve_eddy_current_nonlinear_tet10
from cloudfem.physics.induction3d_tet10 import (
    bh_secant_mu_r_HT, curie_step_T, induction_heating_3d_tet10,
    solve_eddy_current_nonlinear_tet10_T, steel_sigma_T)

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
TC = 770.0

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def _iron_mask(mesh):
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    return np.array([_in_iron(x, y, z) for x, y, z in cen]), cen


# --- independent transcription of the BIG-M10 material laws --------
# (from tests/test_bench_nonlinear_bh.py, written out here so the
# check does not call the module's own formulas)
_M10_MS = 1.9 / MU0
_M10_H_REF = _M10_MS / 200.0


def _m10_step_T(T):
    return 0.5 * (1.0 - np.tanh((np.asarray(T, float) - 770.0) / 15.0))


def _m10_mu_mag_of_H(Hmag):
    h = np.asarray(Hmag, dtype=float) / _M10_H_REF
    ratio = np.where(h > 1.0e-6, np.tanh(h) / np.maximum(h, 1.0e-300),
                     1.0 - h * h / 3.0)
    return 200.0 * ratio


def _m10_mu_r_of_HT(Hmag, T):
    return 1.0 + _m10_mu_mag_of_H(Hmag) * _m10_step_T(T)


def _m10_sigma_T(T):
    s = 0.5 * (1.0 - np.tanh((np.asarray(T, float) - 550.0) / 150.0))
    return 1.0e6 + (6.0e6 - 1.0e6) * s


def test_material_laws_vs_m10():
    """Module laws vs the independent M10 transcription + anchors."""
    H = np.logspace(0, 7, 25)
    T = np.array([0.0, 20.0, 200.0, 550.0, 700.0, 755.0, 770.0,
                  785.0, 900.0, 1200.0])
    HH, TT = np.meshgrid(H, T)
    d_step = float(np.max(np.abs(curie_step_T(TT) - _m10_step_T(TT))))
    d_mu = float(np.max(np.abs(bh_secant_mu_r_HT(HH, TT)
                               - _m10_mu_r_of_HT(HH, TT))))
    d_sig = float(np.max(np.abs(steel_sigma_T(TT) - _m10_sigma_T(TT)))
                  / 6.0e6)
    print(f"[laws] max diff vs M10 transcription: step {d_step:.3e}, "
          f"mu_r {d_mu:.3e}, sigma (rel) {d_sig:.3e} "
          f"(simulation outputs)")
    assert d_step < 1e-12 and d_mu < 1e-9 and d_sig < 1e-12
    # Exact float64 anchors of the tanh switch.
    assert float(curie_step_T(20.0)) == 1.0
    assert float(curie_step_T(770.0)) == 0.5
    assert float(curie_step_T(1200.0)) == 0.0
    # Far above Tc the magnetic part is gone at any field.
    mu_hot = bh_secant_mu_r_HT(H, 1200.0)
    assert np.all(mu_hot == 1.0), "mu_r must be exactly 1 far above Tc"
    # At Tc exactly half the magnetic part survives.
    mu_tc = bh_secant_mu_r_HT(H, 770.0)
    assert np.allclose(mu_tc, 1.0 + 0.5 * _m10_mu_mag_of_H(H),
                       rtol=0.0, atol=1e-9)
    # sigma anchors (M10 case-A convention value at 20 degC).
    assert abs(float(steel_sigma_T(20.0)) - 5995738.27581585) < 1e-3
    assert abs(float(steel_sigma_T(2000.0)) - 1.0e6) < 1.0


def test_isothermal_limit():
    """T == 20 degC: the T-solver reproduces the v1.9 solver."""
    mesh = _mesh()
    omega = 2 * PI * GEO["FREQ"]
    H0 = 2.0e4
    sig20 = float(steel_sigma_T(20.0))

    def sigf(x, y, z):
        return sig20 if _in_iron(x, y, z) else 0.0

    a19, _, info19 = solve_eddy_current_nonlinear_tet10(
        mesh, sigma=sigf, omega=omega,
        B_applied=(0.0, 0.0, MU0 * H0), iron=_in_iron,
        tol=1e-3, max_iter=60, relax=0.5)
    aT, _, infoT = solve_eddy_current_nonlinear_tet10_T(
        mesh, np.full(mesh.n_elements, 20.0), omega,
        B_applied=(0.0, 0.0, MU0 * H0), iron=_in_iron,
        tol=1e-3, max_iter=60, relax=0.5)
    d_a = float(np.linalg.norm(aT - a19) / np.linalg.norm(a19))
    d_P = abs(float(infoT["P"][-1]) - float(info19["P"][-1])) \
        / float(info19["P"][-1])
    print(f"[iso] rel diff a = {d_a:.3e}, P = {d_P:.3e}; iters "
          f"v1.9 {info19['iterations']} vs T {infoT['iterations']} "
          f"(simulation outputs)")
    assert infoT["converged"] and info19["converged"]
    assert d_a < 1e-9, "isothermal T-solve must reproduce v1.9"
    assert d_P < 1e-9


def _coupled_run():
    if "coupled" not in _CACHE:
        mesh = _mesh()
        omega = 2 * PI * GEO["FREQ"]
        _CACHE["coupled"] = induction_heating_3d_tet10(
            mesh, omega, B_applied=(0.0, 0.0, MU0 * 1.0e6),
            iron=_in_iron, T0=20.0, t_end=0.4, dt=0.02, em_every=5,
            em_tol=1e-3, em_max_iter=60, em_relax=0.5)
    return _CACHE["coupled"]


def _bands(mesh):
    iron_mask, cen = _iron_mask(mesh)
    cen_i = cen[iron_mask]
    r_c = np.hypot(cen_i[:, 0], cen_i[:, 1])
    surf = (r_c > 0.75 * GEO["R"]) & (np.abs(cen_i[:, 2]) < 0.01)
    deep = (r_c < 0.4 * GEO["R"]) & (np.abs(cen_i[:, 2]) < 0.01)
    return iron_mask, surf, deep


def test_coupled_curie_crossing():
    """Coupled run: crossing order, mu collapse, redistribution."""
    mesh = _mesh()
    res = _coupled_run()
    iron_mask, surf, deep = _bands(mesh)
    hist = res["history"]
    assert len(hist) == 4, "0.4 s at 0.1 s EM intervals = 4 records"
    assert all(h["em_converged"] for h in hist), \
        "every EM interval's Picard must converge"
    vols_i = mesh.element_volumes()[iron_mask]
    _, cen = _iron_mask(mesh)
    r_c = np.hypot(cen[iron_mask][:, 0], cen[iron_mask][:, 1])

    # (a) crossing order: hottest element over Tc by t = 0.2 s;
    # when the surface-band mean first crosses Tc, the core band
    # is still > 100 K colder.
    Tmax = [h["T_elem_max"] for h in hist]
    Te_hist = res["T_elem_hist"]
    surf_mean = Te_hist[:, surf].mean(axis=1)
    deep_mean = Te_hist[:, deep].mean(axis=1)
    cross = [i for i in range(1, len(surf_mean)) if surf_mean[i] > TC]
    assert Tmax[1] > TC, "hottest element must cross Tc by 0.2 s"
    assert cross, "surface band must cross Tc within 0.4 s"
    k = cross[0]
    gap = float(surf_mean[k] - deep_mean[k])
    print(f"[coupled] Tmax at 0.2 s = {Tmax[1]:.0f} C; surface band "
          f"crosses Tc at t = {res['times'][k]:.1f} s "
          f"({surf_mean[k]:.0f} C) with core band {gap:.0f} K colder "
          f"(simulation outputs)")
    assert gap > 100.0, "surface must cross well before the core"

    # (b) surface mu_r collapse between first and last EM solves.
    mu_first = float(hist[0]["mu_r_iron"][surf].mean())
    mu_last = float(hist[-1]["mu_r_iron"][surf].mean())
    print(f"[coupled] surface mu_r: first interval {mu_first:.2f} "
          f"-> last {mu_last:.2f} (simulation outputs)")
    assert mu_last < 3.0, "surface mu_r must collapse to ~1"
    assert mu_last < 0.3 * mu_first, "collapse must be dramatic"

    # (c) radial centroid of Joule power moves inward in the
    # crossing window (it partly recovers once the whole workpiece
    # is above Tc, so the window minimum is asserted).
    rP = []
    for h in hist:
        Q = h["Q_iron"]
        rP.append(float(np.sum(Q * vols_i * r_c)
                        / np.sum(Q * vols_i)))
    print(f"[coupled] power centroid radius [mm]: "
          f"{[f'{1e3 * r:.2f}' for r in rP]} (simulation outputs)")
    assert min(rP) < 0.97 * rP[0], \
        "power deposition must move inward during the crossing"

    # (d) total power falls below half its initial value.
    ratio = float(hist[-1]["P"] / hist[0]["P"])
    print(f"[coupled] total power {hist[0]['P']:.4e} -> "
          f"{hist[-1]['P']:.4e} W (ratio {ratio:.3f}); EM iters "
          f"{[h['em_iterations'] for h in hist]} (simulation outputs)")
    assert ratio < 0.5

    # (e) discrete energy account closes at solver precision.
    print(f"[coupled] E_in = {res['E_in']:.6e} J, enthalpy gain = "
          f"{res['enthalpy_gain']:.6e} J, balance rel = "
          f"{res['balance_rel']:.3e} (simulation outputs)")
    assert abs(res["balance_rel"]) < 1e-9

    # By the end almost the whole workpiece is above Tc (the
    # reason the case is truncated at 0.4 s -- see scope doc).
    assert hist[-1]["frac_above_Tc"] > 0.8


if __name__ == "__main__":
    test_material_laws_vs_m10()
    test_isothermal_limit()
    test_coupled_curie_crossing()
    print("ALL THERMALLY COUPLED BH VALIDATIONS PASSED")
