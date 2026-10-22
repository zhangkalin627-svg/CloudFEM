"""Validation of the v2.1 cp(T) Curie-enthalpy-peak thermal law.

All numbers are simulation outputs on the same deliberately small
mesh as the v2.0 test (iron cylinder R = 15 mm, L = 60 mm in an air
box, h = 16 mm); every assertion is a same-mesh comparison, an exact
degeneration limit or a direction statement -- the v2.0 protocol.

cp(T) law (cloudfem/physics/induction3d_tet10.steel_cp_T):
    c(T) = rho_cp * (1 + cp_slope * (T - 20))
           + cp_peak_h * exp(-(T - t_curie)^2/(2*cp_peak_w^2))
Documented analytic model assumptions, NOT measured material data.
Defaults (cp_slope = 0, cp_peak_h = 0) give the constant rho_cp,
i.e. the v2.0 thermal law bitwise.

Tests:
1. Law unit check: peak-off gives exactly rho_cp; Gaussian peak
   shape/centre; dH/dT == c(T) for steel_enthalpy_T.
2. Peak-off runs the v2.0 path: cp_varying is False and the
   discrete energy account closes at solver precision.
3. Peak-on: the per-interval discrete enthalpy identity
   sum_iv ones^T C_iv dT_iv == E_in closes at solver precision;
   the analytic-enthalpy diagnostic is close to E_in.
4. Direction: with the peak enabled the Curie crossing is delayed
   (lower Tmax at the same end time, same drive).
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d_tet10 import (
    induction_heating_3d_tet10, steel_cp_T, steel_enthalpy_T)

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
TC = 770.0
RHO_CP = 3.768e6

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def _run(peak_on):
    key = "on" if peak_on else "off"
    if key not in _CACHE:
        kw = dict(cp_peak_h=RHO_CP, cp_peak_w=25.0) if peak_on else {}
        _CACHE[key] = induction_heating_3d_tet10(
            _mesh(), 2 * PI * GEO["FREQ"],
            B_applied=(0.0, 0.0, MU0 * 1.0e6),
            iron=_in_iron, T0=20.0, t_end=0.2, dt=0.02, em_every=5,
            em_tol=1e-3, em_max_iter=60, em_relax=0.5, **kw)
    return _CACHE[key]


def test_cp_law_unit():
    """steel_cp_T / steel_enthalpy_T unit checks."""
    T = np.array([20.0, 500.0, 770.0, 1200.0])
    # peak off -> exactly the v2.0 constant
    c0 = steel_cp_T(T)
    assert np.all(c0 == RHO_CP), "peak-off cp must be exactly rho_cp"
    # peak shape: maximum at Tc
    c1 = steel_cp_T(T, cp_peak_h=RHO_CP, cp_peak_w=25.0)
    assert c1[2] == RHO_CP * 2.0, "peak height adds exactly at Tc"
    assert c1[2] > c1[1] and c1[2] > c1[3]
    # enthalpy derivative == cp (central differences)
    kw = dict(cp_peak_h=RHO_CP, cp_peak_w=25.0, cp_slope=2e-4)
    for T0 in (100.0, 769.9, 770.0, 1200.0):
        e = 1e-4
        dH = (steel_enthalpy_T(T0 + e, **kw)
              - steel_enthalpy_T(T0 - e, **kw)) / (2 * e)
        c = float(steel_cp_T(T0, **kw))
        assert abs(dH - c) / c < 1e-8, (T0, dH, c)
    assert steel_enthalpy_T(20.0) == 0.0
    print("[cp-law] peak-off exact, peak shape ok, dH/dT == c(T) "
          "(simulation outputs)")


def test_peak_off_v20_path():
    """Peak-off runs the verbatim v2.0 thermal path."""
    res = _run(False)
    assert res["cp_varying"] is False
    print(f"[v20-path] balance_rel = {res['balance_rel']:.3e} "
          f"(simulation outputs)")
    assert abs(res["balance_rel"]) < 1e-9


def test_enthalpy_identity_peak_on():
    """Peak-on: discrete enthalpy identity closes at solver precision."""
    res = _run(True)
    assert res["cp_varying"] is True
    print(f"[enthalpy] E_in = {res['E_in']:.6e} J, discrete H gain = "
          f"{res['enthalpy_gain']:.6e} J, analytic H gain = "
          f"{res['enthalpy_gain_phys']:.6e} J, balance_rel = "
          f"{res['balance_rel']:.3e} (simulation outputs)")
    assert abs(res["balance_rel"]) < 1e-9, \
        "discrete enthalpy identity must close at solver precision"
    rel_phys = abs(res["E_in"] - res["enthalpy_gain_phys"]) / res["E_in"]
    print(f"[enthalpy] analytic-H vs E_in rel = {rel_phys:.3e} "
          f"(simulation outputs)")
    assert rel_phys < 0.05, \
        "analytic enthalpy must track E_in within freezing error"


def test_peak_delays_curie_crossing():
    """Direction: the Curie enthalpy peak delays the temperature rise.

    Thermal-only comparison (no EM solve): uniform T0 = 750 C, fixed
    power, BE stepping with element-level cp(T).  With the peak on,
    the same energy input produces a lower Tmax -- the transition
    enthalpy is absorbed.  (In the coupled driver the peak's dynamic
    effect additionally depends on the EM-interval freezing
    granularity; see docs/v2.1_scope.md.)
    """
    import scipy.sparse.linalg as spla
    from cloudfem.fem.tet10 import (assemble_mass_tet10, assemble_tet10)
    from cloudfem.physics.induction3d_tet10 import (
        _centroid_lookup, _extract_submesh_tet10, _iron_mask)

    mesh = _mesh()
    iron_mask, _ = _iron_mask(mesh, _in_iron)
    sub, _, _ = _extract_submesh_tet10(mesh, iron_mask)
    K, _ = assemble_tet10(sub, k=30.0, f=0.0)
    C1 = assemble_mass_tet10(sub, rho_cp=1.0)
    ones = np.ones(sub.n_nodes)
    n_el = sub.n_elements
    cen = sub.nodes[sub.elements[:, :4]].mean(axis=1)
    vols = sub.element_volumes()
    P_total = 50e3
    F = P_total * (C1 @ ones) / float(ones @ C1 @ ones)

    def run(peak_on):
        T = np.full(sub.n_nodes, 750.0)
        dt, n_steps = 0.005, 40
        for _ in range(n_steps):
            Te = T[sub.elements].mean(axis=1)
            c_e = (steel_cp_T(Te, cp_peak_h=RHO_CP, cp_peak_w=25.0)
                   if peak_on else np.full(n_el, RHO_CP))
            C = assemble_mass_tet10(
                sub, rho_cp=_centroid_lookup(cen, c_e))
            A = C + dt * K
            T = spla.spsolve(A.tocsc(), C @ T + dt * F)
        return float(np.max(T))

    Tmax_off = run(False)
    Tmax_on = run(True)
    print(f"[delay] thermal-only Tmax: peak-off {Tmax_off:.1f} C vs "
          f"peak-on {Tmax_on:.1f} C (simulation outputs)")
    assert Tmax_on < Tmax_off, \
        "enthalpy peak must delay the temperature rise"


if __name__ == "__main__":
    test_cp_law_unit()
    test_peak_off_v20_path()
    test_enthalpy_identity_peak_on()
    test_peak_delays_curie_crossing()
    print("ALL v2.1 cp(T) VALIDATIONS PASSED")
