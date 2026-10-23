"""Validation of the v2.2 Robin cooling boundary on the coupled driver.

All numbers are simulation outputs.  Scope: docs/v2.2_scope.md §4.1
(local light smoke S1-S5 only; A1-A5 full runs go to the butler's
cloud).  Threshold discipline: assertion *forms* are fixed here;
numeric thresholds were set from the measured first-run values with
margin (see V22_IMPL_REPORT.md), and any post-acceptance change
requires re-review.

Cooling parameters used are *model-assumption example values, not
measured data* (scope §7): h = 5000 W/(m^2 K), T_inf = 20 degC.

S1  adiabatic degeneration (bitwise): cooling=None and
    cooling={"surface": (0.0, 20.0)} reproduce the pre-edit
    baseline short run (Tmax, E_in, balance_rel).
S2  Robin assembly gate (thin-wall trap): the driver's K_h/F_h vs
    an independent face-occurrence enumeration + the same
    integrand assembler; the geometry includes a one-cell-thick
    wall, where node-set-derived faces would wrongly collect
    interior faces.
S3  pure-cooling cross path: driver heat stepping with Q == 0 vs
    transient_heat_tet10 with f = 0 and the same Robin spec on
    the same submesh.  The Q == 0 construction is FIXED here
    (butler condition c): J_s is given a zero excitation
    (a callable returning zeros(3)), so the EM solver returns zero
    Joule power; the construction is written into this comment
    and must not be changed afterwards.
S4  three-party balance: |balance_rel| (cooled heated run) and
    |balance_cool| (pure-cooling run) at solver precision;
    adiabatic E_loss is exactly 0.0.
S5  cooling direction: Tmax(cooled) < Tmax(adiabatic) strictly,
    E_loss > 0, surface-mean(cooled) < surface-mean(adiabatic)
    (the cooling effect direction; see the S5 comment for why the
    literal "surface cooler than core" does not apply under
    strong skin-effect heating).
"""

import numpy as np

from cloudfem.fem.tet10 import (assemble_boundary_terms_tet10,
                                _boundary_faces10)
from cloudfem.fem.tet10 import _FACES10
from cloudfem.fem.tet10 import transient_heat_tet10
from cloudfem.mesh.tet10_mesh import Tet10Mesh, box_mesh_tet10, tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.induction3d_tet10 import (
    _cooling_boundary_terms, _extract_submesh_tet10,
    _register_surface_boundary_tet10, induction_heating_3d_tet10)

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016, FREQ=2000.0)
H_COOL = 5000.0   # model-assumption example value, NOT measured data
T_INF = 20.0

_CACHE = {}

# Thresholds (fixed from measured first-run values with margin;
# see V22_IMPL_REPORT.md).  Do not change without re-review
# (butler condition b: any post-acceptance-run change needs re-review).
_THR = {
    "s1_rel": 1e-12,     # bitwise degeneration
    "s2_rel": 1e-12,     # assembly gate
    "s3_rel": 1e-9,      # cross-path node temperatures
    "s4_rel": 1e-9,      # three-party balance
    "a2_rel": 1e-9,       # measured 9.973e-16; margin ~6 orders
    "a3_pq": 1e-2,        # measured 2.5e-3 at 60 s plateau; 4x margin
    "a3_dTdt": 1.0,        # measured 0.28 K/s; ~3.5x margin [K/s]
    "a4_rel": 1e-9,       # measured 0.0 (enthalpy form closes); margin ~9 orders
}


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
    return np.array([_in_iron(x, y, z) for x, y, z in cen])


def _thin_wall_mesh():
    """One-cell-thick wall (the thin-wall trap geometry) via the
    codebase's own box_mesh_tet10 (Kuhn cells + canonical Tet10
    enrichment, so the local node order matches _FACES10)."""
    return box_mesh_tet10(2, 2, 1, xlim=(0.0, 2.0), ylim=(0.0, 2.0),
                          zlim=(0.0, 1.0))


def _bipyramid_mesh():
    """Minimal trap geometry: two tets sharing one face, apices on
    opposite sides.  The shared (interior, occurrence-2) face has
    all six nodes on the exterior surface, so the banned
    node-set-derived-face inference would wrongly collect it.
    Built with the canonical EDGES mid ordering."""
    EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    corners = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                        (0., 0., 1.), (0., 0., -1.)])
    tets = [(0, 1, 2, 3), (0, 1, 2, 4)]
    mids = {}
    rows = []
    for tet in tets:
        row = list(tet)
        for p, q in EDGES:
            key = (min(tet[p], tet[q]), max(tet[p], tet[q]))
            if key not in mids:
                mids[key] = len(corners) + len(mids)
            row.append(mids[key])
        rows.append(row)
    nodes = np.zeros((len(corners) + len(mids), 3))
    nodes[:len(corners)] = corners
    for (p, q), m in mids.items():
        nodes[m] = 0.5 * (corners[p] + corners[q])
    return Tet10Mesh(nodes, np.array(rows, dtype=int),
                     n_corners=len(corners))


def _independent_exterior_faces(elements):
    """Own face-occurrence enumeration: (face_list, node_set).

    face_list: 6-node faces (corners + mids) occurring exactly
    once; node_set: union of their nodes.  Does NOT use the
    driver's registration.
    """
    counts, faces = {}, {}
    for el in elements:
        el = [int(v) for v in el]
        for corners, mids in _FACES10:
            tri = tuple(el[c] for c in corners)
            key = tuple(sorted(tri))
            counts[key] = counts.get(key, 0) + 1
            if key not in faces:
                faces[key] = tuple(el[c] for c in corners + mids)
    out, nodes = [], set()
    for key, cnt in counts.items():
        if cnt == 1:
            out.append(faces[key])
            nodes.update(faces[key])
    return out, nodes


def _driver_run(cooling, T0=20.0, J_s=None, t_end=0.1, dt=0.02):
    mesh = _mesh()
    omega = 2 * PI * GEO["FREQ"]
    kw = dict(mesh=mesh, omega=omega, iron=_in_iron, T0=T0,
              t_end=t_end, dt=dt, em_every=5, em_tol=1e-3,
              em_max_iter=60, em_relax=0.5, cooling=cooling)
    if J_s is not None:
        kw["J_s"] = J_s
    else:
        kw["B_applied"] = (0.0, 0.0, MU0 * 1.0e6)
    return induction_heating_3d_tet10(**kw)


# --- S2: Robin assembly gate -----------------------------------------
def test_s2_robin_assembly_gate():
    """Driver K_h/F_h vs independent occurrence enumeration.

    Runs on (a) the real iron submesh and (b) a one-cell-thick
    wall where node-set-derived faces would wrongly collect
    interior faces.  The independent path enumerates exterior
    faces itself and assembles with the same integrand; the
    driver path uses _cooling_boundary_terms (occurrence-based
    registration + the v1.6 assembler).
    """
    h, Tinf = H_COOL, T_INF

    def check(sub, tag, trap_check=False):
        K1, F1 = _cooling_boundary_terms(sub, h, Tinf)
        faces_ind, _ = _independent_exterior_faces(sub.elements)
        # driver-registered face set must equal the independent one
        faces_drv = [tuple(int(v) for v in f)
                     for f in _boundary_faces10(sub, "surface")]
        set_ind = {tuple(sorted(f[:3])) for f in faces_ind}
        set_drv = {tuple(sorted(f[:3])) for f in faces_drv}
        assert set_ind == set_drv, \
            f"[{tag}] driver face set != independent face set"
        # safety property: no occurrence-2 (interior) face leaked in
        counts = {}
        for el in sub.elements:
            el = [int(v) for v in el]
            for corners, _ in _FACES10:
                key = tuple(sorted(el[c] for c in corners))
                counts[key] = counts.get(key, 0) + 1
        assert all(counts[tuple(sorted(f[:3]))] == 1 for f in faces_drv), \
            f"[{tag}] an interior face leaked into the surface set"
        if trap_check:
            # negative control: the banned node-set-derived inference
            # (faces whose six nodes all lie in the surface node set,
            # over ALL faces incl. occurrence-2) MUST differ here --
            # it wrongly collects the shared interior face of the
            # bipyramid.
            surf_nodes = set(int(v) for v in sub.boundaries["surface"])
            all_faces = set()
            for el in sub.elements:
                el = [int(v) for v in el]
                for corners, mids in _FACES10:
                    all_faces.add(tuple(el[c] for c in corners + mids))
            banned_all = {f for f in all_faces
                          if all(int(v) in surf_nodes for v in f)}
            assert len(banned_all) == len(faces_drv) + 1, \
                f"[{tag}] trap geometry must discriminate " \
                f"(banned={len(banned_all)} vs driver={len(faces_drv)})"
        # independent assembly on the same face set
        nodes = set()
        for f in faces_ind:
            nodes.update(f)
        sub2 = Tet10Mesh(sub.nodes, sub.elements,
                         boundaries={"surface": np.array(sorted(nodes))},
                         n_corners=sub.n_corners)
        K2, F2 = assemble_boundary_terms_tet10(
            sub2, robin={"surface": (h, Tinf)})
        dK = abs(K1 - K2)
        dF = float(np.max(np.abs(F1 - F2))
                   / max(float(np.max(np.abs(F2))), 1e-300))
        dK_rel = float(dK.max()
                       / max(float(abs(K2).max()), 1e-300))
        print(f"[s2 {tag}] faces={len(faces_ind)} dK_rel={dK_rel:.3e} "
              f"dF_rel={dF:.3e} (simulation outputs)")
        assert dK_rel <= _THR["s2_rel"], "K_h gate failed"
        assert dF <= _THR["s2_rel"], "F_h gate failed"

    # (a) the real iron submesh
    mesh = _mesh()
    sub, _, _ = _extract_submesh_tet10(mesh, _iron_mask(mesh))
    check(sub, "iron-submesh")
    # (b) one-cell-thick wall
    wall = _thin_wall_mesh()
    check(wall, "thin-wall")
    # (c) bipyramid: the shared interior face's six nodes are all
    # on the exterior surface -- the banned inference would bite
    check(_bipyramid_mesh(), "bipyramid", trap_check=True)


# --- S1: adiabatic degeneration (bitwise) -----------------------------
def test_s1_adiabatic_degeneration():
    """cooling=None and h=0 reproduce the pre-edit baseline.

    Baseline values were recorded from the unmodified driver
    (main@7f57cc7) on the same short case before the v2.2 edit;
    they are embedded here.  All numbers are simulation outputs.
    """
    base = {"Tmax": _BASE["Tmax"], "E_in": _BASE["E_in"],
            "balance_rel": _BASE["balance_rel"]}
    r_none = _driver_run(None)
    r_h0 = _driver_run({"surface": (0.0, 20.0)})
    for tag, r in (("cooling=None", r_none), ("h=0", r_h0)):
        Tmax = float(np.max(r["T_hist"][-1]))
        for key, got in (("Tmax", Tmax), ("E_in", r["E_in"]),
                         ("balance_rel", r["balance_rel"])):
            ref = base[key]
            rel = abs(got - ref) / max(abs(ref), 1e-300)
            print(f"[s1 {tag}] {key}: got {got:.6e} ref {ref:.6e} "
                  f"rel {rel:.3e} (simulation outputs)")
            assert rel <= _THR["s1_rel"], f"S1 failed: {tag} {key}"
        assert r["E_loss"] == 0.0, "adiabatic E_loss must be exactly 0.0"
        assert all(hh["E_loss_iv"] == 0.0 for hh in r["history"])


# --- S3/S4-pure: pure-cooling cross path + balance_cool ---------------
def test_s3_pure_cooling_cross_path():
    """Driver (Q==0) vs transient_heat_tet10 pure cooling.

    Q == 0 construction (FIXED, butler condition c): J_s is given a
    zero excitation (a callable returning zeros(3)), so the EM
    solver returns zero Joule power; the heat stepping then sees
    F_Q == 0 exactly.  Do not change this construction afterwards.
    """
    T0, h, Tinf = 100.0, H_COOL, T_INF
    # driver side
    res = _driver_run({"surface": (h, Tinf)}, T0=T0,
                      J_s=lambda x, y, z: np.zeros(3),
                      t_end=0.1, dt=0.02)
    assert res["E_in"] == 0.0, "zero excitation must give E_in == 0"
    assert res["balance_cool"] is not None, \
        "pure-cooling metric must be defined (D > 0)"
    print(f"[s3] balance_cool={res['balance_cool']:.3e} "
          f"(simulation output)")
    assert abs(res["balance_cool"]) <= _THR["s4_rel"], \
        "pure-cooling balance must close"
    # independent path: transient_heat_tet10 on the same submesh
    mesh = _mesh()
    sub, _, _ = _extract_submesh_tet10(mesh, _iron_mask(mesh))
    _register_surface_boundary_tet10(sub)
    times, Th = transient_heat_tet10(
        sub, k=30.0, rho_cp=3.768e6, f=0.0,
        robin={"surface": (h, Tinf)}, t_end=0.1, dt=0.02, T0=T0,
        method="be")
    T_drv = res["T_hist"][-1]
    T_ind = Th[-1]
    rel = float(np.max(np.abs(T_drv - T_ind)) /
                max(float(np.max(np.abs(T_ind))), 1e-300))
    print(f"[s3] cross-path max rel diff = {rel:.3e} "
          f"(simulation outputs)")
    assert rel <= _THR["s3_rel"], "S3 cross-path failed"


# --- S4/S5: heated cooled run -----------------------------------------
def test_s4_s5_heated_cooled():
    """Three-party balance + cooling direction on a heated run."""
    r_ad = _driver_run(None)
    r_cool = _driver_run({"surface": (H_COOL, T_INF)})
    for tag, r in (("adiabatic", r_ad), ("cooled", r_cool)):
        print(f"[s4 {tag}] balance_rel={r['balance_rel']:.3e} "
              f"E_loss={r['E_loss']:.6e} (simulation outputs)")
        assert abs(r["balance_rel"]) <= _THR["s4_rel"], \
            "three-party balance must close"
    assert r_ad["E_loss"] == 0.0
    assert r_cool["E_loss"] > 0.0, "cooled run must lose energy"
    Tmax_ad = float(np.max(r_ad["T_hist"][-1]))
    Tmax_cool = float(np.max(r_cool["T_hist"][-1]))
    print(f"[s5] Tmax adiabatic={Tmax_ad:.4f} cooled={Tmax_cool:.4f} "
          f"(simulation outputs, degC)")
    assert Tmax_cool < Tmax_ad, "cooling must lower Tmax strictly"
    # surface-vs-adiabatic direction: the cooling must lower the
    # surface temperature relative to the no-cooling run.  (Scope
    # S5's "surface cooler than core" read literally does not hold
    # under strong skin-effect heating -- the surface is the
    # hottest zone with or without cooling; the direction of the
    # cooling effect is the cross-run surface comparison.  See
    # V22_IMPL_REPORT.md §5.)
    sub = r_cool["submesh"]
    surf_nodes = sorted(int(v) for v in sub.boundaries["surface"])
    s_ad = float(r_ad["T_hist"][-1][surf_nodes].mean())
    s_cool = float(r_cool["T_hist"][-1][surf_nodes].mean())
    print(f"[s5] surface-mean adiabatic={s_ad:.4f} cooled={s_cool:.4f} "
          f"(simulation outputs, degC)")
    assert s_cool < s_ad, "cooling must lower the surface temperature"


# Pre-edit baseline (main@7f57cc7, unmodified driver) on the S1 short
# case (t_end=0.1, dt=0.02, em_every=5, B_applied H0=1e6 A/m,
# f=2 kHz, T0=20).  All numbers are simulation outputs.
_BASE = {
    "Tmax": 666.6673583840344,
    "E_in": 46898.18990038223,
    "balance_rel": 0.0,
}


# --- A1/A2: full heated run with cooling (cloud acceptance) -----------
def _a1_runs():
    """0.4 s v2.0-parameter coupled runs, adiabatic vs cooled.

    Cached: the adiabatic reference is the v2.0 coupled run
    (~12 min locally); the cooled run adds the model-assumption
    example cooling {"surface": (5000, 20)}.
    """
    if "a1" not in _CACHE:
        mesh = _mesh()
        omega = 2 * PI * GEO["FREQ"]
        kw = dict(mesh=mesh, omega=omega,
                  B_applied=(0.0, 0.0, MU0 * 1.0e6), iron=_in_iron,
                  T0=20.0, t_end=0.4, dt=0.02, em_every=5,
                  em_tol=1e-3, em_max_iter=60, em_relax=0.5)
        r_ad = induction_heating_3d_tet10(**kw)
        r_cool = induction_heating_3d_tet10(
            **dict(kw, cooling={"surface": (H_COOL, T_INF)}))
        _CACHE["a1"] = (r_ad, r_cool)
    return _CACHE["a1"]


def test_a1_cooling_effect_direction():
    """A1: cooled run strictly below the adiabatic对照.

    Direction assertions only (no unmeasured magnitudes):
    Tmax(cooled) < Tmax(adiabatic), E_loss > 0, and the
    surface-vs-core temperature relation recorded.
    """
    r_ad, r_cool = _a1_runs()
    Tmax_ad = float(np.max(r_ad["T_hist"][-1]))
    Tmax_cool = float(np.max(r_cool["T_hist"][-1]))
    print(f"[a1] Tmax adiabatic={Tmax_ad:.2f} cooled={Tmax_cool:.2f} "
          f"E_loss={r_cool['E_loss']:.6e} (simulation outputs, degC/J)")
    assert Tmax_cool < Tmax_ad, "A1: cooling must lower Tmax strictly"
    assert r_cool["E_loss"] > 0.0, "A1: cooled run must lose energy"
    sub = r_cool["submesh"]
    Tn_cool = r_cool["T_hist"][-1]
    Tn_ad = r_ad["T_hist"][-1]
    surf = np.array(sorted(int(v) for v in sub.boundaries["surface"]))
    d = float(Tn_cool[surf].mean() - Tn_ad[surf].mean())
    print(f"[a1] surface-mean cooled minus adiabatic = {d:.2f} degC "
          f"(simulation output, recorded)")
    assert d < 0.0, "A1: the cooled surface band must be cooler than " \
        "the adiabatic one"


def test_a2_three_party_balance():
    """A2: three-party energy account on the A1 cooled run."""
    _, r_cool = _a1_runs()
    print(f"[a2] balance_rel={r_cool['balance_rel']:.3e} "
          f"(simulation output)")
    if _THR["a2_rel"] is None:
        print("[a2] threshold not set: measurement only")
        return
    assert abs(r_cool["balance_rel"]) <= _THR["a2_rel"], \
        "A2: three-party balance must close"


# --- A3: steady-state balance (independent physical check) -------------
def test_a3_steady_state_balance():
    """A3: weak drive + cooling to a temperature plateau.

    Independent physical check (not an algebraic identity): at the
    plateau the interval power P must match the interval-end
    loss rate q_loss, and Tmax must stop moving.  Sizing: thermal
    time constant tau = rho_c*V/(h*A) ~ 4.5 s for the iron
    cylinder at h = 5000 (model-assumption example value), but
    Bi ~ 1 (internal gradients) and the T-dependent P slow the
    approach; t_end = 60 s with dt = 0.1 s and em_every = 30
    (20 EM intervals) reaches pq = 2.5e-3 and dTmax/dt = 0.28 K/s
    locally.  The cloud acceptance run uses the same parameters.
    """
    mesh = _mesh()
    omega = 2 * PI * GEO["FREQ"]
    res = induction_heating_3d_tet10(
        mesh, omega, B_applied=(0.0, 0.0, MU0 * 1.2e5), iron=_in_iron,
        T0=20.0, t_end=60.0, dt=0.1, em_every=30,
        em_tol=1e-3, em_max_iter=60, em_relax=0.5,
        cooling={"surface": (H_COOL, T_INF)})
    hist = res["history"]
    assert len(hist) == 20, "60 s / 3 s intervals = 20 records"
    last = hist[-1]
    pq = abs(last["P"] - last["loss_rate_end"]) / max(last["P"], 1e-300)
    Tmax = [float(np.max(T)) for T in res["T_hist"]]
    dTdt = abs(Tmax[-1] - Tmax[-2]) / (hist[-1]["t_end"]
                                       - hist[-2]["t_end"])
    print(f"[a3] P={last['P']:.6e} q_loss_end={last['loss_rate_end']:.6e} "
          f"rel={pq:.3e} dTmax/dt={dTdt:.3e} K/s (simulation outputs)")
    if _THR["a3_pq"] is None or _THR["a3_dTdt"] is None:
        print("[a3] thresholds not set: measurement only")
        return
    assert pq <= _THR["a3_pq"], "A3: P must balance q_loss at plateau"
    assert dTdt <= _THR["a3_dTdt"], "A3: Tmax must stop moving"


# --- A4: cp(T) x cooling joint (R3 gate) -------------------------------
def _a4_runs():
    """Cooling through the Curie window, peak on vs off.

    T0 = 900 degC > Tc, pure cooling (J_s zero excitation, fixed
    Q==0 construction as in S3), so the temperature falls through
    the Curie window in the cooling-dominated (slow) phase and the
    enthalpy peak is genuinely sampled.  em_every = 1: the
    capacity matrix is rebuilt every heat step, so the peak is
    not skipped by the v2.1 interval-freezing granularity (with
    em_every = 5 the peak effect is underestimated and the delay
    falls below the interval resolution).
    """
    if "a4" not in _CACHE:
        mesh = _mesh()
        omega = 2 * PI * GEO["FREQ"]
        kw = dict(mesh=mesh, omega=omega,
                  J_s=lambda x, y, z: np.zeros(3), iron=_in_iron,
                  T0=900.0, t_end=0.6, dt=0.02, em_every=1,
                  em_tol=1e-3, em_max_iter=60, em_relax=0.5,
                  cooling={"surface": (H_COOL, T_INF)})
        r_on = induction_heating_3d_tet10(
            **dict(kw, cp_peak_h=3.768e6, cp_peak_w=25.0))
        r_off = induction_heating_3d_tet10(**kw)
        _CACHE["a4"] = (r_on, r_off)
    return _CACHE["a4"]


def _cross_time(res, Tc=770.0):
    """First interval-end time the surface-band mean element
    temperature falls below Tc.

    The core cools by conduction and does not cross within a
    short window; the surface band (the zone that actually
    traverses the Curie window under cooling) is the consistent
    crossing metric for the peak-delay direction.
    """
    mesh = _mesh()
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    iron = _iron_mask(mesh)
    cen_i = cen[iron]
    r_c = np.hypot(cen_i[:, 0], cen_i[:, 1])
    surf = (r_c > 0.75 * GEO["R"]) & (np.abs(cen_i[:, 2]) < 0.01)
    for t, Te in zip(res["times"], res["T_elem_hist"]):
        if float(np.mean(Te[surf])) < Tc:
            return float(t)
    return None


def test_a4_cp_cooling_joint():
    """A4: enthalpy-form balance + peak-delay direction + R3 gate.

    Direction only for the crossing times (no absolute times):
    t_cross(peak on) > t_cross(peak off), consistent with the v2.1
    pure-thermal delay conclusion.  R3 gate: at least one interval
    of the peak-on run must have peak_engaged True; if the case
    design cannot achieve it, the report must declare the
    interaction check vacuous and redesign the case (never pass
    silently).
    """
    r_on, r_off = _a4_runs()
    for tag, r in (("peak-on", r_on), ("peak-off", r_off)):
        print(f"[a4 {tag}] balance_rel={r['balance_rel']:.3e} "
              f"(simulation output)")
        if _THR["a4_rel"] is not None:
            assert abs(r["balance_rel"]) <= _THR["a4_rel"], \
                "A4: enthalpy-form balance must close"
        else:
            print(f"[a4 {tag}] threshold not set: measurement only")
    engaged = [h["peak_engaged"] for h in r_on["history"]]
    print(f"[a4] peak_engaged per interval (peak-on): {engaged}")
    assert any(engaged), \
        "A4 R3 gate: the peak must be genuinely sampled in at least " \
        "one interval (redesign the case instead of passing silently)"
    t_on, t_off = _cross_time(r_on), _cross_time(r_off)
    print(f"[a4] t_cross peak-on={t_on} peak-off={t_off} s "
          f"(simulation outputs)")
    assert t_on is not None and t_off is not None, \
        "A4: both runs must cross Tc within the window"
    assert t_on > t_off, \
        "A4: the enthalpy peak must delay the Curie crossing"
