"""Validation of the Tet10 Neumann / Robin boundary conditions (v1.6).

transient_heat_tet10 learned the boundary conditions of the P1
solver (cloudfem.fem.tet_heat) in v1.6: Neumann inward fluxes
(float, callable(x, y, z), or callable(x, y, z, t)) and Robin
convection (h, T_inf), assembled on 6-node faces with the 6-point
degree-4 triangle rule. Every check uses an independent reference:

1. Face machinery: on a box mesh each named boundary yields the
   expected number of 6-node faces tiling the unit face exactly, and
   assemble_boundary_terms_tet10 reproduces the closed-form totals
   int q dA, h * T_inf * int dA and h * int dA to round-off (the
   degree-4 rule integrates the face integrands exactly).
2. Regression: the v1.5 transient checks (cube cooling vs the
   product-of-slabs series, adiabatic consistent-mass drift) still
   pass with the new signature, and the default call is bit-wise
   identical to passing neumann=None / robin=None explicitly.
3. Neumann energy balance: a constant inward flux into an otherwise
   insulated cube raises the consistent-mass total heat by exactly
   q * A * t (drift ~1e-15). The same holds for a time-dependent
   flux q(t) = 2t (injected energy t_end^2, integrated with the
   theta weighting) and for a space-varying flux 1 + x.
4. Pure-Neumann conservation: equal flux in at xmin and out at xmax
   conserves the total heat to round-off while a real temperature
   gradient develops (field range ~3 K for the run below).
5. Robin steady limit: a slab with Dirichlet 0 at xmin and
   convection (h = 2, T_inf = 1) at xmax converges to the analytic
   profile T(x) = (2/3) x; the P1 solver on the same problem is run
   for comparison. A cube convecting to T_inf on all faces reaches
   the uniform equilibrium to round-off.
6. Maximum principle (spot check, Robin cooling of a hot cube):
   backward Euler and Crank-Nicolson at dt = 0.02 (above the
   h^2 / (6 alpha) threshold of the v1.0 consistent-mass
   characterisation) keep the field inside [0, 1] exactly; CN at
   dt = 0.005 (below threshold) shows the documented small
   overshoot (~2e-3 on a 1 K range), reported, not hidden.
7. Combination: Dirichlet + Neumann + Robin in one run -- the
   Dirichlet values are exact and the corner-node field agrees with
   the P1 solver on the same problem to discretisation level.
"""

import numpy as np

from cloudfem.fem.tet10 import (
    assemble_boundary_terms_tet10, assemble_mass_tet10,
    transient_heat_tet10, _boundary_faces10)
from cloudfem.fem.tet_heat import transient_heat_tet
from cloudfem.mesh.tet10_mesh import box_mesh_tet10
from cloudfem.mesh.tet_mesh import box_mesh

FACES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


def _total_heat(mesh, TH):
    C = assemble_mass_tet10(mesh, rho_cp=1.0)
    return TH @ (C @ np.ones(mesh.n_nodes))


def test_faces_and_terms():
    m = box_mesh_tet10(4, 4, 4)
    out = {"n_faces": {}, "area_err": 0.0}
    for s in FACES:
        F = _boundary_faces10(m, s)
        pc = m.nodes[F[:, :3]]
        areas = 0.5 * np.linalg.norm(
            np.cross(pc[:, 1] - pc[:, 0], pc[:, 2] - pc[:, 0]),
            axis=1)
        out["n_faces"][s] = len(F)
        out["area_err"] = max(out["area_err"], abs(areas.sum() - 1.0))
    _, F_n = assemble_boundary_terms_tet10(m, neumann={"xmax": 3.0})
    out["neu_total_err"] = abs(F_n.sum() - 3.0)
    K_r, F_r = assemble_boundary_terms_tet10(m, robin={"xmax": (2.0, 1.0)})
    out["robin_F_err"] = abs(F_r.sum() - 2.0)
    k_tot = np.asarray(K_r.sum(axis=1)).ravel().sum()
    out["robin_K_err"] = abs(k_tot - 2.0)
    return out


def test_regression():
    m = box_mesh_tet10(4, 4, 4)
    cen = int(np.argmin(np.sum((m.nodes - 0.5)**2, axis=1)))

    def slab_theta(t):
        n = np.arange(200)
        return float(np.sum(4.0 / (np.pi * (2 * n + 1)) * (-1.0)**n
                            * np.exp(-((2 * n + 1) * np.pi)**2 * t)))

    kw = dict(k=1.0, rho_cp=1.0, bc={s: 0.0 for s in FACES},
              t_end=0.1, dt=0.005, T0=1.0, method="cn")
    _, THa = transient_heat_tet10(m, **kw)
    _, THb = transient_heat_tet10(m, neumann=None, robin=None, **kw)
    out = {"bitwise": bool(np.array_equal(THa, THb)),
           "cool_err": float(THa[-1, cen] - slab_theta(0.1)**3)}
    rng = np.random.default_rng(7)
    T0 = rng.random(m.n_nodes)
    _, TH = transient_heat_tet10(m, k=1.0, rho_cp=1.0, bc=None,
                                 t_end=0.05, dt=0.01, T0=T0, method="cn")
    heat = _total_heat(m, TH)
    out["drift"] = float(np.abs(heat - heat[0]).max() / abs(heat[0]))
    return out


def test_neumann_balance():
    m = box_mesh_tet10(4, 4, 4)
    out = {}
    _, TH = transient_heat_tet10(m, k=1.0, rho_cp=1.0,
                                 neumann={"xmax": 2.5},
                                 t_end=0.4, dt=0.02, T0=0.0, method="cn")
    heat = _total_heat(m, TH)
    out["const"] = float(abs(heat[-1] - 2.5 * 0.4) / (2.5 * 0.4))
    _, TH = transient_heat_tet10(
        m, k=1.0, rho_cp=1.0,
        neumann={"xmax": lambda x, y, z, t: 2.0 * t},
        t_end=0.4, dt=0.02, T0=0.0, method="cn")
    heat = _total_heat(m, TH)
    out["tdep"] = float(abs(heat[-1] - 0.4**2) / 0.4**2)
    _, TH = transient_heat_tet10(
        m, k=1.0, rho_cp=1.0,
        neumann={"zmax": lambda x, y, z: 1.0 + x},
        t_end=0.4, dt=0.02, T0=0.0, method="be")
    heat = _total_heat(m, TH)
    out["xyz"] = float(abs(heat[-1] - 1.5 * 0.4) / (1.5 * 0.4))
    _, TH = transient_heat_tet10(m, k=1.0, rho_cp=1.0,
                                 neumann={"xmin": 3.0, "xmax": -3.0},
                                 t_end=0.5, dt=0.01, T0=0.0, method="cn")
    heat = _total_heat(m, TH)
    out["net0_drift"] = float(np.abs(heat - heat[0]).max())
    out["net0_range"] = float(TH[-1].max() - TH[-1].min())
    return out


def test_robin_steady():
    ms = box_mesh_tet10(8, 2, 2)
    _, TR = transient_heat_tet10(ms, k=1.0, rho_cp=1.0,
                                 bc={"xmin": 0.0},
                                 robin={"xmax": (2.0, 1.0)},
                                 t_end=5.0, dt=0.25, T0=0.0, method="be")
    err10 = float(np.abs(TR[-1] - (2.0 / 3.0) * ms.nodes[:, 0]).max())
    m4 = box_mesh(8, 2, 2)
    _, TR4 = transient_heat_tet(m4, k=1.0, rho_cp=1.0,
                                bc={"xmin": 0.0},
                                robin={"xmax": (2.0, 1.0)},
                                t_end=5.0, dt=0.25, T0=0.0, method="be")
    err4 = float(np.abs(TR4[-1] - (2.0 / 3.0) * m4.nodes[:, 0]).max())
    m = box_mesh_tet10(4, 4, 4)
    _, TU = transient_heat_tet10(m, k=1.0, rho_cp=1.0,
                                 robin={s: (1.5, 1.0) for s in FACES},
                                 t_end=8.0, dt=0.2, T0=0.0, method="be")
    equil = float(np.abs(TU[-1] - 1.0).max())
    return {"err10": err10, "err4": err4, "equil": equil}


def test_max_principle():
    m = box_mesh_tet10(4, 4, 4)
    out = {}
    for tag, method, dt in (("be", "be", 0.02), ("cn", "cn", 0.02),
                            ("cn_fine", "cn", 0.005)):
        _, TC = transient_heat_tet10(
            m, k=1.0, rho_cp=1.0, robin={s: (1.0, 0.0) for s in FACES},
            t_end=0.2, dt=dt, T0=1.0, method=method)
        out[tag] = (float(TC.min()), float(TC.max()))
    return out


def test_combined():
    mc = box_mesh_tet10(6, 3, 3)
    _, TA = transient_heat_tet10(mc, k=1.0, rho_cp=1.0, bc={"xmin": 0.3},
                                 neumann={"xmax": 1.2},
                                 robin={"zmax": (0.8, 0.1)},
                                 t_end=0.3, dt=0.01, T0=0.0, method="cn")
    m4 = box_mesh(6, 3, 3)
    _, TB = transient_heat_tet(m4, k=1.0, rho_cp=1.0, bc={"xmin": 0.3},
                               neumann={"xmax": 1.2},
                               robin={"zmax": (0.8, 0.1)},
                               t_end=0.3, dt=0.01, T0=0.0, method="cn")
    fixed_err = float(np.abs(TA[-1][mc.boundaries["xmin"]] - 0.3).max())
    diff = float(np.abs(TA[-1, :m4.n_nodes] - TB[-1]).max())
    return {"fixed_err": fixed_err, "tet4_diff": diff}


def main():
    ft = test_faces_and_terms()
    print(f"[faces] counts = {ft['n_faces']}, area err = "
          f"{ft['area_err']:.3e}")
    print(f"[terms] neumann total err = {ft['neu_total_err']:.3e}, "
          f"robin F err = {ft['robin_F_err']:.3e}, robin K err = "
          f"{ft['robin_K_err']:.3e}")
    assert all(v == 32 for v in ft["n_faces"].values())
    assert ft["area_err"] < 1e-12
    assert ft["neu_total_err"] < 1e-12 and ft["robin_F_err"] < 1e-12
    assert ft["robin_K_err"] < 1e-12

    rg = test_regression()
    print(f"[regression] default vs explicit-None bitwise = "
          f"{rg['bitwise']}, cube-cooling centre err = "
          f"{rg['cool_err']:+.3e}, adiabatic drift = {rg['drift']:.3e}")
    assert rg["bitwise"]
    assert abs(rg["cool_err"]) < 5e-3
    assert rg["drift"] < 1e-12

    nb = test_neumann_balance()
    print(f"[neumann] energy-balance rel drift: const = "
          f"{nb['const']:.3e}, q(t) = {nb['tdep']:.3e}, q(x) = "
          f"{nb['xyz']:.3e}; net-zero drift = {nb['net0_drift']:.3e} "
          f"with field range = {nb['net0_range']:.3f} K")
    assert nb["const"] < 1e-12 and nb["tdep"] < 1e-12
    assert nb["xyz"] < 1e-12
    assert nb["net0_drift"] < 1e-12 and nb["net0_range"] > 1.0

    rs = test_robin_steady()
    print(f"[robin] slab steady inf err: Tet10 = {rs['err10']:.3e}, "
          f"Tet4 = {rs['err4']:.3e}; uniform equilibrium err = "
          f"{rs['equil']:.3e}")
    assert rs["err10"] < 1e-6 and rs["equil"] < 1e-12

    mp = test_max_principle()
    for tag, (lo, hi) in mp.items():
        print(f"[max principle {tag}] field range = [{lo:+.3e}, "
              f"{hi:.6f}] on [0, 1]")
    assert mp["be"][0] >= -1e-12 and mp["be"][1] <= 1.0 + 1e-12
    assert mp["cn"][0] >= -1e-12 and mp["cn"][1] <= 1.0 + 1e-12
    assert mp["cn_fine"][1] - 1.0 < 5e-3, \
        "CN fine-dt overshoot beyond the documented v1.0 behaviour"

    cb = test_combined()
    print(f"[combined] Dirichlet exact err = {cb['fixed_err']:.3e}, "
          f"corner-field diff vs Tet4 = {cb['tet4_diff']:.3e}")
    assert cb["fixed_err"] == 0.0
    assert cb["tet4_diff"] < 5e-2

    print("ALL TET10 BC VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
