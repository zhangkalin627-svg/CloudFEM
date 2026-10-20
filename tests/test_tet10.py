"""Validation of the Tet10 (second-order tetrahedral) elements (v1.5).

Contents -- every check compares against an independent reference
(closed-form integrals, polynomial patch fields, a manufactured
solution, beam theory, or a series solution), and Tet4 numbers are
produced by the existing P1 solvers on the same problems, so the
P1/P2 comparison is apples-to-apples:

1. Shape functions and mass matrix: partition of unity, Kronecker
   property at the 10 nodes, and the exact closed-form integrals
   (row sums int N_i dV = -V/20 at corners, +V/5 at midpoints;
   total = V; matrix symmetric positive definite).
2. Mesh conversion: corner indices and boundary sets are preserved
   by tet4_to_tet10; element volumes still tile the box.
3. Heat patch tests: a linear field is reproduced exactly by both
   Tet4 and Tet10; a quadratic field (with the constant source
   -lap(u)) is reproduced exactly by Tet10 on an unstructured sphere
   mesh (error ~1e-15) but not by Tet4 (error ~2e-2).
4. Heat convergence (manufactured solution sin(pi x) sin(pi y)
   sin(pi z), unit cube, zero Dirichlet faces): least-squares orders
   over cell counts n = 2, 4, 6, 8 --
       Tet10: L2 order ~2.9 (expect 3), H1 order ~1.9 (expect 2)
       Tet4:  L2 order ~1.6 rising to 2, H1 order ~0.85 rising to 1
   At equal cell size n = 4 the Tet10 L2 error is already below the
   Tet4 error at n = 8.
5. Cantilever (the BIG-M6 geometry: L = 0.5 m, 0.05 x 0.05 m, steel,
   P = 1000 N as a uniform end traction, full model):
       Tet4  (100, 4, 8), 19200 tets: err vs Euler-Bernoulli -5.95 %
       Tet10 (25, 1, 2),    300 tets: -0.33 %
       Tet10 (50, 2, 4),   2400 tets: -0.03 % (err vs Timoshenko
       -0.81 %, i.e. essentially the converged 3D value; the residual
       is the Timoshenko shear share, not discretisation error).
   The (50, 2, 4) Tet10 mesh and the (100, 4, 8) Tet4 mesh have the
   same node count (4545): at equal DOFs the Tet10 tip error is
   ~180x smaller.
6. Elasticity patch tests: a constant-strain field is reproduced
   exactly (both elements); a quadratic displacement u_x = s x^2
   with the compensating constant body force -div(sigma) is
   reproduced exactly by Tet10 on the sphere mesh (displacement
   error ~1e-19, centroid stresses to ~1e-14 relative) but not by
   Tet4 (~9e-6). A fully constrained uniform temperature rise gives
   the closed-form hydrostatic stress -E alpha dT / (1 - 2 nu).
7. Transient heat (Tet10): the consistent mass matrix conserves the
   total heat of an adiabatic body to round-off (~1e-16); the cube
   cooling problem matches the product-of-slabs series solution at
   the centre; temporal orders from successive step halvings on a
   fixed mesh (full-field inf-norm, the v1.0 protocol): backward
   Euler ~0.9 (expect 1), Crank-Nicolson ~1.9-2.4 (expect 2).
"""

import numpy as np

from cloudfem.fem.tet10 import (
    assemble_mass_tet10, element_operators, quadrature,
    shape_functions, solve_tet10_poisson, transient_heat_tet10)
from cloudfem.fem.tet_poisson import (
    _tet_gradients, solve_tet_poisson)
from cloudfem.mesh.tet10_mesh import (
    Tet10Mesh, box_mesh_tet10, tet4_to_tet10)
from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.mesh.unstructured3d import sphere_mesh
from cloudfem.physics.elastic3d import (
    constitutive_3d, solve_elasticity_3d)
from cloudfem.physics.elastic_tet10 import (
    assemble_elasticity_tet10, element_stress_tet10,
    solve_elasticity_tet10)

FACES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


# ------------------------------------------------------------------
# 1. Shape functions and mass matrix
# ------------------------------------------------------------------

def test_shape_and_mass():
    Lnodes = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [.5, .5, 0, 0], [0, .5, .5, 0], [.5, 0, .5, 0],
        [.5, 0, 0, .5], [0, .5, 0, .5], [0, 0, .5, .5]])
    Nn, _ = shape_functions(Lnodes)
    kron = float(np.abs(Nn - np.eye(10)).max())
    Lq, wq = quadrature()
    Nq, _ = shape_functions(Lq)
    pu = float(np.abs(Nq.sum(axis=1) - 1.0).max())

    # one reference tet: corners (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    corners = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mids = np.array([0.5 * (corners[i] + corners[j]) for i, j in
                     ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))])
    mesh = Tet10Mesh(np.vstack([corners, mids]),
                     np.arange(10)[None, :], n_corners=4)
    C = assemble_mass_tet10(mesh, rho_cp=2.0)
    V = 2.0 / 6.0  # rho * volume
    rows = (C @ np.ones(10))
    expect = V * np.array([-1.0 / 20] * 4 + [1.0 / 5] * 6)
    row_err = float(np.abs(rows - expect).max())
    total_err = abs(float(rows.sum()) - V)
    sym_err = float(np.abs((C - C.T).toarray()).max())
    lam_min = float(np.linalg.eigvalsh(C.toarray())[0])
    return kron, pu, row_err, total_err, sym_err, lam_min


# ------------------------------------------------------------------
# 2. Mesh conversion
# ------------------------------------------------------------------

def test_mesh_conversion():
    m4 = box_mesh(3, 2, 2)
    m10 = tet4_to_tet10(m4)
    corner_ok = bool(np.array_equal(m10.nodes[:m4.n_nodes], m4.nodes))
    elems_ok = bool(np.array_equal(m10.elements[:, :4], m4.elements))
    vol_err = abs(float(m10.element_volumes().sum()) - 1.0)
    bnd_ok = all(set(m4.boundaries[s]) <= set(m10.boundaries[s])
                 for s in FACES)
    mb = box_mesh_tet10(2, 2, 2)
    face_counts = {s: len(mb.boundaries[s]) for s in FACES}
    return corner_ok, elems_ok, vol_err, bnd_ok, face_counts


# ------------------------------------------------------------------
# 3. Heat patch tests
# ------------------------------------------------------------------

def _boundary_mask(mesh):
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    for s in mesh.boundaries:
        mask[mesh.boundaries[s]] = True
    return mask


def test_heat_patch():
    ul = lambda x, y, z: 1.0 + 2.0 * x - y + 0.5 * z      # noqa: E731
    uq = lambda x, y, z: x**2 + 2.0 * y**2 - z**2 + x * y  # noqa: E731
    out = {}
    # linear field on a box: both elements exact
    bc = {s: ul for s in FACES}
    u = solve_tet10_poisson(box_mesh_tet10(2, 2, 2), f=0.0, bc=bc)
    m = box_mesh_tet10(2, 2, 2)
    out["lin10"] = float(np.abs(u - ul(*m.nodes.T)).max())
    m = box_mesh(2, 2, 2)
    u = solve_tet_poisson(m, f=0.0, bc=bc)
    out["lin4"] = float(np.abs(u - ul(*m.nodes.T)).max())
    # quadratic field on an unstructured sphere mesh: -lap(uq) = -4
    m4 = sphere_mesh(0.5, 0.22)
    m10 = tet4_to_tet10(m4)
    bc = {"boundary": uq}
    u10 = solve_tet10_poisson(m10, f=-4.0, bc=bc)
    u4 = solve_tet_poisson(m4, f=-4.0, bc=bc)
    out["quad10"] = float(np.abs(
        u10 - uq(*m10.nodes.T))[~_boundary_mask(m10)].max())
    out["quad4"] = float(np.abs(
        u4 - uq(*m4.nodes.T))[~_boundary_mask(m4)].max())
    return out


# ------------------------------------------------------------------
# 4. Heat convergence (manufactured solution)
# ------------------------------------------------------------------

def _ue(X):
    return (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
            * np.sin(np.pi * X[:, 2]))


def _ge(X):
    return np.stack([
        np.pi * np.cos(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
        * np.sin(np.pi * X[:, 2]),
        np.pi * np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])
        * np.sin(np.pi * X[:, 2]),
        np.pi * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
        * np.cos(np.pi * X[:, 2])], axis=1)


def _src(x, y, z):
    return 3.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) \
        * np.sin(np.pi * z)


def _errors10(n, Lq, wq):
    m = box_mesh_tet10(n, n, n)
    u = solve_tet10_poisson(m, k=1.0, f=_src, bc={s: 0.0 for s in FACES})
    L2 = H1 = 0.0
    for el in m.elements:
        G, N, xq, vol = element_operators(m, el)
        uh = N @ u[el]
        gh = np.einsum("a,qai->qi", u[el], G)
        L2 += vol * np.sum(wq * (uh - _ue(xq))**2)
        H1 += vol * np.sum(wq * np.sum((gh - _ge(xq))**2, axis=1))
    return np.sqrt(L2), np.sqrt(H1)


def _errors4(n, Lq, wq):
    m = box_mesh(n, n, n)
    u = solve_tet_poisson(m, k=1.0, f=_src, bc={s: 0.0 for s in FACES})
    L2 = H1 = 0.0
    for el in m.elements:
        p = m.nodes[el]
        G, vol = _tet_gradients(p)
        xq = Lq @ p
        uh = Lq @ u[el]
        gh = np.tile(u[el] @ G, (len(wq), 1))
        L2 += vol * np.sum(wq * (uh - _ue(xq))**2)
        H1 += vol * np.sum(wq * np.sum((gh - _ge(xq))**2, axis=1))
    return np.sqrt(L2), np.sqrt(H1)


def test_heat_convergence():
    Lq, wq = quadrature()
    ns = (2, 4, 6, 8)
    e10 = [_errors10(n, Lq, wq) for n in ns]
    e4 = [_errors4(n, Lq, wq) for n in ns]
    hs = np.log([1.0 / n for n in ns])
    fit = lambda errs, i: float(np.polyfit(  # noqa: E731
        hs, np.log([e[i] for e in errs]), 1)[0])
    orders = {"L2_10": fit(e10, 0), "H1_10": fit(e10, 1),
              "L2_4": fit(e4, 0), "H1_4": fit(e4, 1)}
    return ns, e10, e4, orders


# ------------------------------------------------------------------
# 5. Cantilever (BIG-M6 geometry, full model)
# ------------------------------------------------------------------

E_STEEL, NU_STEEL = 200e9, 0.3
L_BEAM, B_BEAM, H_BEAM, P_LOAD = 0.5, 0.05, 0.05, 1000.0
I_BEAM = B_BEAM * H_BEAM**3 / 12.0
A_BEAM = B_BEAM * H_BEAM
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))
DELTA_EB = P_LOAD * L_BEAM**3 / (3.0 * E_STEEL * I_BEAM)
DELTA_TIM = DELTA_EB + P_LOAD * L_BEAM / ((5.0 / 6.0) * G_STEEL * A_BEAM)


def _beam(mesh_fn, solver, nx, ny, nz):
    m = mesh_fn(nx, ny, nz, xlim=(0.0, L_BEAM), ylim=(0.0, B_BEAM),
                zlim=(0.0, H_BEAM))
    ux, uy, uz = solver(m, E_STEEL, NU_STEEL,
                        bc={"xmin": (0.0, 0.0, 0.0)},
                        traction={"xmax": (0.0, 0.0, -P_LOAD / A_BEAM)})
    tip = abs(float(uz[m.boundaries["xmax"]].mean()))
    return tip, m.n_elements, m.n_nodes


def test_cantilever():
    m = box_mesh_tet10(4, 1, 1, xlim=(0.0, L_BEAM), ylim=(0.0, B_BEAM),
                       zlim=(0.0, H_BEAM))
    _, rhs = assemble_elasticity_tet10(
        m, E_STEEL, NU_STEEL, traction={"xmax": (0.0, 0.0, -P_LOAD / A_BEAM)})
    resultant = float(rhs[2::3].sum())
    rows = []
    tip4, nt4, nn4 = _beam(box_mesh, solve_elasticity_3d, 100, 4, 8)
    rows.append(("Tet4", (100, 4, 8), nt4, nn4, tip4))
    for grid in ((25, 1, 2), (50, 2, 4)):
        tip, nt, nn = _beam(box_mesh_tet10, solve_elasticity_tet10, *grid)
        rows.append(("Tet10", grid, nt, nn, tip))
    return resultant, rows


# ------------------------------------------------------------------
# 6. Elasticity patch tests + thermal stress closed form
# ------------------------------------------------------------------

def test_elastic_patch():
    out = {}
    # (a) constant strain on a box: u = A x + c prescribed on all faces
    A = np.array([[1e-3, 2e-4, 0.0],
                  [0.0, -5e-4, 3e-4],
                  [1e-4, 0.0, 8e-4]])
    c = np.array([1e-3, -2e-3, 5e-4])
    bc = {}
    for s in FACES:
        bc[s] = tuple(
            (lambda x, y, z, r=r: A[r, 0] * x + A[r, 1] * y
             + A[r, 2] * z + c[r]) for r in range(3))
    m10 = box_mesh_tet10(2, 2, 2)
    u10 = solve_elasticity_tet10(m10, E_STEEL, NU_STEEL, bc=bc)
    ex = (m10.nodes @ A.T) + c
    got = np.stack(u10, axis=1)
    out["const10"] = float(np.abs(got - ex).max())
    D = constitutive_3d(E_STEEL, NU_STEEL)
    eps_v = np.array([A[0, 0], A[1, 1], A[2, 2],
                      A[1, 2] + A[2, 1], A[0, 2] + A[2, 0],
                      A[0, 1] + A[1, 0]])
    sig_ex = D @ eps_v
    sig10, _ = element_stress_tet10(m10, u10, E_STEEL, NU_STEEL)
    out["const10_sig"] = float(np.abs(sig10 - sig_ex).max()
                               / np.abs(sig_ex).max())
    m4 = box_mesh(2, 2, 2)
    u4 = solve_elasticity_3d(m4, E_STEEL, NU_STEEL, bc=bc)
    out["const4"] = float(np.abs(np.stack(u4, axis=1)
                                 - ((m4.nodes @ A.T) + c)).max())

    # (b) quadratic displacement on the sphere, compensating body force
    lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
    mu = E_STEEL / (2 * (1 + NU_STEEL))
    s = 1e-3
    body = (-2.0 * (lam + 2 * mu) * s, 0.0, 0.0)
    bc = {"boundary": (lambda x, y, z: s * x * x, 0.0, 0.0)}
    m4 = sphere_mesh(0.5, 0.22)
    m10 = tet4_to_tet10(m4)
    u10 = solve_elasticity_tet10(m10, E_STEEL, NU_STEEL, bc=bc, body=body)
    ex = np.zeros((m10.n_nodes, 3))
    ex[:, 0] = s * m10.nodes[:, 0]**2
    out["quad10"] = float(np.abs(np.stack(u10, axis=1) - ex)
                          [~_boundary_mask(m10)].max())
    sig10, _ = element_stress_tet10(m10, u10, E_STEEL, NU_STEEL)
    cen = np.array([m10.nodes[el[:4]].mean(axis=0)
                    for el in m10.elements])
    sxx = (lam + 2 * mu) * 2 * s * cen[:, 0]
    out["quad10_sig"] = float(np.abs(sig10[:, 0] - sxx).max()
                              / np.abs(sxx).max())
    u4 = solve_elasticity_3d(m4, E_STEEL, NU_STEEL, bc=bc, body=body)
    ex4 = np.zeros((m4.n_nodes, 3))
    ex4[:, 0] = s * m4.nodes[:, 0]**2
    out["quad4"] = float(np.abs(np.stack(u4, axis=1) - ex4)
                         [~_boundary_mask(m4)].max())

    # (c) fully constrained uniform heating: closed-form stress
    alpha, dT = 12e-6, 100.0
    m10 = box_mesh_tet10(2, 2, 2)
    u10 = solve_elasticity_tet10(
        m10, E_STEEL, NU_STEEL, bc={s: (0.0, 0.0, 0.0) for s in FACES},
        T=lambda x, y, z: dT, alpha=alpha, T_ref=0.0)
    sig10, _ = element_stress_tet10(m10, u10, E_STEEL, NU_STEEL,
                                    T=lambda x, y, z: dT, alpha=alpha)
    sig_ex = -E_STEEL * alpha * dT / (1 - 2 * NU_STEEL)
    out["thermal_sig"] = float(np.abs(sig10[:, :3] - sig_ex).max()
                               / abs(sig_ex))
    out["thermal_shear"] = float(np.abs(sig10[:, 3:]).max() / abs(sig_ex))
    out["thermal_u"] = float(np.abs(np.stack(u10, axis=1)).max())
    return out


# ------------------------------------------------------------------
# 7. Transient heat
# ------------------------------------------------------------------

def _slab_theta(t, n_terms=200):
    n = np.arange(n_terms)
    return float(np.sum(4.0 / (np.pi * (2 * n + 1)) * (-1.0)**n
                        * np.exp(-((2 * n + 1) * np.pi)**2 * t)))


def test_transient():
    out = {}
    m = box_mesh_tet10(4, 4, 4)
    cen = int(np.argmin(np.sum((m.nodes - 0.5)**2, axis=1)))
    # cube cooling vs product-of-slabs series (centre value)
    _, TH = transient_heat_tet10(m, k=1.0, rho_cp=1.0,
                                 bc={s: 0.0 for s in FACES},
                                 t_end=0.1, dt=0.005, T0=1.0, method="cn")
    out["cool_err"] = float(TH[-1, cen] - _slab_theta(0.1)**3)
    # adiabatic conservation of the consistent-mass total heat
    C = assemble_mass_tet10(m, rho_cp=1.0)
    rng = np.random.default_rng(7)
    T0 = rng.random(m.n_nodes)
    _, TH = transient_heat_tet10(m, k=1.0, rho_cp=1.0, bc=None,
                                 t_end=0.05, dt=0.01, T0=T0, method="cn")
    heat = TH @ (C @ np.ones(m.n_nodes))
    out["drift"] = float(np.abs(heat - heat[0]).max() / abs(heat[0]))
    # temporal orders (v1.0 protocol: full-field inf-norm differences)
    m6 = box_mesh_tet10(6, 6, 6)
    x, y, z = m6.nodes[:, 0], m6.nodes[:, 1], m6.nodes[:, 2]
    T0f = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    bc = {s: 0.0 for s in FACES}
    for method in ("be", "cn"):
        diffs, prev = [], None
        for dt in (0.01, 0.005, 0.0025, 0.00125):
            _, hist = transient_heat_tet10(
                m6, k=1.0, rho_cp=1.0, bc=bc, t_end=0.05, dt=dt,
                T0=T0f, method=method)
            if prev is not None:
                diffs.append(float(np.abs(hist[-1] - prev).max()))
            prev = hist[-1]
        out[method] = [float(np.log(diffs[i] / diffs[i + 1])
                             / np.log(2.0))
                       for i in range(len(diffs) - 1)]
    return out


# ------------------------------------------------------------------

def main():
    kron, pu, row_err, total_err, sym_err, lam_min = test_shape_and_mass()
    print(f"[shape] Kronecker err = {kron:.3e}, partition-of-unity "
          f"err = {pu:.3e}")
    print(f"[mass] row-sum err = {row_err:.3e}, total err = "
          f"{total_err:.3e}, symmetry err = {sym_err:.3e}, "
          f"min eigenvalue = {lam_min:.3e}")
    assert kron < 1e-14 and pu < 1e-14
    assert row_err < 1e-13 and total_err < 1e-13 and sym_err < 1e-13
    assert lam_min > 0.0, "Tet10 mass matrix is not positive definite"

    corner_ok, elems_ok, vol_err, bnd_ok, fc = test_mesh_conversion()
    print(f"[mesh] corners preserved = {corner_ok}, elements "
          f"preserved = {elems_ok}, volume err = {vol_err:.3e}, "
          f"boundary superset = {bnd_ok}, face counts = {fc}")
    assert corner_ok and elems_ok and bnd_ok and vol_err < 1e-12
    assert all(v == 25 for v in fc.values())

    hp = test_heat_patch()
    print(f"[heat patch] linear: Tet10 = {hp['lin10']:.3e}, "
          f"Tet4 = {hp['lin4']:.3e}; quadratic (sphere, interior): "
          f"Tet10 = {hp['quad10']:.3e}, Tet4 = {hp['quad4']:.3e}")
    assert hp["lin10"] < 1e-11 and hp["lin4"] < 1e-11
    assert hp["quad10"] < 1e-11, "Tet10 must reproduce quadratics"
    assert hp["quad4"] > 1e-3, "Tet4 must NOT reproduce quadratics"

    ns, e10, e4, orders = test_heat_convergence()
    for n, a, b in zip(ns, e10, e4):
        print(f"[conv n={n}] Tet10 L2 = {a[0]:.4e} H1 = {a[1]:.4e} | "
              f"Tet4 L2 = {b[0]:.4e} H1 = {b[1]:.4e}")
    print(f"[conv orders] Tet10 L2 = {orders['L2_10']:.2f} "
          f"H1 = {orders['H1_10']:.2f} | Tet4 L2 = {orders['L2_4']:.2f} "
          f"H1 = {orders['H1_4']:.2f}")
    assert 2.6 < orders["L2_10"] < 3.4 and 1.7 < orders["H1_10"] < 2.3
    assert 1.4 < orders["L2_4"] < 2.4 and 0.7 < orders["H1_4"] < 1.3
    assert e10[1][0] < e4[3][0], \
        "Tet10 at n=4 should beat Tet4 at n=8 in L2"

    resultant, rows = test_cantilever()
    print(f"[cantilever] Tet10 traction resultant = {resultant:.6e} N "
          f"(expect {-P_LOAD:.6e})")
    assert abs(resultant + P_LOAD) < 1e-9 * P_LOAD
    errs = {}
    for tag, grid, nt, nn, tip in rows:
        e_eb = (tip - DELTA_EB) / DELTA_EB
        e_tm = (tip - DELTA_TIM) / DELTA_TIM
        errs.setdefault(tag, []).append((grid, e_eb, e_tm))
        print(f"[cantilever] {tag} {grid}: tets = {nt}, nodes = {nn}, "
              f"tip = {tip * 1e3:.5f} mm, err EB = {100 * e_eb:+.3f} %, "
              f"err Timoshenko = {100 * e_tm:+.3f} %")
    e4_eb = abs(errs["Tet4"][0][1])
    e10 = errs["Tet10"]
    assert abs(e10[0][1]) > abs(e10[1][1]), "Tet10 levels not monotone"
    assert abs(e10[1][1]) < 0.005 and abs(e10[1][2]) < 0.01
    assert abs(e10[1][1]) < e4_eb / 10.0, \
        "Tet10 must be >10x more accurate than Tet4 on the cantilever"

    ep = test_elastic_patch()
    print(f"[elastic patch] constant strain: Tet10 u err = "
          f"{ep['const10']:.3e}, stress rel err = {ep['const10_sig']:.3e}, "
          f"Tet4 u err = {ep['const4']:.3e}")
    print(f"[elastic patch] quadratic (sphere): Tet10 u err = "
          f"{ep['quad10']:.3e}, stress rel err = {ep['quad10_sig']:.3e}, "
          f"Tet4 u err = {ep['quad4']:.3e}")
    print(f"[thermal stress] sigma rel err = {ep['thermal_sig']:.3e}, "
          f"shear ratio = {ep['thermal_shear']:.3e}, "
          f"|u| max = {ep['thermal_u']:.3e}")
    assert ep["const10"] < 1e-12 and ep["const4"] < 1e-12
    assert ep["const10_sig"] < 1e-9
    assert ep["quad10"] < 1e-14 and ep["quad10_sig"] < 1e-8
    assert ep["quad4"] > 1e-6
    assert ep["thermal_sig"] < 1e-9 and ep["thermal_shear"] < 1e-9
    assert ep["thermal_u"] < 1e-12

    tr = test_transient()
    print(f"[transient] cube-cooling centre err = {tr['cool_err']:+.3e}, "
          f"adiabatic heat drift = {tr['drift']:.3e}")
    print(f"[transient] temporal orders BE = "
          f"{[f'{o:.2f}' for o in tr['be']]}, CN = "
          f"{[f'{o:.2f}' for o in tr['cn']]}")
    assert abs(tr["cool_err"]) < 5e-3
    assert tr["drift"] < 1e-12
    assert all(0.75 < o < 1.25 for o in tr["be"])
    assert all(1.6 < o < 2.6 for o in tr["cn"])

    print("ALL TET10 VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
