"""3D transient eddy currents: Tet10 nodal A-V, theta method (v1.8).

Validates cloudfem.physics.em3d_tet10.transient_eddy_current_tet10:

1. DC step (BE): the long-time limit matches an independent
   low-frequency phasor solve (relative diff < 1e-4), and the Joule
   power decays to ~0 (eddy currents vanish at DC steady state).
2. Temporal order by step-halving differences of the solution:
   BE ~ 1 on the step problem; CN ~ 2 on the smooth sinusoidal
   problem (Fourier-extracted phasor amplitude vs the v1.7 phasor
   solver at two step sizes).
3. Sinusoidal drive (CN): last-period Fourier extraction of the
   complex amplitude vs solve_eddy_current_tet10 (rel. L2 < 1%).
4. Power closure: the transient Joule power (E = -dA/dt - grad V,
   exact mass matrix) vs an independent 5-point quadrature of |E|^2.
5. Documented negative result: CN on a step-discontinuous source
   keeps a non-decaying stiff-mode ringing (gauge regularisation
   modes, R(z) -> -1); the test asserts the BE/CN disagreement so
   the limitation is pinned, not hidden.  CN is for smooth
   time_scale only; BE (L-stable) is the robust default.

All numbers printed by this file are simulation output.
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import box_mesh_tet10
from cloudfem.physics.em3d_tet10 import (
    assemble_tet10_av, solve_eddy_current_tet10,
    transient_eddy_current_tet10, transient_joule_power_tet10)
from cloudfem.physics.em import MU0

PI = np.pi
NAMES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


def _block_sigma(x, y, z):
    return (1e6 if (0.35 <= x <= 0.65 and 0.35 <= y <= 0.65
                    and 0.35 <= z <= 0.65) else 0.0)


def _coil_Js(x, y, z):
    return ((0.0, 0.0, 1e6)
            if (0.35 <= x <= 0.65 and 0.35 <= y <= 0.65
                and 0.70 <= z <= 0.90) else (0.0, 0.0, 0.0))


# ----------------------------------------------------------------------
# 1. DC step -> steady state (BE)
# ----------------------------------------------------------------------
def test_dc_step_steady_state():
    mesh = box_mesh_tet10(4, 4, 4)
    out = transient_eddy_current_tet10(mesh, sigma=_block_sigma,
                                       t_end=3.0, dt=0.02, method="be",
                                       J_s=_coil_Js, boundary=NAMES)
    # independent DC limit: phasor solve at very low frequency
    a_ph, _ = solve_eddy_current_tet10(mesh, sigma=_block_sigma,
                                       omega=2 * PI * 0.01,
                                       J_s=_coil_Js, boundary=NAMES)
    rel = (np.linalg.norm(out["a_end"] - a_ph.real)
           / np.linalg.norm(a_ph.real))
    prat = out["P"][-1] / out["P"].max()
    print(f"[dc-step] a_end vs low-freq phasor rel diff = {rel:.3e}, "
          f"P_end/P_max = {prat:.3e}")
    assert rel < 1e-4, "transient DC limit off the phasor DC limit"
    assert prat < 1e-6, "eddy power did not decay at DC steady state"
    return rel, prat


# ----------------------------------------------------------------------
# 2. Temporal orders
# ----------------------------------------------------------------------
def test_temporal_order_be():
    mesh = box_mesh_tet10(4, 4, 4)
    As = [transient_eddy_current_tet10(
        mesh, sigma=_block_sigma, t_end=0.06, dt=dt, method="be",
        J_s=_coil_Js, boundary=NAMES)["a_end"]
        for dt in (0.02, 0.01, 0.005)]
    d1 = np.linalg.norm(As[0] - As[1])
    d2 = np.linalg.norm(As[1] - As[2])
    order = float(np.log2(d1 / d2))
    print(f"[order-be] step-halving diffs {d1:.4g} {d2:.4g}, "
          f"order = {order:.2f}")
    assert 0.7 < order < 2.0, "BE temporal order off"
    return order


def _sinusoidal_phasor_error(mesh, dt, freq=50.0, t_end=4.0):
    omega = 2 * PI * freq
    T = 1.0 / freq

    def sigma3(x, y, z):
        return (1e6 if (1 / 3 <= x <= 2 / 3 and 1 / 3 <= y <= 2 / 3
                        and 1 / 3 <= z <= 2 / 3) else 0.0)

    def Js3(x, y, z):
        return ((0.0, 0.0, 1e6)
                if (1 / 3 <= x <= 2 / 3 and 1 / 3 <= y <= 2 / 3
                    and 0.70 <= z <= 0.90) else (0.0, 0.0, 0.0))

    out = transient_eddy_current_tet10(
        mesh, sigma=sigma3, t_end=t_end, dt=dt, method="cn", J_s=Js3,
        time_scale=lambda t: np.sin(omega * t), boundary=NAMES,
        history=(t_end - T, t_end))
    th, ah = out["t_hist"], out["a_hist"]
    w = np.exp(-1j * omega * th)
    a_num = (2.0 / T) * np.trapz(ah * w[:, None], th, axis=0)
    a_ph, _ = solve_eddy_current_tet10(
        mesh, sigma=sigma3, omega=omega, boundary=NAMES,
        J_s=lambda x, y, z: tuple(-1j * c for c in Js3(x, y, z)))
    # sin(wt) drive <-> phasor -i*J0 (Re{-i J0 e^{iwt}} = J0 sin(wt))
    return float(np.linalg.norm(a_num - a_ph) / np.linalg.norm(a_ph))


def test_sinusoidal_vs_phasor():
    mesh = box_mesh_tet10(3, 3, 3)
    err = _sinusoidal_phasor_error(mesh, dt=(1 / 50.0) / 40)
    print(f"[sin-cn] Fourier amplitude vs phasor rel L2 = {err:.3e}")
    assert err < 1e-2, "transient sinusoidal steady state off phasor"
    return err


def test_temporal_order_cn():
    mesh = box_mesh_tet10(3, 3, 3)
    e1 = _sinusoidal_phasor_error(mesh, dt=(1 / 50.0) / 20)
    e2 = _sinusoidal_phasor_error(mesh, dt=(1 / 50.0) / 40)
    order = float(np.log2(e1 / e2))
    print(f"[order-cn] errs {e1:.3e} {e2:.3e}, order = {order:.2f}")
    assert 1.6 < order < 2.4, "CN temporal order off on smooth problem"
    return order


# ----------------------------------------------------------------------
# 4. Power closure: exact-mass vs independent high-order quadrature
# ----------------------------------------------------------------------
def _duffy_rule(n=4):
    """Degree-7 tensor Gauss-Legendre on the reference tet via Duffy.

    Maps [0,1]^3 -> tet{(x,y,z): x,y,z>=0, x+y+z<=1} with
    x = u, y = v(1-u), z = w(1-u)(1-v), Jacobian (1-u)^2(1-v).
    n = 4 GL points per direction integrates the degree-4 |E|^2
    integrand exactly.  Returns (pts (M,3), w (M,)).
    """
    u, wu = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (u + 1.0)
    wu = 0.5 * wu
    pts, w = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = u[i]
                y = u[j] * (1 - u[i])
                z = u[k] * (1 - u[i]) * (1 - u[j])
                jac = (1 - u[i]) ** 2 * (1 - u[j])
                pts.append((x, y, z))
                w.append(wu[i] * wu[j] * wu[k] * jac)
    return np.array(pts), np.array(w)


def _check_duffy():
    """Self-verify the Duffy rule on monomials up to degree 4."""
    pts, w = _duffy_rule()
    # exact integrals on the reference tet: int x^a y^b z^c =
    # a! b! c! / (a+b+c+3)!
    import math
    for (a_, b_, c_) in [(4, 0, 0), (0, 4, 0), (3, 1, 0), (2, 2, 0),
                         (2, 1, 1), (1, 1, 2)]:
        num = float(np.sum(w * pts[:, 0] ** a_ * pts[:, 1] ** b_
                           * pts[:, 2] ** c_))
        exact = (math.factorial(a_) * math.factorial(b_)
                 * math.factorial(c_)
                 / math.factorial(a_ + b_ + c_ + 3))
        assert abs(num - exact) / exact < 1e-10, \
            f"Duffy rule failed on x^{a_}y^{b_}z^{c_}"
    return True


def test_power_closure():
    assert _check_duffy()
    mesh = box_mesh_tet10(4, 4, 4)
    out = transient_eddy_current_tet10(mesh, sigma=_block_sigma,
                                       t_end=0.1, dt=0.02, method="be",
                                       J_s=_coil_Js, boundary=NAMES,
                                       history=True)
    ah = out["a_hist"]
    k = 2  # a mid-transient step
    a_prev, a_new = ah[k - 1], ah[k]
    h = out["times"][k] - out["times"][k - 1]
    # the constraint-consistent V of step k (same construction as the
    # solver's diagnostics): V = -S^-1 G^T adot
    import scipy.sparse.linalg as spla
    _, _, Gs, S, v_nodes, _, _ = assemble_tet10_av(mesh,
                                                   sigma=_block_sigma)
    solve_S = spla.factorized(S.tocsc())
    vv = np.zeros(mesh.n_nodes)
    vv[v_nodes] = -solve_S(Gs.T @ ((a_new - a_prev) / h))
    P_mass, _ = transient_joule_power_tet10(mesh, a_prev, a_new, vv,
                                            _block_sigma, h)
    # independent Duffy quadrature of 0.5*sigma*|E|^2,
    # E = -(a-a_prev)/h - grad V, mapped per element
    from cloudfem.fem.tet10 import shape_functions as _sh
    dpts, dw = _duffy_rule()
    Pq = 0.0
    ap = a_prev.reshape(-1, 3)
    an = a_new.reshape(-1, 3)
    for e, el in enumerate(np.asarray(mesh.elements)):
        corners = mesh.nodes[el[:4]]
        xc = corners.mean(axis=0)
        if _block_sigma(*xc) <= 0.0:
            continue
        M3 = np.column_stack([corners[1] - corners[0],
                              corners[2] - corners[0],
                              corners[3] - corners[0]])
        vol = abs(np.linalg.det(M3)) / 6.0
        Minv = np.linalg.inv(M3)
        # dlam_i/dx_j for the affine barycentric map
        dlam_dx = np.vstack([-Minv.sum(axis=0), Minv])  # (4,3)
        Eq = np.zeros((len(dpts), 3))
        for q in range(len(dpts)):
            x_, y_, z_ = dpts[q]
            lam = np.array([1.0 - x_ - y_ - z_, x_, y_, z_])
            Nq, dN_dL = _sh(lam)          # (10,), (10,4)
            dNdx = dN_dL @ dlam_dx          # (10,3) physical grads
            Aq = Nq @ an[el]
            Apq = Nq @ ap[el]
            Eq[q] = -((Aq - Apq) / h + dNdx.T @ vv[el])
        Pq += 0.5 * 1e6 * (vol / (1.0 / 6.0)) * float(np.sum(
            dw * np.einsum("qm,qm->q", Eq, Eq)))
    rel = abs(P_mass - Pq) / P_mass
    print(f"[power] P_mass = {P_mass:.6e} W, P_duffy = {Pq:.6e} W, "
          f"rel diff = {rel:.2e}")
    assert rel < 2e-2, "transient moment/quad power mismatch"
    return rel


# ----------------------------------------------------------------------
# 5. Documented negative result: CN ringing on step sources
# ----------------------------------------------------------------------
def test_cn_step_ringing_documented():
    """CN keeps a non-decaying stiff-mode ringing on step sources.

    BE (L-stable) damps the gauge-regularisation modes; CN (R->-1)
    does not.  This test pins the disagreement so the limitation
    cannot silently regress: CN is for smooth time_scale only.
    """
    mesh = box_mesh_tet10(4, 4, 4)
    abe = transient_eddy_current_tet10(
        mesh, sigma=_block_sigma, t_end=2.0, dt=0.01, method="be",
        J_s=_coil_Js, boundary=NAMES)["a_end"]
    acn = transient_eddy_current_tet10(
        mesh, sigma=_block_sigma, t_end=2.0, dt=0.01, method="cn",
        J_s=_coil_Js, boundary=NAMES)["a_end"]
    rel = float(np.linalg.norm(abe - acn) / np.linalg.norm(abe))
    print(f"[cn-ringing] BE vs CN a_end rel diff on step source = {rel:.3f} "
          f"(expected large: documents CN's stiff ringing)")
    assert rel > 0.1, "CN step ringing unexpectedly absent"
    return rel


if __name__ == "__main__":
    test_dc_step_steady_state()
    test_temporal_order_be()
    test_sinusoidal_vs_phasor()
    test_temporal_order_cn()
    test_power_closure()
    test_cn_step_ringing_documented()
    print("ALL TRANSIENT EDDY-CURRENT VALIDATIONS PASSED")
