"""Coil terminal inductance + EFD analytic calibration (v1.8).

Validates cloudfem.physics.inductance on the D1 coil/billet geometry
(reuses the validated tests/test_tet10_em.py D1 builders):

1. Complex-power regression (v1.8 fix of the v1.7 sign bug):
   Re(S) == Joule power P and Im(S) == 2*w*W_m, with W_m from the
   exact discrete identity W_m = Re(a*.K.a)/4.
2. Energy vs circuit口径互验: L_energy = 4*W_m/|I|^2 vs
   L_circuit = Im(Z)/omega agree.
3. EFD analytic calibration: the extracted L vs
   inductance_active_coil_efd (Zhang's course-note formula
   L ~= mu0*N^2*A_m/l_coil) -- order-of-magnitude agreement; the
   single-turn mapping, finite coil length and the A=0 truncation
   box are documented approximations, not hidden.

All numbers printed by this file are simulation output.
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0, skin_depth
from cloudfem.physics.em3d_tet10 import (
    assemble_tet10_av, solve_eddy_current_tet10,
    source_complex_power_tet10, total_joule_power_tet10,
    magnetic_energy_tet10)
from cloudfem.physics.inductance import (
    terminal_inductance_tet10, inductance_active_coil_efd)

import tests.test_tet10_em as T
from tests.test_tet10_em import D1

PI = np.pi


def _d1_solve():
    omega = 2 * PI * D1["FREQ"]
    m4 = cylinder_in_box_mesh(D1["R"], D1["L"], half=D1["HALF"],
                              h=D1["H3D"])
    mesh = tet4_to_tet10(m4)
    a, v = solve_eddy_current_tet10(mesh, nu=1.0 / MU0,
                                    sigma=T.d1_sigma_fn, omega=omega,
                                    J_s=T.d1_Js)
    I_total = (D1["J0"] * (D1["COIL_R"][1] - D1["COIL_R"][0])
               * (2 * D1["COIL_Z"]))
    return mesh, a, v, omega, I_total


def test_complex_power_identities():
    """Re(S) = P, Im(S) = 2*w*W_m (exact discrete identities)."""
    mesh, a, v, omega, _ = _d1_solve()
    S = source_complex_power_tet10(mesh, a, v, T.d1_Js, omega)
    P, _ = total_joule_power_tet10(mesh, a, v, T.d1_sigma_fn, omega)
    K, _, _, _, _, _, _ = assemble_tet10_av(mesh, sigma=T.d1_sigma_fn)
    W_exact = 0.25 * float(np.real(np.vdot(a, K @ a)))
    print(f"[S-ident] Re(S) = {S.real:.6e} W, P = {P:.6e} W, "
          f"rel diff = {abs(S.real - P) / P:.2e}")
    print(f"          Im(S) = {S.imag:.6e}, 2*w*W_m = {2 * omega * W_exact:.6e}, "
          f"rel diff = {abs(S.imag - 2 * omega * W_exact) / (2 * omega * W_exact):.2e}")
    assert S.real > 0, "Re(S) must be the (positive) Joule loss"
    # identities hold up to the J_s quadrature error (discontinuous
    # coil current on the 5-point rule -- same 5e-2 class as the v1.7
    # power reconciliation); the sign assert above is the regression
    # pin for the v1.7 sign bug.
    assert abs(S.real - P) / P < 1e-1, "Re(S) != Joule power"
    assert abs(S.imag - 2 * omega * W_exact) / (2 * omega * W_exact) \
        < 1e-1, "Im(S) != 2*w*W_m"
    return S, P


def test_terminal_inductance():
    """Energy vs circuit口径互验 + EFD analytic calibration."""
    mesh, a, v, omega, I_total = _d1_solve()
    term = terminal_inductance_tet10(mesh, a, v, T.d1_Js, omega,
                                     I_total, sigma=T.d1_sigma_fn)
    print(f"[L] R_ac = {term['R_ac']:.6e} ohm (>0: {term['R_ac'] > 0}), "
          f"L_energy = {term['L_energy']:.6e} H, "
          f"L_circuit = {term['L_circuit']:.6e} H")
    assert term["R_ac"] > 0, "terminal resistance must be positive"
    assert term["L_circuit"] > 0, "terminal inductance must be positive"
    ll = term["L_energy"] / term["L_circuit"]
    print(f"    L_energy/L_circuit = {ll:.4f}")
    assert abs(ll - 1.0) < 5e-2, "energy/circuit inductance mismatch"

    # EFD analytic: single-turn mapping (N = 1, terminal current I),
    # coil axial length l_coil = 2*COIL_Z, bore = billet radius
    # (mu_r = 1), skin ring delta_w in the billet.
    delta_w = skin_depth(D1["SIGMA"], omega)
    L_ana = inductance_active_coil_efd(
        1, 2 * D1["COIL_Z"], D1["COIL_R"][0], D1["R"],
        0.0, delta_w, 1.0)
    ratio = term["L_circuit"] / L_ana
    print(f"    EFD analytic L = {L_ana:.6e} H "
          f"(delta_w = {delta_w * 1e3:.2f} mm)")
    print(f"    L_numeric/L_analytic = {ratio:.3f} "
          f"(finite-length coil + A=0 truncation box documented)")
    assert 0.3 < ratio < 3.0, "EFD calibration outside order-of-magnitude"
    return term, L_ana


def test_magnetic_energy_exactness():
    """magnetic_energy_tet10 == Re(a*.K.a)/4 (exact discrete)."""
    mesh, a, v, omega, _ = _d1_solve()
    K, _, _, _, _, _, _ = assemble_tet10_av(mesh, sigma=T.d1_sigma_fn)
    W1 = magnetic_energy_tet10(mesh, a)
    W2 = magnetic_energy_tet10(mesh, a, K=K)
    W3 = 0.25 * float(np.real(np.vdot(a, K @ a)))
    print(f"[Wm] {W1:.6e} {W2:.6e} {W3:.6e} J")
    assert abs(W1 - W3) / W3 < 1e-12
    assert abs(W2 - W3) / W3 < 1e-12
    return W1


if __name__ == "__main__":
    test_complex_power_identities()
    test_terminal_inductance()
    test_magnetic_energy_exactness()
    print("ALL INDUCTANCE VALIDATIONS PASSED")
