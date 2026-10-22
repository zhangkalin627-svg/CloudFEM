"""Coil terminal inductance extraction + EFD analytic calibration (v1.8).

Two ways to get a coil's inductance from a 3D phasor eddy-current
solution, plus the analytic "active-coil" formula from Zhang's EFD
course note (2026-10-21)::

    L ~= mu0 * mu_r * N^2 * A_m / l_coil

where A_m is the magnetic cross-section: dominated by the
coil-workpiece air-gap area, plus the skin-layer areas where the
field penetrates the copper (delta_Cu) and the workpiece (delta_w);
the workpiece part carries the mu_r weight when the workpiece is
magnetic.  In the fully magnetic-core limit the weighted form below
reduces to the lecture-note form.

All numbers produced here are simulation output.
"""

import numpy as np

from .em import MU0
from .em3d_tet10 import (element_flux_density_tet10,
                         magnetic_energy_tet10,
                         source_complex_power_tet10,
                         total_joule_power_tet10)

__all__ = [
    "terminal_inductance_tet10",
    "inductance_active_coil_efd",
]


def terminal_inductance_tet10(mesh, a, v, J_s, omega, I_total,
                              nu=1.0 / MU0, sigma=0.0):
    """Terminal R/L/Z of a coil from a phasor eddy-current solution.

    Parameters
    ----------
    a, v : phasor potentials from solve_eddy_current_tet10.
    J_s : the imposed source-current callable used in the solve.
    omega : angular frequency [rad/s].
    I_total : total coil current phasor amplitude [A] belonging to
        J_s (so that Z = 2*S/|I|^2 is the terminal impedance).

    Returns dict(R_ac, L_energy, L_circuit, Z): R_ac = 2P/|I|^2 with
    the time-averaged Joule power P; L_energy = 4*W_m/|I|^2 with the
    time-averaged stored magnetic energy W_m (peak-phasor
    <W_m> = L|I|^2/4); L_circuit = Im(Z)/omega with Z = 2*S/|I|^2
    from the complex source power.  The two L estimates must agree
    (energy vs circuit口径互验; exact when the J_s quadrature is
    exact).
    """
    if omega <= 0:
        raise ValueError("omega must be positive")
    I2 = abs(complex(I_total)) ** 2
    if I2 == 0.0:
        raise ValueError("I_total must be nonzero")
    P, _ = total_joule_power_tet10(mesh, a, v, sigma, omega)
    W = magnetic_energy_tet10(mesh, a, nu=nu)
    S = source_complex_power_tet10(mesh, a, v, J_s, omega)
    Z = 2.0 * S / I2
    # Peak-phasor conventions: <P> = R|I|^2/2, <W_m> = L|I|^2/4.
    return {"R_ac": 2.0 * P / I2,
            "L_energy": 4.0 * W / I2,
            "L_circuit": Z.imag / omega,
            "Z": complex(Z)}


def inductance_active_coil_efd(N, l_coil, r_coil_in, r_wp,
                               delta_cu=0.0, delta_w=0.0, mu_r_w=1.0):
    """EFD active-coil inductance estimate [H] (Zhang's course note).

    L ~= mu0 * N^2 * A_m / l_coil with the weighted magnetic area

        A_m = A_gap + A_Cu + mu_r_w * A_w,

    * A_gap = pi * (r_coil_in^2 - r_wp^2): coil-workpiece air gap;
    * A_Cu  = pi * ((r_coil_in + delta_cu)^2 - r_coil_in^2): copper
      skin ring the field penetrates (delta_cu = copper skin depth);
    * A_w   = pi * (r_wp^2 - (r_wp - delta_w)^2): workpiece skin ring
      (delta_w = workpiece skin depth), weighted by mu_r_w when the
      workpiece is magnetic.

    For a fully magnetic core (r_wp -> r_coil_in, mu_r_w = mu_r) this
    reduces to the lecture-note form L ~= mu0*mu_r*N^2*A_m/l_coil.
    """
    if l_coil <= 0 or N <= 0:
        raise ValueError("N and l_coil must be positive")
    r_ci = float(r_coil_in)
    r_w = float(r_wp)
    if not (0.0 < r_w <= r_ci):
        raise ValueError("need 0 < r_wp <= r_coil_in")
    dcu = max(float(delta_cu), 0.0)
    dw = min(max(float(delta_w), 0.0), r_w)
    A_gap = np.pi * (r_ci ** 2 - r_w ** 2)
    A_cu = np.pi * ((r_ci + dcu) ** 2 - r_ci ** 2)
    A_w = np.pi * (r_w ** 2 - (r_w - dw) ** 2)
    A_m = A_gap + A_cu + float(mu_r_w) * A_w
    return float(MU0 * N ** 2 * A_m / l_coil)
