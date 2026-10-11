"""Sequential EM-thermal coupling for induction heating.

Workflow: solve eddy currents -> Joule heating Q -> transient heat.
Material properties are taken as temperature-independent in this MVP
(temperature-dependent sigma/k is on the roadmap).
"""

import numpy as np

from ..fem.timestep import (
    implicit_euler_heat, source_load_vector, element_to_nodal)
from .em import solve_eddy_current, joule_heating_density, total_joule_power, MU0


def induction_heating(mesh, sigma, omega, J_s=0.0, nu=None,
                      k_th=1.0, rho_cp=1.0, bc_em=None, bc_th=None,
                      t_end=1.0, dt=0.01, T0=0.0, depth=1.0):
    """Run a sequential induction-heating simulation.

    Returns dict with A (phasor), Q_nodal, Q_elem, times, T_history, P_total.
    """
    if nu is None:
        nu = 1.0 / MU0

    A = solve_eddy_current(mesh, nu=nu, sigma=sigma, omega=omega,
                           J_s=J_s, bc=bc_em)
    Q_elem, _ = joule_heating_density(mesh, A, sigma, omega)
    Q_nodal = element_to_nodal(mesh, Q_elem)
    P_total = total_joule_power(mesh, A, sigma, omega, depth=depth)

    rhs_q = source_load_vector(mesh, Q_nodal)
    times, T_hist = implicit_euler_heat(
        mesh, k_th, rho_cp, rhs_q, bc_th, t_end, dt, T0)

    return {
        "A": A,
        "Q_nodal": Q_nodal,
        "Q_elem": Q_elem,
        "times": times,
        "T_history": T_hist,
        "P_total": P_total,
    }
