"""Heat transfer physics: steady-state and transient conduction.

Steady:   -div(k grad T) = Q
Transient: rho*cp dT/dt - div(k grad T) = Q   (implicit Euler)
"""

import numpy as np

from ..fem.poisson import solve_poisson, assemble
from ..fem.timestep import implicit_euler_heat, source_load_vector


def steady_heat(mesh, k=1.0, Q=0.0, bc=None):
    """Steady-state heat conduction. Returns nodal temperature T."""
    return solve_poisson(mesh, k=k, f=Q, bc=bc)


def transient_heat(mesh, k=1.0, rho_cp=1.0, Q=0.0, bc=None,
                   t_end=1.0, dt=0.01, T0=0.0):
    """Transient heat conduction with implicit Euler.

    Q may be a float, a callable f(x, y), or a nodal array.
    Returns (times, T_history) with T_history shape (n_steps+1, N).
    """
    if isinstance(Q, np.ndarray):
        rhs_q = source_load_vector(mesh, Q)
    else:
        _, rhs_q = assemble(mesh, k=0.0, f=Q)
    return implicit_euler_heat(mesh, k, rho_cp, rhs_q, bc, t_end, dt, T0)
