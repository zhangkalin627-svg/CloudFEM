"""Sequential thermal -> structural coupling (thermal stress).

Workflow: a temperature field (e.g. from an induction-heating run) is
converted into a thermal-strain load and solved as a linear elasticity
problem. One-way coupling: deformation does not feed back into the
thermal/EM fields, which is the standard assumption for
induction-heating thermal-stress analysis at moderate strains.
"""

import numpy as np

from ..fem.timestep import element_to_nodal
from .coupled import induction_heating
from .elastic import (
    solve_elasticity, element_stress_strain, von_mises)


def thermal_stress(mesh, T, E, nu, alpha, T_ref=0.0, plane="strain", bc=None):
    """Compute thermal stress from a nodal temperature field T.

    Parameters
    ----------
    T : (N,) nodal temperature array (same mesh).
    E, nu : Young's modulus, Poisson's ratio.
    alpha : thermal expansion coefficient (1/K).
    T_ref : stress-free reference temperature.
    plane : 'strain' or 'stress'.
    bc : displacement BCs, see elastic.apply_dirichlet_vec.
         For a free body under self-equilibrated thermal load, pin just
         enough DOFs to remove rigid-body motion (reactions are ~0).

    Returns dict with ux, uy, sig_elem, eps_elem, vm_elem, vm_nodal.
    """
    ux, uy = solve_elasticity(mesh, E, nu, plane=plane, bc=bc,
                              T=T, alpha=alpha, T_ref=T_ref)
    sig, eps = element_stress_strain(mesh, (ux, uy), E, nu, plane=plane,
                                     T=T, alpha=alpha, T_ref=T_ref)
    vm = von_mises(sig)
    umag = np.hypot(ux, uy)
    return {
        "ux": ux,
        "uy": uy,
        "u_max": float(umag.max()),
        "sig_elem": sig,
        "eps_elem": eps,
        "vm_elem": vm,
        "vm_nodal": element_to_nodal(mesh, vm),
        "vm_max": float(vm.max()),
    }


def induction_thermomech(mesh, sigma, omega, J_s=0.0, nu=None,
                         k_th=1.0, rho_cp=1.0, bc_em=None, bc_th=None,
                         t_end=1.0, dt=0.01, T0=0.0, depth=1.0,
                         E=None, nu_p=0.3, alpha=0.0, plane="strain",
                         bc_el=None):
    """Full chain: induction heating -> transient heat -> thermal stress.

    Runs coupled.induction_heating, then solves thermal stress from the
    final temperature field (T_ref = T0). Requires E and alpha.
    Returns the induction_heating result dict plus the thermal_stress
    entries (ux, uy, vm_*, ...).
    """
    if E is None:
        raise ValueError("E (Young's modulus) is required")
    res = induction_heating(mesh, sigma=sigma, omega=omega, J_s=J_s, nu=nu,
                            k_th=k_th, rho_cp=rho_cp,
                            bc_em=bc_em, bc_th=bc_th,
                            t_end=t_end, dt=dt, T0=T0, depth=depth)
    T_final = res["T_history"][-1]
    mech = thermal_stress(mesh, T_final, E, nu_p, alpha, T_ref=T0,
                          plane=plane, bc=bc_el)
    res.update(mech)
    return res
