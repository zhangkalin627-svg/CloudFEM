"""Electromagnetics: 2D time-harmonic eddy currents (A-formulation).

We solve for the z-component of the magnetic vector potential::

    -div(nu grad A) + i*omega*sigma*A = J_s      in Omega

where nu = 1/mu is the reluctivity, sigma the conductivity,
omega the angular frequency and J_s the imposed source current
density (z-directed). All fields are phasors with e^{iwt} convention.

Derived quantities
------------------
- B = curl(A)  ->  Bx = dA/dy, By = -dA/dx   (in-plane flux density)
- Induced current density (phasor): J_e = -i*omega*sigma*A
- Time-averaged Joule heating density: Q = 0.5*sigma*omega^2*|A|^2
  (valid where J_s = 0, i.e. inside the workpiece)

The Joule heating field Q feeds the thermal solver for
sequential EM-thermal (induction heating) coupling.
"""

import numpy as np

from ..fem.poisson import assemble, apply_dirichlet
from ..fem.poisson import _gradients
import scipy.sparse.linalg as spla

MU0 = 4e-7 * np.pi


def solve_eddy_current(mesh, nu=1.0 / MU0, sigma=0.0, omega=0.0,
                       J_s=0.0, bc=None):
    """Solve the time-harmonic A-formulation.

    Parameters
    ----------
    mesh : TriMesh
    nu : float or callable(x, y), reluctivity 1/mu.
    sigma : float or callable(x, y), conductivity [S/m].
    omega : float, angular frequency [rad/s].
    J_s : float or callable(x, y), imposed current density [A/m^2].
    bc : Dirichlet BC dict for A.

    Returns
    -------
    A : (N,) complex array, vector potential phasor at nodes.
    """
    nu_fn = nu if callable(nu) else lambda x, y: nu
    sigma_fn = sigma if callable(sigma) else lambda x, y: sigma

    def c_fn(x, y):
        return 1j * omega * sigma_fn(x, y)

    A_mat, rhs = assemble(mesh, k=nu_fn, c=c_fn, f=J_s, dtype=complex)
    if bc:
        A_mat, rhs = apply_dirichlet(A_mat, rhs, mesh, bc)
    A = spla.spsolve(A_mat, rhs)
    return np.asarray(A).ravel()


def joule_heating_density(mesh, A, sigma, omega):
    """Time-averaged Joule heating Q = 0.5*sigma*omega^2*|A|^2 per element.

    Returns (Q_elem, areas): element-wise heating [W/m^3] and areas.
    sigma may be float or callable.
    """
    sigma_fn = sigma if callable(sigma) else lambda x, y: sigma
    Q_elem = np.zeros(mesh.n_elements)
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        xc, yc = p[:, 0].mean(), p[:, 1].mean()
        s = sigma_fn(xc, yc)
        Ae = A[el]
        # average |A|^2 over the element via nodal values
        A2 = float(np.mean(np.abs(Ae) ** 2))
        Q_elem[e] = 0.5 * s * omega ** 2 * A2
    return Q_elem, mesh.element_areas()


def total_joule_power(mesh, A, sigma, omega, depth=1.0):
    """Total time-averaged Joule power [W] (= integral of Q over domain).

    depth : out-of-plane thickness [m] for the 2D model.
    """
    Q_elem, areas = joule_heating_density(mesh, A, sigma, omega)
    return float(np.sum(Q_elem * areas) * depth)


def flux_density(mesh, A):
    """Element-wise flux density B = (dA/dy, -dA/dx) [T], shape (M, 2)."""
    B = np.zeros((mesh.n_elements, 2), dtype=complex)
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        b, c, _ = _gradients(p)
        Ae = A[el]
        dAdx = np.dot(b, Ae)
        dAdy = np.dot(c, Ae)
        B[e, 0] = dAdy
        B[e, 1] = -dAdx
    return B


def skin_depth(sigma, omega, mu=MU0):
    """Analytical skin depth delta = sqrt(2/(omega*mu*sigma)) [m]."""
    return float(np.sqrt(2.0 / (omega * mu * sigma)))
