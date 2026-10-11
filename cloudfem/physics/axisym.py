"""Axisymmetric physics (r-z plane, x=r, y=z).

Electromagnetics: we solve for utilde = r * A_phi. The weak form on the
r-z half-plane (dV = 2*pi*r dr dz folded in) is::

    int (2*pi*nu/r) grad(utilde).grad(v) + (2*pi*i*omega*sigma/r) utilde*v
        = int 2*pi*J_phi*v

i.e. k = 2*pi*nu/r, c = i*omega*2*pi*sigma/r, f = 2*pi*J_phi.
The 1/r weight is integrable (test functions vanish on the axis) and is
evaluated at element centroids, which never sit exactly on r=0.

Postprocessing (per element, at centroid radius rc > 0):
  A_phi = utilde / r
  B_r = -(1/r) d(utilde)/dz,  B_z = (1/r) d(utilde)/dr
  Q = 0.5*sigma*omega^2*|utilde|^2 / r^2   (time-averaged Joule heating)

Heat conduction: standard weak form with 2*pi*r weight folded into
k -> 2*pi*r*k, rho_cp -> 2*pi*r*rho_cp, source -> 2*pi*r*Q.

Boundary condition on the axis (r=0): utilde = 0.
"""

import numpy as np

from ..fem.poisson import solve_poisson, _gradients
from ..fem.timestep import implicit_euler_heat, mass_matrix
from ..fem.timestep import source_load_vector as _source_load_vector
from .em import MU0

TWO_PI = 2.0 * np.pi


def _as_fn(q):
    return q if callable(q) else (lambda x, y, q=q: q)


def solve_eddy_current_axi(mesh, nu=1.0 / MU0, sigma=0.0, omega=0.0,
                           J_phi=0.0, bc=None, neumann=None):
    """Axisymmetric time-harmonic eddy currents.

    mesh : TriMesh on (r, z), x=r >= 0.
    Returns utilde = r*A_phi (complex nodal array).
    """
    nu_f, sigma_f, J_f = _as_fn(nu), _as_fn(sigma), _as_fn(J_phi)

    def k_fn(x, y):
        return TWO_PI * nu_f(x, y) / max(x, 1e-12)

    def c_fn(x, y):
        return 1j * omega * TWO_PI * sigma_f(x, y) / max(x, 1e-12)

    def f_fn(x, y):
        return TWO_PI * J_f(x, y)

    return solve_poisson(mesh, k=k_fn, c=c_fn, f=f_fn, bc=bc,
                         dtype=complex, neumann=neumann)


def a_phi_nodal(mesh, utilde):
    """Recover A_phi = utilde / r at nodes (0 on the axis by symmetry)."""
    r = mesh.nodes[:, 0]
    out = np.zeros_like(utilde)
    m = r > 0
    out[m] = utilde[m] / r[m]
    return out


def joule_heating_density_axi(mesh, utilde, sigma, omega):
    """Per-element Q = 0.5*sigma*omega^2*|utilde|^2/r^2 at centroid radius."""
    sigma_f = _as_fn(sigma)
    Q_elem = np.zeros(mesh.n_elements)
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        rc, zc = p[:, 0].mean(), p[:, 1].mean()
        s = sigma_f(rc, zc)
        u2 = float(np.mean(np.abs(utilde[el]) ** 2))
        Q_elem[e] = 0.5 * s * omega ** 2 * u2 / max(rc, 1e-12) ** 2 \
            if rc > 0 else 0.0
    return Q_elem, mesh.element_areas()


def total_joule_power_axi(mesh, utilde, sigma, omega):
    """Total Joule power [W]; element volume = 2*pi*r*area."""
    Q_elem, areas = joule_heating_density_axi(mesh, utilde, sigma, omega)
    rc = np.array([mesh.nodes[el][:, 0].mean() for el in mesh.elements])
    return float(np.sum(Q_elem * TWO_PI * rc * areas))


def flux_density_axi(mesh, utilde):
    """Element-wise (B_r, B_z) from utilde, shape (M, 2), complex."""
    B = np.zeros((mesh.n_elements, 2), dtype=complex)
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        rc = p[:, 0].mean()
        b, c, _ = _gradients(p)
        ue = utilde[el]
        dudr = np.dot(b, ue)
        dudz = np.dot(c, ue)
        if rc > 0:
            B[e, 0] = -dudz / rc
            B[e, 1] = dudr / rc
    return B


def source_load_vector_axi(mesh, q_nodal):
    """Consistent load vector with 2*pi*r weight from a nodal source field."""
    n = mesh.n_nodes
    rhs = np.zeros(n)
    for el in mesh.elements:
        p = mesh.nodes[el]
        _, _, area = _gradients(p)
        rc = p[:, 0].mean()
        qe = q_nodal[el].mean()
        w = TWO_PI * rc * area / 3.0
        for i in el:
            rhs[i] += qe * w
    return rhs


def transient_heat_axi(mesh, k=1.0, rho_cp=1.0, Q=0.0, bc=None,
                       t_end=1.0, dt=0.01, T0=0.0):
    """Axisymmetric transient heat conduction (implicit Euler).

    Q may be float, callable f(r, z), or a nodal array.
    """
    k_f, rc_f = _as_fn(k), _as_fn(rho_cp)
    k_axi = lambda x, y: TWO_PI * x * k_f(x, y)          # noqa: E731
    rc_axi = lambda x, y: TWO_PI * x * rc_f(x, y)        # noqa: E731

    if isinstance(Q, np.ndarray):
        rhs_q = source_load_vector_axi(mesh, Q)
    else:
        q_f = _as_fn(Q)

        def f_axi(x, y):
            return TWO_PI * x * q_f(x, y)

        from ..fem.poisson import assemble
        _, rhs_q = assemble(mesh, k=0.0, f=f_axi)
    return implicit_euler_heat(mesh, k_axi, rc_axi, rhs_q, bc, t_end, dt, T0)
