"""Transient heat conduction on tetrahedral P1 meshes (v1.0).

Semi-discrete form of::

    rho_cp * dT/dt = div(k grad T) + f        in Omega

    C dT/dt + K T = F

with C the *consistent* heat-capacity matrix (exact P1 integrals
``int phi_i phi_j dV = V/20 * (1 + delta_ij)`` per tetrahedron,
scaled by rho_cp), K the stiffness matrix from
:func:`cloudfem.fem.tet_poisson.assemble` (plus Robin boundary
contributions) and F the load vector (volume source, Neumann fluxes,
Robin ambient terms).

Time integration is the one-parameter theta method,

    (C + theta*dt*K) T^{n+1} = (C - (1-theta)*dt*K) T^n + dt * F,

with ``method="be"`` (theta = 1, implicit Euler) or ``method="cn"``
(theta = 1/2, Crank-Nicolson). Both are A-stable, hence
unconditionally stable for this parabolic problem; implicit Euler is
additionally L-stable (stiff modes are damped, not just bounded),
while Crank-Nicolson is second order in time but can ring on
under-resolved initial transients. The system matrix is constant in
time, so it is LU-factorised once per run and reused at every step.

Boundary conditions (all constant in time):

- Dirichlet: ``bc`` dict as in tet_poisson (name -> value or
  callable(x, y, z)). The initial field is overwritten by the BC
  values at constrained nodes.
- Neumann: ``neumann`` dict name -> inward heat flux q (W/m^2),
  float or callable(x, y, z); positive q heats the body. Adds
  ``int q * phi ds`` to F, integrated per boundary face with the
  centroid value (each face node receives q * A_face / 3).
- Robin (convection): ``robin`` dict name -> (h, T_inf); h and
  T_inf floats or callables(x, y, z). Models the outward flux
  ``h * (T - T_inf)``: adds the exact P1 face mass
  ``h * A_face/12 * (1 + delta_ij)`` to K and ``h * T_inf * A_face/3``
  to each face node's load. A boundary with neither condition is
  insulated (zero flux) -- the natural condition.

Assumptions: k, rho_cp and f may vary in space (floats or
callables(x, y, z), evaluated at element centroids, as in
tet_poisson.assemble) but not in time; all boundary data are
time-constant.

Maximum principle: with the consistent capacity matrix the discrete
maximum principle is not unconditional. For smooth, boundary-
compatible initial data it holds exactly once dt >= h^2/(6*alpha)
(measured in tests/test_transient3d.py) and degrades gracefully
below that threshold (a few K of undershoot on a 100 K range). For
boundary-incompatible data (e.g. a quench: uniform hot body, cold
surface at t = 0+) a localised overshoot of a few percent of the
temperature jump appears in the surface layer at the first instants,
essentially independent of dt; the total heat remains monotone and
exactly balanced throughout. A lumped capacity matrix would be
needed for a strict guarantee. The consistent matrix is used
because it conserves the total heat ``ones^T C T`` exactly under
adiabatic conditions and matches the exact-integral philosophy of
the v0.9 mass matrices.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .tet_poisson import assemble, apply_dirichlet


def assemble_mass_tet(mesh, rho_cp=1.0):
    """Consistent P1 heat-capacity matrix int rho_cp phi_i phi_j dV.

    rho_cp : float or callable(x, y, z) (element centroid value).
    Returns csr_matrix (N, N). Row sums are rho_cp * int phi_i dV,
    so ones^T C T is the total heat content for a nodal field T.
    """
    n = mesh.n_nodes
    rc_fn = rho_cp if callable(rho_cp) else (lambda x, y, z, v=rho_cp: v)
    # int phi_i phi_j dV = V/20 * (1 + delta_ij) on a tetrahedron
    base = (np.ones((4, 4)) + np.eye(4)) / 20.0
    vols = mesh.element_volumes()
    rows, cols, data = [], [], []
    for e, el in enumerate(mesh.elements):
        p = mesh.nodes[el]
        xc, yc, zc = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()
        me = rc_fn(xc, yc, zc) * vols[e] * base
        for i in range(4):
            for j in range(4):
                rows.append(el[i])
                cols.append(el[j])
                data.append(me[i, j])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _boundary_faces(mesh, name):
    """Exterior triangular faces on the named boundary.

    Same algorithm as elastic3d.boundary_faces (faces belonging to
    exactly one tetrahedron with all 3 nodes in the named set);
    duplicated here to keep fem/ independent of physics/.
    Returns (F, 3) int array.
    """
    ids = np.asarray(mesh.boundaries[name])
    idset = set(int(i) for i in ids)
    counts = {}
    faces = {}
    for el in mesh.elements:
        a, b, c, d = (int(el[0]), int(el[1]), int(el[2]), int(el[3]))
        for tri in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(tri))
            counts[key] = counts.get(key, 0) + 1
            if key not in faces:
                faces[key] = tri
    out = [faces[key] for key, cnt in counts.items()
           if cnt == 1 and all(n in idset for n in key)]
    if not out:
        return np.zeros((0, 3), dtype=int)
    return np.array(out, dtype=int)


def _face_geometry(mesh, tri):
    """Centroid and area of one triangular face (3 node indices)."""
    p = mesh.nodes[tri]
    cen = p.mean(axis=0)
    cr = np.cross(p[1] - p[0], p[2] - p[0])
    return cen, 0.5 * float(np.linalg.norm(cr))


def _scalar_fn(v):
    """Normalise a float-or-callable(x, y, z) spec to a callable."""
    return v if callable(v) else (lambda x, y, z, v=v: v)


def assemble_boundary_terms(mesh, neumann=None, robin=None):
    """Neumann / Robin boundary contributions.

    Returns (K_add, F_add): K_add is a csr_matrix (Robin face mass
    terms only) and F_add the extra load vector (Neumann fluxes plus
    Robin ambient terms). See the module docstring for conventions.
    """
    n = mesh.n_nodes
    F_add = np.zeros(n)
    rows, cols, data = [], [], []
    # exact P1 integrals on a triangle: int phi_i phi_j dA
    face_mass = (np.ones((3, 3)) + np.eye(3)) / 12.0

    if neumann:
        for name, q in neumann.items():
            q_fn = _scalar_fn(q)
            for tri in _boundary_faces(mesh, name):
                (xc, yc, zc), area = _face_geometry(mesh, tri)
                F_add[tri] += q_fn(xc, yc, zc) * area / 3.0

    if robin:
        for name, (h, T_inf) in robin.items():
            h_fn = _scalar_fn(h)
            Tinf_fn = _scalar_fn(T_inf)
            for tri in _boundary_faces(mesh, name):
                (xc, yc, zc), area = _face_geometry(mesh, tri)
                hv = h_fn(xc, yc, zc)
                ke = hv * area * face_mass
                for i in range(3):
                    for j in range(3):
                        rows.append(tri[i])
                        cols.append(tri[j])
                        data.append(ke[i, j])
                F_add[tri] += hv * Tinf_fn(xc, yc, zc) * area / 3.0

    K_add = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return K_add, F_add


def _dirichlet_values(mesh, bc):
    """Resolve a tet_poisson-style bc dict to {node_index: value}."""
    fixed = {}
    if not bc:
        return fixed
    for name, value in bc.items():
        if name == "all":
            ids = np.arange(mesh.n_nodes)
        else:
            ids = mesh.boundaries[name]
        v_fn = _scalar_fn(value)
        for i in ids:
            i = int(i)
            x, y, z = mesh.nodes[i]
            fixed[i] = v_fn(x, y, z)
    return fixed


_THETA = {"be": 1.0, "implicit_euler": 1.0,
          "cn": 0.5, "crank_nicolson": 0.5}


def transient_heat_tet(mesh, k=1.0, rho_cp=1.0, f=0.0, bc=None,
                       neumann=None, robin=None, t_end=1.0, dt=0.1,
                       T0=0.0, method="cn"):
    """Solve transient heat conduction on a TetMesh.

    Parameters
    ----------
    mesh : TetMesh
    k, rho_cp, f : floats or callables(x, y, z); thermal conductivity,
        volumetric heat capacity, and volumetric heat source.
    bc : None or Dirichlet dict (see module docstring), time-constant.
    neumann, robin : None or boundary-condition dicts (module docstring).
    t_end, dt : final time and nominal step. The step actually used is
        t_end / ceil(t_end / dt), so the last step lands exactly on
        t_end.
    T0 : scalar or (N,) initial temperature field.
    method : "be" (implicit Euler, theta = 1) or "cn"
        (Crank-Nicolson, theta = 1/2); long names accepted too.

    Returns
    -------
    times : (S + 1,) array including t = 0.
    T_history : (S + 1, N) nodal temperature at each output time.
    """
    if method not in _THETA:
        raise ValueError(f"unknown method {method!r}; "
                         f"use one of {sorted(_THETA)}")
    theta = _THETA[method]
    n = mesh.n_nodes

    K, F = assemble(mesh, k=k, f=f)
    K_add, F_add = assemble_boundary_terms(mesh, neumann=neumann,
                                           robin=robin)
    K = K + K_add
    F = F + F_add
    C = assemble_mass_tet(mesh, rho_cp=rho_cp)

    n_steps = int(np.ceil(t_end / dt - 1e-12))
    if n_steps < 1:
        raise ValueError("t_end must be positive")
    h = t_end / n_steps

    A_sys = C + theta * h * K
    R_sys = C - (1.0 - theta) * h * K

    fixed = _dirichlet_values(mesh, bc)
    if fixed:
        # Factorise the BC-modified matrix once; the per-step RHS
        # correction rhs - A_sys[:, fixed] @ v reproduces exactly the
        # elimination that tet_poisson.apply_dirichlet performs.
        A_bc, _ = apply_dirichlet(A_sys, np.zeros(n), mesh, bc)
        fixed_idx = np.array(sorted(fixed), dtype=int)
        fixed_val = np.array([fixed[i] for i in fixed_idx])
        cols = A_sys.tocsr()[:, fixed_idx]
    else:
        A_bc = A_sys
        fixed_idx = fixed_val = cols = None
    solve = spla.factorized(A_bc.tocsc())

    T = (np.full(n, T0, dtype=float) if np.isscalar(T0)
         else np.asarray(T0, dtype=float).copy())
    if fixed_idx is not None:
        T[fixed_idx] = fixed_val

    times = [0.0]
    hist = [T.copy()]
    for _ in range(n_steps):
        rhs = R_sys @ T + h * F
        if fixed_idx is not None:
            rhs = rhs - cols @ fixed_val
            rhs[fixed_idx] = fixed_val
        T = np.asarray(solve(rhs)).ravel()
        if fixed_idx is not None:
            T[fixed_idx] = fixed_val
        times.append(times[-1] + h)
        hist.append(T.copy())
    times[-1] = t_end  # exact landing (h * n_steps may round)
    return np.array(times), np.array(hist)
