"""Structural dynamics in 3D: mass matrices and modal analysis.

Extends the static tet-P1 elasticity solver (elastic3d) to free
vibration. The undamped eigenproblem is::

    K phi = omega^2 M phi,     f = omega / (2 pi)

with K the stiffness matrix from
:func:`cloudfem.physics.elastic3d.assemble_elasticity_3d` and M one of
two mass matrices assembled here on the same interleaved DOF layout
(DOF of component c of node i is 3*i + c):

- consistent mass: exact P1 integrals
  ``int phi_i phi_j dV = V/20 * (1 + delta_ij)`` per tetrahedron, so
  the element matrix couples the 4 nodes of each component block;
- lumped (row-sum) mass: diagonal, each node of an element receives
  ``rho * V / 4`` per component. For P1 tetrahedra the row sum of the
  consistent matrix equals this value, so both matrices conserve the
  total mass ``rho * V`` exactly.

The eigenproblem is solved on the free DOFs (displacement BCs removed)
with ``scipy.sparse.linalg.eigsh`` in shift-invert mode. The shift is
placed slightly below zero (``sigma = -1e-8`` times a Rayleigh-scale
estimate), which keeps K - sigma*M positive definite even for
unconstrained bodies, whose six rigid-body modes then appear as the
first six eigenvalues omega^2 ~ 0.

Assumptions: linear elasticity, small strains, no damping, constant
density. Mode shapes are returned M-orthonormal (phi^T M phi = I on
the full DOF vector, constrained entries zero).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .elastic3d import assemble_elasticity_3d, _dof_map


def assemble_mass_3d(mesh, rho, lumped=False):
    """Assemble the 3N x 3N mass matrix for density ``rho``.

    Parameters
    ----------
    mesh : TetMesh
    rho : float, mass density (kg/m^3), constant over the body.
    lumped : bool. False -> consistent mass matrix (exact P1
        integrals); True -> diagonal row-sum lumped matrix.

    Returns
    -------
    M : csr_matrix (3N, 3N), dof order ux0, uy0, uz0, ux1, ...
    """
    n3 = 3 * mesh.n_nodes
    vols = mesh.element_volumes()
    if lumped:
        diag = np.zeros(n3)
        for e, el in enumerate(mesh.elements):
            diag[_dof_map(el)] += rho * vols[e] / 4.0
        return sp.diags(diag, format="csr")

    rows, cols, data = [], [], []
    # int phi_i phi_j dV = V/20 * (1 + delta_ij) on a tetrahedron
    base = (np.ones((4, 4)) + np.eye(4)) / 20.0
    for e, el in enumerate(mesh.elements):
        me = rho * vols[e] * base  # (4, 4) per component block
        dofs = _dof_map(el)
        for c in range(3):
            dc = dofs[c::3]
            for i in range(4):
                for j in range(4):
                    rows.append(dc[i])
                    cols.append(dc[j])
                    data.append(me[i, j])
    return sp.coo_matrix((data, (rows, cols)), shape=(n3, n3)).tocsr()


def _fixed_dofs(mesh, bc):
    """Fixed DOF index set for a Dirichlet bc dict.

    Mirrors the semantics of elastic3d.apply_dirichlet_vec3: ``bc``
    maps a boundary name (or 'all') to (ux, uy, uz); a component that
    is not None fixes that DOF (the prescribed value is irrelevant for
    the homogeneous eigenproblem).
    """
    fixed = set()
    for name, vals in bc.items():
        if name == "all":
            ids = np.arange(mesh.n_nodes)
        else:
            ids = np.asarray(mesh.boundaries[name])
        for comp, v in enumerate(vals):
            if v is None:
                continue
            for i in ids:
                fixed.add(3 * int(i) + comp)
    return np.array(sorted(fixed), dtype=int)


def solve_modes_3d(mesh, E, nu, rho, bc=None, n_modes=6, lumped=False):
    """Lowest ``n_modes`` natural frequencies and mode shapes.

    Parameters
    ----------
    mesh : TetMesh
    E, nu : Young's modulus, Poisson's ratio.
    rho : mass density (kg/m^3).
    bc : None or dict like elastic3d's ``bc``; listed DOFs are removed
        (homogeneous Dirichlet). None -> free-free body (the first six
        modes are the rigid-body modes with f ~ 0).
    n_modes : number of eigenpairs to compute.
    lumped : use the lumped mass matrix instead of the consistent one.

    Returns
    -------
    freqs : (n_modes,) array of natural frequencies in Hz, ascending.
    modes : (3N, n_modes) array; column j is the mode shape belonging
        to freqs[j], M-orthonormal, zero at constrained DOFs.
    """
    K, _ = assemble_elasticity_3d(mesh, E, nu)
    M = assemble_mass_3d(mesh, rho, lumped=lumped)
    n3 = 3 * mesh.n_nodes

    fixed = _fixed_dofs(mesh, bc) if bc else np.zeros(0, dtype=int)
    free = np.ones(n3, dtype=bool)
    free[fixed] = False
    free_idx = np.where(free)[0]
    n_free = len(free_idx)
    if not 0 < n_modes < n_free:
        raise ValueError(f"n_modes={n_modes} must be in [1, {n_free})")

    Kf = K[free][:, free].tocsc()
    Mf = M[free][:, free].tocsc()

    # shift-invert just below zero: K - sigma*M stays positive
    # definite even when K alone is singular (free-free body), and the
    # eigenvalues nearest the shift (the lowest omega^2) are found by
    # the largest-magnitude transformed eigenvalues.
    scale = float(np.mean(Kf.diagonal()) / np.mean(Mf.diagonal()))
    sigma = -1e-8 * scale
    ncv = min(n_free, max(2 * n_modes + 2, 24))
    vals, vecs = spla.eigsh(Kf, M=Mf, k=n_modes, sigma=sigma,
                            which="LM", mode="normal",
                            tol=1e-10, maxiter=10000, ncv=ncv)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # M-orthonormalise explicitly (contract of this function)
    for j in range(n_modes):
        nrm = np.sqrt(vecs[:, j] @ (Mf @ vecs[:, j]))
        vecs[:, j] /= nrm

    modes = np.zeros((n3, n_modes))
    modes[free_idx, :] = vecs
    freqs = np.sqrt(np.clip(vals, 0.0, None)) / (2.0 * np.pi)
    return freqs, modes
