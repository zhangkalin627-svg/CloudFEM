"""Nonlinear magnetostatics: 2D A-formulation with B-H saturation.

We solve for the z-component of the magnetic vector potential::

    -div( nu(|B|^2, x, y) grad A ) = J_s      in Omega

with B = curl(A), i.e. Bx = dA/dy, By = -dA/dx (in-plane flux density).
The reluctivity nu = 1/mu may depend on the local flux density
(saturation) and on position (iron vs. air regions).

Solver
------
Two solvers are provided:

- ``'newton'`` (default): Newton's method on the nonlinear residual,
  with a backtracking line search on the residual norm. Robust for
  strong saturation (high mu_r contrast, field singularities at
  corners), where fixed-point iteration stalls or oscillates.
- ``'picard'``: fixed-point iteration on the element-wise reluctivity.
  Simple and adequate for mild nonlinearities.

Constitutive models
-------------------
- ``bh_cubic``: nu(B^2) = nu0 * (1 + alpha*B^2). Smooth and simple;
  used for the manufactured-solution convergence test.
- ``bh_saturation``: tanh knee from nu0/mu_r0 at low flux density
  to nu0 (= 1/mu0) at deep saturation, with an iron/air mask::

      nu = nu0 * (1/mu_r0 + (1 - 1/mu_r0) * s),  s = (1+tanh((|B|-B_knee)/(w*B_knee)))/2

  in iron, nu0 in air. B_knee is the middle of the saturation
  transition (typical lamination steel: mu_r0 ~ 2000, B_knee ~ 1 T);
  w (default 0.15) is the transition half-width as a fraction of
  B_knee. The flat low-field branch keeps mu_r0 intact below the knee.

A ``nu_model`` is any callable ``nu(B2, xc, yc)`` taking per-element
arrays of |B|^2 and centroid coordinates and returning ``(nu, dnu)``:
the per-element reluctivity and its derivative d(nu)/d(B^2), both
(M,) arrays. The derivative feeds the Newton Jacobian; Picard ignores
it.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .em import MU0


# ---------------------------------------------------------------------------
# constitutive models
# ---------------------------------------------------------------------------

def bh_cubic(nu0=1.0 / MU0, alpha=1.0):
    """Smooth cubic reluctivity model: nu(B^2) = nu0 * (1 + alpha*B^2).

    Monotone in |B|^2; convenient for manufactured-solution tests.
    Returns (nu, dnu/dB^2).
    """
    def nu(B2, xc, yc):
        B2 = np.asarray(B2, dtype=float)
        return nu0 * (1.0 + alpha * B2), np.full_like(B2, nu0 * alpha)
    return nu


def bh_saturation(mu_r0=2000.0, B_knee=1.0, nu0=1.0 / MU0, iron=None,
                 width=0.15):
    """Saturation model with a tanh knee, plus an iron/air mask.

    Parameters
    ----------
    mu_r0 : initial relative permeability of the iron at low flux.
    B_knee : flux density [T] at the middle of the saturation transition.
    nu0 : reluctivity of free space (and of air regions).
    iron : callable(x, y) -> bool array, True where the material is iron.
        May assume vectorised (numpy-array) inputs. None means all iron.
    width : knee transition half-width as a fraction of B_knee
        (default 0.15, i.e. most of the drop happens within
        B_knee*(1 +/- 0.3)). Below ~0.7*B_knee the permeability stays
        essentially at mu_r0, as in real lamination steel.

    In iron: nu = nu0*(1/mu_r0 + (1-1/mu_r0)*s),
        s = 1/2 * (1 + tanh((|B| - B_knee) / (width*B_knee))),
    which tends to nu0/mu_r0 for |B| -> 0 and to nu0 for |B| -> infinity.
    The flat low-field branch keeps the Picard iteration well behaved
    far from saturation.
    Returns (nu, dnu/dB^2); the derivative is regularised at B = 0.
    """
    def nu(B2, xc, yc):
        B2 = np.asarray(B2, dtype=float)
        B = np.sqrt(B2 + 1e-24)
        z = (B - B_knee) / (width * B_knee)
        t = np.tanh(z)
        s = 0.5 * (1.0 + t)
        # ds/d(B^2) = ds/dB * dB/d(B^2)
        #           = [0.5*sech^2(z)/(w*B_knee)] * [1/(2*B)]
        ds = 0.5 * (1.0 - t * t) / (width * B_knee) * (0.5 / B)
        nu_iron = nu0 * (1.0 / mu_r0 + (1.0 - 1.0 / mu_r0) * s)
        dnu_iron = nu0 * (1.0 - 1.0 / mu_r0) * ds
        if iron is None:
            return nu_iron, dnu_iron
        mask = np.asarray(iron(xc, yc), dtype=bool)
        return (np.where(mask, nu_iron, nu0),
                np.where(mask, dnu_iron, 0.0))
    return nu


# ---------------------------------------------------------------------------
# assembly with per-element reluctivity (vectorised)
# ---------------------------------------------------------------------------

def _element_operators(mesh):
    """Precompute P1 gradient data shared by every Picard iteration.

    Returns b, c (M, 3) with grad(phi_i) = (b_i, c_i) per element,
    areas (M,), centroids xc, yc (M,).
    """
    el = mesh.elements
    p = mesh.nodes[el]                       # (M, 3, 2)
    x = p[:, :, 0]
    y = p[:, :, 1]
    area2 = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
             - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    area = 0.5 * np.abs(area2)
    b = np.stack([y[:, 1] - y[:, 2],
                  y[:, 2] - y[:, 0],
                  y[:, 0] - y[:, 1]], axis=1) / area2[:, None]
    c = np.stack([x[:, 2] - x[:, 1],
                  x[:, 0] - x[:, 2],
                  x[:, 1] - x[:, 0]], axis=1) / area2[:, None]
    return b, c, area, x.mean(axis=1), y.mean(axis=1)


def _stiffness_matrix(mesh, ops, nu_elem):
    """Assemble K for -div(nu_elem grad A); nu_elem is (M,) per element."""
    el = mesh.elements
    n = mesh.n_nodes
    b, c, area, _, _ = ops
    G = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    Ke = (nu_elem * area)[:, None, None] * G          # (M, 3, 3)
    rows = np.repeat(el, 3, axis=1).ravel()
    cols = np.tile(el, (1, 3)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)),
                         shape=(n, n)).tocsr()


def _load_vector(mesh, ops, f_fn):
    """Assemble the source vector for J_s (float or vectorised callable)."""
    el = mesh.elements
    n = mesh.n_nodes
    _, _, area, xc, yc = ops
    if callable(f_fn):
        fe = np.asarray(f_fn(xc, yc), dtype=float)
    else:
        fe = np.full(len(xc), float(f_fn))
    rhs = np.zeros(n)
    np.add.at(rhs, el.ravel(), np.repeat(fe * area / 3.0, 3))
    return rhs


def _jacobian_matrix(mesh, ops, nu, dnu, A):
    """Newton Jacobian of R(A) = K(nu(A)) A - f.

    Per element (B = curl A, B^2 = Bx^2 + By^2)::

        J^e = area_e * [nu_e G^e + 2 dnu_e outer(G^e A^e, Bx_e c^e - By_e b^e)]

    from dK^e/dA^e_k = nu'(B^2_e) d(B^2_e)/dA^e_k area_e G^e with
    d(B^2_e)/dA^e_k = 2 (Bx_e c_k - By_e b_k).
    """
    el = mesh.elements
    n = mesh.n_nodes
    b, c, area, _, _ = ops
    G = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    Ae = A[el]                                            # (M, 3)
    GA = np.einsum("eij,ej->ei", G, Ae)                   # G^e A^e
    Bx = np.einsum("ei,ei->e", c, Ae)                     # dA/dy
    By = -np.einsum("ei,ei->e", b, Ae)                    # -dA/dx
    dB2 = Bx[:, None] * c - By[:, None] * b               # (M, 3)
    Je = area[:, None, None] * (
        nu[:, None, None] * G
        + (2.0 * dnu)[:, None, None] * GA[:, :, None] * dB2[:, None, :])
    rows = np.repeat(el, 3, axis=1).ravel()
    cols = np.tile(el, (1, 3)).ravel()
    return sp.coo_matrix((Je.ravel(), (rows, cols)),
                         shape=(n, n)).tocsr()


def _dirichlet_split(mesh, bc):
    """Split DOFs into fixed/free once; reused by every Picard iteration.

    Returns (fixed, free, v_fixed): sorted fixed node indices, free node
    indices, and the prescribed values at the fixed nodes. Avoids the
    per-iteration matrix surgery of apply_dirichlet.
    """
    n = mesh.n_nodes
    fixed_vals = {}
    for name, value in (bc or {}).items():
        ids = np.arange(n) if name == "all" else mesh.boundaries[name]
        v_fn = value if callable(value) else (lambda x, y, v=value: v)
        for i in ids:
            x, y = mesh.nodes[int(i)]
            fixed_vals[int(i)] = v_fn(x, y)
    fixed = np.array(sorted(fixed_vals), dtype=int)
    free = np.setdiff1d(np.arange(n), fixed, assume_unique=True)
    v = np.array([fixed_vals[i] for i in fixed], dtype=float)
    return fixed, free, v


def _element_flux_density(ops, el, A):
    """|B|^2 per element from the nodal potential (B = curl A)."""
    b, c, _, _, _ = ops
    Ae = A[el]                                       # (M, 3)
    Bx = np.einsum("ei,ei->e", c, Ae)                 # dA/dy
    By = -np.einsum("ei,ei->e", b, Ae)                # -dA/dx
    return Bx * Bx + By * By, Bx, By


# ---------------------------------------------------------------------------
# Picard solver
# ---------------------------------------------------------------------------

def solve_nonlinear_magnetostatics(mesh, nu_model, J_s=0.0, bc=None,
                                   tol=1e-8, max_iter=50, A_init=None,
                                   verbose=False, method="newton",
                                   relax=1.0):
    """Solve -div(nu(|B|^2,x,y) grad A) = J_s.

    Parameters
    ----------
    mesh : TriMesh
    nu_model : callable (B2, xc, yc) -> (nu, dnu/dB^2), per-element arrays.
    J_s : float or vectorised callable(x, y), imposed current density.
    bc : Dirichlet BC dict for A (boundary name or 'all' -> value or
        callable(x, y), same convention as fem.poisson.apply_dirichlet).
    tol : convergence tolerance (relative residual for Newton, relative
        update for Picard).
    max_iter : cap on nonlinear iterations (Newton needs far fewer).
    A_init : optional starting field; default is the linear solve with
        nu evaluated at |B| = 0.
    verbose : print per-iteration residuals.
    method : 'newton' (default; line search, robust for strong
        saturation) or 'picard' (fixed-point; only for mild
        nonlinearities).
    relax : Picard relaxation factor in (0, 1]; ignored by Newton.

    Returns
    -------
    A : (N,) nodal vector potential.
    info : dict with 'iterations', 'residual' (final relative residual
        or update) and 'converged' (bool).
    """
    el = mesh.elements
    n = mesh.n_nodes
    ops = _element_operators(mesh)
    _, _, _, xc, yc = ops
    fixed, free, v_fixed = _dirichlet_split(mesh, bc)
    rhs = _load_vector(mesh, ops, J_s)

    def initial_guess():
        nu0, _ = nu_model(np.zeros(len(xc)), xc, yc)
        K = _stiffness_matrix(mesh, ops, nu0)
        return _solve_free(K, rhs, mesh, fixed, free, v_fixed)

    if A_init is None:
        A = initial_guess()
    else:
        A = np.asarray(A_init, dtype=float).ravel().copy()
        if len(fixed):
            A[fixed] = v_fixed

    if method == "newton":
        return _newton_solve(mesh, ops, nu_model, rhs, fixed, free,
                             v_fixed, A, tol, max_iter, verbose)
    elif method == "picard":
        return _picard_solve(mesh, ops, nu_model, rhs, fixed, free,
                             v_fixed, A, tol, max_iter, relax, verbose)
    raise ValueError(f"unknown method {method!r}; want 'newton'/'picard'")


def _solve_free(K, rhs, mesh, fixed, free, v_fixed):
    """Solve K A = rhs for the free DOFs (precomputed Dirichlet split)."""
    n = mesh.n_nodes
    Kf = K[free]                                  # csr row slice
    K_ff = Kf[:, free].tocsr()
    rhs_f = rhs[free]
    if len(fixed):
        rhs_f = rhs_f - Kf[:, fixed] @ v_fixed
    A = np.zeros(n)
    A[free] = np.asarray(spla.spsolve(K_ff, rhs_f)).ravel()
    if len(fixed):
        A[fixed] = v_fixed
    return A


def _fields(mesh, ops, nu_model, A):
    """Residual R = K(nu(A)) A - f plus the fields Newton needs."""
    el = mesh.elements
    _, _, _, xc, yc = ops
    B2, Bx, By = _element_flux_density(ops, el, A)
    nu, dnu = nu_model(B2, xc, yc)
    K = _stiffness_matrix(mesh, ops, nu)
    return K, B2, Bx, By, nu, dnu


def _newton_solve(mesh, ops, nu_model, rhs, fixed, free, v_fixed, A,
                  tol, max_iter, verbose):
    """Newton's method with backtracking line search on ||R||."""
    el = mesh.elements

    def residual(A):
        K, B2, Bx, By, nu, dnu = _fields(mesh, ops, nu_model, A)
        return K @ A - rhs, (B2, Bx, By, nu, dnu)

    R, fields = residual(A)
    rf_norm = float(np.linalg.norm(R[free]))
    scale = float(np.linalg.norm(rhs[free])) or rf_norm or 1.0

    converged = False
    rel = rf_norm / scale
    it = 0
    for it in range(1, max_iter + 1):
        if rel < tol:
            converged = True
            break
        B2, Bx, By, nu, dnu = fields
        J = _jacobian_matrix(mesh, ops, nu, dnu, A)
        Jf = J[free]
        J_ff = Jf[:, free].tocsr()
        delta = np.zeros_like(A)
        delta[free] = np.asarray(spla.spsolve(J_ff, -R[free])).ravel()
        # backtracking line search on the free residual norm
        alpha, R_trial, fields_trial = 1.0, None, None
        for _ in range(8):
            A_trial = A + alpha * delta
            R_t, f_t = residual(A_trial)
            if float(np.linalg.norm(R_t[free])) < rf_norm:
                R_trial, fields_trial = R_t, f_t
                break
            alpha *= 0.5
        else:
            # no decrease found; take a small step rather than stalling
            alpha = 1.0 / 256.0
            A_trial = A + alpha * delta
            R_trial, fields_trial = residual(A_trial)
        A = A_trial
        if len(fixed):
            A[fixed] = v_fixed
        R, fields = R_trial, fields_trial
        rf_norm = float(np.linalg.norm(R[free]))
        rel = rf_norm / scale
        if verbose:
            print(f"  newton {it:3d}: rel residual = {rel:.3e} "
                  f"(alpha = {alpha:.3f})")

    info = {"iterations": it, "residual": float(rel),
            "converged": bool(converged)}
    return A, info


def _picard_solve(mesh, ops, nu_model, rhs, fixed, free, v_fixed, A,
                  tol, max_iter, relax, verbose):
    """Fixed-point iteration on the element-wise reluctivity."""
    el = mesh.elements
    _, _, _, xc, yc = ops
    converged = False
    rel = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        B2, _, _ = _element_flux_density(ops, el, A)
        nu, _ = nu_model(B2, xc, yc)
        K = _stiffness_matrix(mesh, ops, nu)
        A_new = _solve_free(K, rhs, mesh, fixed, free, v_fixed)
        if relax != 1.0:
            A_new = relax * A_new + (1.0 - relax) * A
            if len(fixed):
                A_new[fixed] = v_fixed
        dA = A_new - A
        nrm = np.linalg.norm(A_new)
        rel = np.linalg.norm(dA) / nrm if nrm > 0 else np.linalg.norm(dA)
        A = A_new
        if verbose:
            print(f"  picard {it:3d}: rel update = {rel:.3e}")
        if rel < tol:
            converged = True
            break

    info = {"iterations": it, "residual": float(rel),
            "converged": bool(converged)}
    return A, info


def flux_density(mesh, A):
    """Element-wise flux density B = (dA/dy, -dA/dx) [T], shape (M, 2)."""
    ops = _element_operators(mesh)
    _, Bx, By = _element_flux_density(ops, mesh.elements, A)
    return np.stack([Bx, By], axis=1)
