"""Validation of the 3D structural dynamics module (mass + modal).

All reference values are derived independently of the solver under
test (closed-form integrals, rigid-body theory, Euler-Bernoulli beam
theory, 1D wave equation):

1. Single-tetrahedron consistent mass entries vs 4-point Gauss
   quadrature of int phi_i phi_j dV (the barycentric rule with
   a = 0.5854101966249685, b = 0.1381966011250105, weights V/4 is
   exact for degree-2 polynomials, hence for P1 products phi_i*phi_j).
   This checks the closed form V/20*(1 + delta_ij) against a numerical
   integration that does not use it.
2. Total mass conservation: for a rigid translation e_c (unit vector
   on all DOFs of component c), e_c^T M e_c = rho * V_total, for both
   the consistent and the lumped matrix and all three components.
3. Rigid-body modes of a free cube: the first six eigenvalues from the
   Rayleigh quotient phi^T K phi (modes are M-normalised) must satisfy
   |lambda| < 1e-6 * lambda_7, lambda_7 > 0, and the mode shapes must
   be M-orthonormal, |Phi^T M Phi - I| ~ 0.
4. Cantilever beam, first bending frequency vs Euler-Bernoulli
   f1 = (beta1^2 / (2 pi L^2)) * sqrt(E I / (rho A)), beta1*L =
   1.8751040687. P1 constant-strain tetrahedra are overly stiff in
   bending, so the FEM value must lie ABOVE the beam-theory value;
   on the 40x5x5 mesh the measured excess is ~7 % (mesh sweep during
   development: +29 % on 20x2x2, +17 % on 24x3x3, +10 % on 32x4x4 --
   monotone convergence from above). Euler-Bernoulli also neglects
   shear deformation, a ~1 % effect at L/h = 10 acting the other way.
5. Fixed-free rod, first longitudinal frequency vs the 1D wave result
   f1 = (1 / (4L)) * sqrt(E / rho). The longitudinal mode is identified
   as the first mode whose displacement is > 90 % x-directed (L2 share
   of the ux component). The FEM end face is fixed in all three
   components (the analytical rod allows free lateral contraction),
   which perturbs f1 only weakly for a slender rod.
"""

import numpy as np

from cloudfem.mesh.tet_mesh import TetMesh, box_mesh
from cloudfem.physics.elastic3d import assemble_elasticity_3d
from cloudfem.physics.modal3d import assemble_mass_3d, solve_modes_3d

E_STEEL = 200e9
NU_STEEL = 0.3
RHO_STEEL = 7850.0


def test_element_mass_quadrature():
    """Test 1: one tet, consistent mass vs independent Gauss rule."""
    p = np.array([[0.0, 0.0, 0.0],
                  [1.3, 0.1, 0.2],
                  [0.2, 1.1, 0.3],
                  [0.1, 0.2, 0.9]])
    mesh = TetMesh(p, np.array([[0, 1, 2, 3]]))
    M = assemble_mass_3d(mesh, RHO_STEEL, lumped=False).toarray()
    Ml = assemble_mass_3d(mesh, RHO_STEEL, lumped=True).toarray()
    V = mesh.element_volumes()[0]

    # 4-point Gauss rule in barycentric coordinates (exact, degree 2)
    a = 0.5854101966249685
    b = 0.1381966011250105
    lam = np.array([[a, b, b, b],
                    [b, a, b, b],
                    [b, b, a, b],
                    [b, b, b, a]])
    integ = (lam.T @ lam) * (V / 4.0)  # int phi_i phi_j dV, (4, 4)

    err = 0.0
    for c in range(3):
        for i in range(4):
            for j in range(4):
                ref = RHO_STEEL * integ[i, j]
                err = max(err, abs(M[3 * i + c, 3 * j + c] - ref))
    # off-block entries (different components) must vanish
    err_off = abs(M[0, 1]) + abs(M[0, 2]) + abs(M[1, 2])
    # lumped: rho * V / 4 on each diagonal, zero elsewhere
    err_l = np.abs(Ml - np.diag(np.full(12, RHO_STEEL * V / 4.0))).max()
    scale = RHO_STEEL * V
    return err / scale, err_off / scale, err_l / scale


def test_mass_conservation():
    """Test 2: e_c^T M e_c = rho * V for both matrices, all comps."""
    mesh = box_mesh(3, 3, 3)
    V = mesh.element_volumes().sum()
    errs = []
    for lumped in (False, True):
        M = assemble_mass_3d(mesh, RHO_STEEL, lumped=lumped)
        for c in range(3):
            ec = np.zeros(3 * mesh.n_nodes)
            ec[c::3] = 1.0
            tot = ec @ (M @ ec)
            errs.append(abs(tot - RHO_STEEL * V) / (RHO_STEEL * V))
    return max(errs)


def test_rigid_body_modes():
    """Test 3: free cube -- six rigid modes, then flexible ones."""
    mesh = box_mesh(3, 3, 3)
    freqs, modes = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                                  bc=None, n_modes=8)
    K, _ = assemble_elasticity_3d(mesh, E_STEEL, NU_STEEL)
    M = assemble_mass_3d(mesh, RHO_STEEL)
    # M-normalised modes => Rayleigh quotient is just phi^T K phi
    lam = np.array([modes[:, j] @ (K @ modes[:, j]) for j in range(8)])
    ratio = np.abs(lam[:6]).max() / lam[6]
    G = modes.T @ (M @ modes)
    err_orth = np.abs(G - np.eye(8)).max()
    return freqs, lam, ratio, err_orth


def test_cantilever_beam():
    """Test 4: cantilever first bending frequency vs Euler-Bernoulli."""
    L, h = 1.0, 0.1
    A = h * h
    I = h * h ** 3 / 12.0
    f_eb = 1.8751040687 ** 2 / (2.0 * np.pi * L ** 2) \
        * np.sqrt(E_STEEL * I / (RHO_STEEL * A))
    mesh = box_mesh(40, 5, 5, xlim=(0, L), ylim=(0, h), zlim=(0, h))
    bc = {"xmin": (0.0, 0.0, 0.0)}
    fc, _ = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL, bc=bc,
                           n_modes=2)
    fl, _ = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL, bc=bc,
                           n_modes=2, lumped=True)
    return f_eb, fc[0], fl[0]


def test_rod_longitudinal():
    """Test 5: fixed-free rod, first longitudinal frequency."""
    L, h = 1.0, 0.1
    f_exact = 0.25 / L * np.sqrt(E_STEEL / RHO_STEEL)
    mesh = box_mesh(20, 2, 2, xlim=(0, L), ylim=(0, h), zlim=(0, h))
    freqs, modes = solve_modes_3d(mesh, E_STEEL, NU_STEEL, RHO_STEEL,
                                  bc={"xmin": (0.0, 0.0, 0.0)},
                                  n_modes=8)
    f_long = None
    for j in range(8):
        ux = modes[0::3, j]
        share = (ux @ ux) / (modes[:, j] @ modes[:, j])
        if share > 0.9:
            f_long = freqs[j]
            break
    return f_exact, f_long


def main():
    err_q, err_off, err_l = test_element_mass_quadrature()
    print(f"[element mass] consistent vs Gauss quadrature: "
          f"rel err = {err_q:.3e}, off-block = {err_off:.3e}, "
          f"lumped diag rel err = {err_l:.3e}")
    assert err_q < 1e-12 and err_off < 1e-12 and err_l < 1e-12, \
        "element mass matrix disagrees with independent quadrature"

    err_m = test_mass_conservation()
    print(f"[mass conservation] max rel err over both matrices x "
          f"3 components = {err_m:.3e}")
    assert err_m < 1e-12, "total mass not conserved"

    freqs, lam, ratio, err_orth = test_rigid_body_modes()
    print(f"[rigid body] freqs = {[f'{f:.4g}' for f in freqs]} Hz, "
          f"max|lambda_1..6|/lambda_7 = {ratio:.3e}, "
          f"orthonormality err = {err_orth:.3e}")
    assert lam[6] > 0.0, "7th eigenvalue must be positive"
    assert ratio < 1e-6, "rigid-body eigenvalues not ~0"
    assert err_orth < 1e-8, "mode shapes not M-orthonormal"

    f_eb, f_cons, f_lump = test_cantilever_beam()
    err_c = f_cons / f_eb - 1.0
    err_l2 = f_lump / f_eb - 1.0
    print(f"[cantilever] EB f1 = {f_eb:.3f} Hz, consistent "
          f"f1 = {f_cons:.3f} Hz ({err_c:+.2%}), lumped "
          f"f1 = {f_lump:.3f} Hz ({err_l2:+.2%})")
    assert 0.0 < err_c < 0.10, \
        "cantilever f1 must exceed EB by < 10 % (P1 overstiff)"
    assert abs(err_l2) < 0.10, "lumped-mass f1 off by > 10 %"
    assert abs(f_lump - f_cons) / f_cons < 0.01, \
        "lumped and consistent f1 should nearly coincide here"

    f_exact, f_long = test_rod_longitudinal()
    assert f_long is not None, "no longitudinal mode found"
    err_r = f_long / f_exact - 1.0
    print(f"[rod longitudinal] exact f1 = {f_exact:.2f} Hz, "
          f"FEM f1 = {f_long:.2f} Hz ({err_r:+.2%})")
    assert abs(err_r) < 0.03, "rod longitudinal frequency off by > 3 %"

    print("ALL 3D MODAL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
