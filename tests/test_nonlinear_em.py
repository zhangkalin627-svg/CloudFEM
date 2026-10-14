"""Validation of the nonlinear magnetostatics (B-H saturation) solver.

1. Linear limit: with a constant reluctivity both solvers must
   reproduce the linear Poisson solve to solver precision.
2. Manufactured solution: A = x(1-x)*y(1-y) on the unit square with the
   cubic model nu = nu0*(1+alpha*|B|^2); the source term is derived in
   closed form (polynomial). P1 elements should show O(h^2) in L2.
   Newton must converge in a handful of iterations.
3. Deep saturation: with the saturation model at very large excitation
   the iron reluctivity collapses to nu0 everywhere, so the solution
   must approach the linear solve with nu = nu0.
4. Saturation knee (Newton): an iron bar driven into saturation --
   Newton must converge where Picard stalls, and the answer must show
   the physical saturation signature (sublinear B vs excitation).
"""

import numpy as np

from cloudfem.mesh.tri_mesh import rect_mesh
from cloudfem.fem.poisson import solve_poisson
from cloudfem.physics.em import MU0
from cloudfem.physics.nonlinear_em import (
    solve_nonlinear_magnetostatics, bh_cubic, bh_saturation, flux_density,
)

NU0 = 1.0 / MU0
BC0 = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}


def test_linear_limit():
    # alpha = 0 -> constant reluctivity nu0/500
    mesh = rect_mesh(24, 24, xlim=(0, 1), ylim=(0, 1))
    J = lambda x, y: 1e6 * np.sin(np.pi * x) * np.sin(np.pi * y)
    A_lin = solve_poisson(mesh, k=NU0 / 500.0, f=J, bc=BC0)
    out = {}
    for method in ("newton", "picard"):
        A_nl, info = solve_nonlinear_magnetostatics(
            mesh, bh_cubic(nu0=NU0 / 500.0, alpha=0.0), J_s=J, bc=BC0,
            tol=1e-10, method=method)
        assert info["converged"], f"{method} failed on linear problem"
        out[method] = (np.abs(A_nl - A_lin).max() / np.abs(A_lin).max(),
                       info["iterations"])
    return out


def _mms_source(nu0, alpha):
    """Closed-form source for A = x(1-x)*y(1-y), nu = nu0*(1+alpha*|B|^2).

    With u = x-x^2, v = y-y^2: A = u*v, and
    f = -nu0*( 2*alpha*[u'^2*v*(-2*v^2+u*v'^2) + u*v'^2*(-2*u^2+u'^2*v)]
                + (1+alpha*|grad A|^2)*(-2)*(u+v) ).
    """
    def f(x, y):
        u = x - x ** 2
        v = y - y ** 2
        up = 1 - 2 * x
        vp = 1 - 2 * y
        B2 = (up * v) ** 2 + (u * vp) ** 2
        gg = 2 * alpha * (up ** 2 * v * (-2 * v ** 2 + u * vp ** 2)
                          + u * vp ** 2 * (-2 * u ** 2 + up ** 2 * v))
        lap = -2.0 * (u + v)
        return -nu0 * (gg + (1 + alpha * B2) * lap)
    return f


def test_mms_convergence(alpha=5.0, method="newton"):
    errs, iters = [], []
    for nx in (16, 32, 64):
        mesh = rect_mesh(nx, nx, xlim=(0, 1), ylim=(0, 1))
        A, info = solve_nonlinear_magnetostatics(
            mesh, bh_cubic(nu0=NU0, alpha=alpha),
            J_s=_mms_source(NU0, alpha), bc=BC0, tol=1e-10, method=method)
        assert info["converged"], f"{method} failed at nx={nx}"
        iters.append(info["iterations"])
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        A_exact = (x - x ** 2) * (y - y ** 2)
        errs.append(float(np.sqrt(np.mean((A - A_exact) ** 2))))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    return errs, rates, iters


def test_deep_saturation():
    # uniform iron square, enormous uniform excitation: nu -> nu0
    # almost everywhere, so the answer must approach the linear nu0 solve
    mesh = rect_mesh(24, 24, xlim=(0, 1), ylim=(0, 1))
    J = 3e9
    A_nl, info = solve_nonlinear_magnetostatics(
        mesh, bh_saturation(mu_r0=2000.0, B_knee=1.0), J_s=J, bc=BC0,
        tol=1e-9, method="newton")
    assert info["converged"], "newton failed to converge in deep saturation"
    A_lin = solve_poisson(mesh, k=NU0, f=J, bc=BC0)
    err = np.sqrt(np.mean((A_nl - A_lin) ** 2)) / np.abs(A_lin).max()
    B = flux_density(mesh, A_nl)
    Bmax = float(np.abs(B).max())
    return err, Bmax, info


def test_saturation_knee_newton():
    # Iron picture-frame (closed flux path, no air gap) with a coil around
    # the left leg, driven from the linear regime into saturation.
    # Newton must converge at every level and the B-vs-excitation curve
    # must bend over (sublinear): the signature Picard cannot reach.
    L = 0.12
    mesh = rect_mesh(48, 48, xlim=(0, L), ylim=(0, L))

    def in_r(x, y, x0, x1, y0, y1):
        x = np.asarray(x)
        y = np.asarray(y)
        return (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)

    def is_iron(x, y):
        outer = in_r(x, y, 0.02, 0.10, 0.02, 0.10)
        window = in_r(x, y, 0.04, 0.08, 0.04, 0.08)
        return outer & ~window

    def js(J0):
        return lambda x, y: np.where(
            in_r(x, y, 0.008, 0.020, 0.02, 0.10), J0,
            np.where(in_r(x, y, 0.040, 0.052, 0.04, 0.08), -J0, 0.0))

    nu_model = bh_saturation(mu_r0=2000.0, B_knee=1.0, iron=is_iron)
    p = mesh.nodes[mesh.elements]
    xc, yc = p[:, :, 0].mean(axis=1), p[:, :, 1].mean(axis=1)
    leg = in_r(xc, yc, 0.02, 0.04, 0.04, 0.08)   # left leg mid-section

    COIL_AREA = 0.012 * 0.08
    B_leg, Bmax_iron, max_iter = [], [], 0
    A = None
    for NI in (10.0, 20.0, 3000.0):
        J0 = NI / (2 * COIL_AREA)
        A, info = solve_nonlinear_magnetostatics(
            mesh, nu_model, J_s=js(J0), bc=BC0, tol=1e-7,
            method="newton", A_init=A)
        assert info["converged"], f"newton failed at NI={NI:.0f}"
        max_iter = max(max_iter, info["iterations"])
        B = flux_density(mesh, A)
        Bmag = np.hypot(B[:, 0], B[:, 1])
        B_leg.append(float(Bmag[leg].mean()))
        iron_e = is_iron(xc, yc)
        Bmax_iron.append(float(Bmag[iron_e].max()))
    # linear regime: doubling NI doubles B
    lin_ratio = B_leg[1] / B_leg[0]
    # saturation: 150x excitation gives far less than 150x B
    sat_ratio = B_leg[2] / B_leg[1]
    return B_leg, Bmax_iron, lin_ratio, sat_ratio, max_iter


if __name__ == "__main__":
    out = test_linear_limit()
    for method, (err, it) in out.items():
        print(f"[linear limit/{method}] max rel diff = {err:.3e} "
              f"({it} iters)")
        assert err < 1e-9, "nonlinear solver does not reproduce linear limit"

    errs, rates, iters = test_mms_convergence(method="newton")
    print(f"[mms/newton] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0), "
          f"iters = {iters}")
    assert all(r > 1.7 for r in rates), "convergence rate too low"
    assert max(iters) <= 8, "newton took too many iterations on MMS"

    errs, rates, iters = test_mms_convergence(method="picard")
    print(f"[mms/picard] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]}, iters = {iters}")
    assert all(r > 1.7 for r in rates), "picard convergence rate too low"

    err, Bmax, info = test_deep_saturation()
    print(f"[deep saturation] rel L2 diff vs nu0-linear solve = {err:.3e}, "
          f"max |B| = {Bmax:.1f} T ({info['iterations']} newton iters)")
    assert err < 0.05, "deep-saturation limit not recovered"
    assert Bmax > 50.0, "excitation did not actually saturate the iron"

    B_leg, Bmax_iron, lin_ratio, sat_ratio, max_iter = \
        test_saturation_knee_newton()
    print(f"[saturation knee] B_leg = {[f'{b:.3f}' for b in B_leg]} T, "
          f"max|B|_iron = {[f'{b:.2f}' for b in Bmax_iron]} T, "
          f"linear ratio = {lin_ratio:.3f} (expect 2.0), "
          f"saturation ratio = {sat_ratio:.2f} (150x NI; expect < 20), "
          f"max newton iters = {max_iter}")
    assert abs(lin_ratio - 2.0) < 0.1, "linear regime not linear"
    assert sat_ratio < 20.0, "no saturation knee observed"
    assert Bmax_iron[-1] > 0.7, "iron never entered the knee transition"

    print("ALL NONLINEAR-EM VALIDATIONS PASSED")
