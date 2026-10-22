"""Validation of the v1.9 nonlinear B-H phasor solver (Tet10 A-V).

All numbers are simulation outputs on a deliberately small mesh
(2-core workstation): an iron cylinder (R = 15 mm, L = 60 mm,
L/D = 2) in an air box (half = 45 mm, h = 16 mm), sigma = 6e6 S/m,
f = 2 kHz, uniform applied field along z.  The skin depth at the
low-field mu_r = 201 is delta = 0.32 mm, so h/delta ~ 49: the skin
is NOT resolved and absolute values are coarse-mesh values.  Every
assertion below is designed to be robust to that (same-mesh
comparisons, exact linear limits, wide documented bands) -- the
same protocol as the v1.7 coil consistency check.  A
resolution-converged study is a compute task, not this version.

Material law: the BIG-M4/M10 analytic model curve
M(H) = Ms*tanh(H/H_REF), Ms = 1.9/MU0, H_REF = Ms/200 (secant
mu_r -> 201 at low field, -> 1 in deep saturation).  A documented
*model assumption*, not measured material data; isothermal.

Tests:
1. bh_secant_mu_r anchor values (exact arithmetic on the curve).
2. Linear limit: mu_mag0 = 0 (mu_r == 1 identically) and
   iron = None both reproduce solve_eddy_current_tet10 to < 1e-6.
3. Saturation direction (strong drive, H0 = 1e6 A/m): nonlinear
   total power strictly between the constant-mu_r = 1 and
   constant-mu_r = 201 linear powers (same mesh, iron-only mu);
   iron max |B| clamped below the mu_r = 201 linear value (which
   extrapolates to an unphysical 4.4 T); converged mu_r field has
   the saturation gradient (surface layer mean mu_r far below the
   deep mean, min mu_r in single digits).
4. Picard convergence on the same strong run: converged = True,
   rel_dB history bounded (peak 1.6e-2), final < tol = 1e-3,
   power history finite and decreasing overall.  Measured:
   40 iterations (prototype run, this machine).
5. M10 order-of-magnitude check (weak drive, H0 = 2e4 A/m, the
   BIG-M10 case-A excitation): mid-section unit-length power of
   the finite 3D cylinder vs M10's axisymmetric long-cylinder
   nonlinear P/L = 10112.38 W/m -- ratio inside the wide
   0.2--5 band.  NOT a tight benchmark: finite length (end /
   demagnetising effects), B_applied box drive vs axi Neumann
   H0, square-vs-graded meshes and h/delta ~ 49 all differ.
"""

import numpy as np

from cloudfem.mesh.tet10_mesh import tet4_to_tet10
from cloudfem.mesh.unstructured3d import cylinder_in_box_mesh
from cloudfem.physics.em import MU0
from cloudfem.physics.em3d_tet10 import (
    bh_secant_mu_r, element_flux_density_tet10,
    solve_eddy_current_nonlinear_tet10, solve_eddy_current_tet10,
    total_joule_power_tet10)

PI = np.pi
GEO = dict(R=0.015, L=0.06, HALF=0.045, H=0.016,
           SIGMA=6.0e6, FREQ=2000.0)
M10_PL = 10112.38          # BIG-M10 case A nonlinear P/L [W/m]

_CACHE = {}


def _mesh():
    if "mesh" not in _CACHE:
        m4 = cylinder_in_box_mesh(GEO["R"], GEO["L"], half=GEO["HALF"],
                                  h=GEO["H"])
        _CACHE["mesh"] = tet4_to_tet10(m4)
    return _CACHE["mesh"]


def _in_iron(x, y, z):
    return (x * x + y * y < GEO["R"] ** 2) and abs(z) < GEO["L"] / 2


def _sigf(x, y, z):
    return GEO["SIGMA"] if _in_iron(x, y, z) else 0.0


def _iron_mask(mesh):
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    return np.array([_in_iron(x, y, z) for x, y, z in cen]), cen


def _linear_const_mu(mesh, mu_r, H0):
    """Linear solve with constant mu_r in the iron, 1 in air."""
    def nuf(x, y, z):
        return 1.0 / (MU0 * mu_r) if _in_iron(x, y, z) else 1.0 / MU0
    omega = 2 * PI * GEO["FREQ"]
    a, v = solve_eddy_current_tet10(mesh, nu=nuf, sigma=_sigf,
                                    omega=omega,
                                    B_applied=(0.0, 0.0, MU0 * H0))
    P, _ = total_joule_power_tet10(mesh, a, v, _sigf, omega)
    B = element_flux_density_tet10(mesh, a)
    Bamp = np.sqrt(np.sum(np.abs(B) ** 2, axis=1))
    return a, P, Bamp


def _strong_run():
    """The H0 = 1e6 nonlinear run shared by tests 3 and 4."""
    if "strong" not in _CACHE:
        mesh = _mesh()
        omega = 2 * PI * GEO["FREQ"]
        a, v, info = solve_eddy_current_nonlinear_tet10(
            mesh, sigma=_sigf, omega=omega,
            B_applied=(0.0, 0.0, MU0 * 1.0e6), iron=_in_iron,
            tol=1e-3, max_iter=60, relax=0.5)
        _CACHE["strong"] = (a, v, info)
    return _CACHE["strong"]


def test_bh_curve_anchors():
    """Secant mu_r curve: exact anchor values (M10 law)."""
    Ms = 1.9 / MU0
    H_REF = Ms / 200.0
    mu0 = float(bh_secant_mu_r(0.0))
    mu_knee = float(bh_secant_mu_r(H_REF))
    mu_deep = float(bh_secant_mu_r(100.0 * H_REF))
    print(f"[bh] mu_r(0) = {mu0:.4f}, mu_r(H_REF) = {mu_knee:.4f}, "
          f"mu_r(100 H_REF) = {mu_deep:.4f} (simulation outputs)")
    assert abs(mu0 - 201.0) < 1e-9
    assert abs(mu_knee - (1.0 + 200.0 * np.tanh(1.0))) < 1e-9
    assert abs(mu_deep - (1.0 + 200.0 * np.tanh(100.0) / 100.0)) < 1e-9
    H = np.logspace(0, 7, 50)
    mu = bh_secant_mu_r(H)
    assert np.all(np.diff(mu) < 0), "mu_r must decrease monotonically"
    assert float(bh_secant_mu_r(123.0, mu_mag0=0.0)) == 1.0


def test_linear_limit():
    """mu_mag0 = 0 and iron = None reproduce the linear solver."""
    mesh = _mesh()
    omega = 2 * PI * GEO["FREQ"]
    H0 = 2.0e4
    a_lin, _, _ = _linear_const_mu(mesh, 1.0, H0)
    scale = float(np.linalg.norm(a_lin))

    a0, _, info0 = solve_eddy_current_nonlinear_tet10(
        mesh, sigma=_sigf, omega=omega,
        B_applied=(0.0, 0.0, MU0 * H0), iron=_in_iron, mu_mag0=0.0)
    d0 = float(np.linalg.norm(a0 - a_lin) / scale)

    aN, _, infoN = solve_eddy_current_nonlinear_tet10(
        mesh, sigma=_sigf, omega=omega,
        B_applied=(0.0, 0.0, MU0 * H0), iron=None)
    dN = float(np.linalg.norm(aN - a_lin) / scale)
    print(f"[lin-limit] rel diff mu_mag0=0: {d0:.3e} "
          f"(iters {info0['iterations']}), iron=None: {dN:.3e} "
          f"(iters {infoN['iterations']}) (simulation outputs)")
    assert info0["converged"] and infoN["converged"]
    assert d0 < 1e-6, "mu_mag0=0 must reproduce the linear solve"
    assert dN < 1e-6, "iron=None must reproduce the linear solve"


def test_saturation_direction():
    """Strong drive: P bracketed, |B| clamped, mu_r graded."""
    mesh = _mesh()
    iron_mask, cen = _iron_mask(mesh)
    _, _, info = _strong_run()
    _, P1, _ = _linear_const_mu(mesh, 1.0, 1.0e6)
    _, P201, Bamp201 = _linear_const_mu(mesh, 201.0, 1.0e6)
    P_nl = float(info["P"][-1])
    Bamp = np.sqrt(np.sum(np.abs(info["B_elem"]) ** 2, axis=1))
    maxB_nl = float(Bamp[iron_mask].max())
    maxB_201 = float(Bamp201[iron_mask].max())
    mu_e = info["mu_r_elem"]
    r_c = np.hypot(cen[:, 0], cen[:, 1])
    surf = iron_mask & (r_c > 0.75 * GEO["R"]) & (np.abs(cen[:, 2]) < 0.01)
    deep = iron_mask & (r_c < 0.4 * GEO["R"]) & (np.abs(cen[:, 2]) < 0.01)
    mu_surf = float(mu_e[surf].mean())
    mu_deep = float(mu_e[deep].mean())
    print(f"[sat] P: mu_r=1 {P1:.1f} W < nonlinear {P_nl:.1f} W "
          f"< mu_r=201 {P201:.1f} W (simulation outputs)")
    print(f"[sat] max|B| iron: nonlinear {maxB_nl:.3f} T vs "
          f"mu_r=201 linear {maxB_201:.3f} T; mu_r surface mean "
          f"{mu_surf:.1f}, deep mean {mu_deep:.1f}, "
          f"min {mu_e[iron_mask].min():.1f}")
    assert P1 < P_nl < P201, "nonlinear P must sit between the brackets"
    assert maxB_nl < 0.85 * maxB_201, "|B| must be clamped by saturation"
    assert mu_surf < 0.5 * mu_deep, "surface must saturate below the core"
    assert float(mu_e[iron_mask].min()) < 20.0, "deep saturation reached"


def test_picard_convergence():
    """Picard history of the strong run: bounded, ends below tol."""
    _, _, info = _strong_run()
    hist = info["rel_dB"]
    P = info["P"]
    print(f"[picard] iters = {info['iterations']}, converged = "
          f"{info['converged']}, rel_dB peak = {hist.max():.3e}, "
          f"final = {hist[-1]:.3e}; P {P[0]:.1f} -> {P[-1]:.1f} W "
          f"(simulation outputs)")
    assert info["converged"], "strong-drive Picard must converge"
    assert np.all(np.isfinite(hist)) and np.all(np.isfinite(P))
    assert hist.max() < 5e-2, "rel_dB history must stay bounded"
    assert hist[-1] < info["tol"], "final rel_dB must be below tol"
    assert hist[-1] < hist[-5], "tail must still be decreasing"
    assert P[-1] < P[0], "power must decrease overall as mu saturates"


def test_m10_order_of_magnitude():
    """Weak drive (M10 case-A excitation): mid-section P/L band."""
    mesh = _mesh()
    omega = 2 * PI * GEO["FREQ"]
    a, v, info = solve_eddy_current_nonlinear_tet10(
        mesh, sigma=_sigf, omega=omega,
        B_applied=(0.0, 0.0, MU0 * 2.0e4), iron=_in_iron,
        tol=1e-3, max_iter=60, relax=0.5)
    _, Q = total_joule_power_tet10(mesh, a, v, _sigf, omega)
    _, cen = _iron_mask(mesh)
    vols = mesh.element_volumes()
    slab = np.abs(cen[:, 2]) < GEO["L"] / 8
    P_per_L = float(np.sum(Q[slab] * vols[slab]) / (GEO["L"] / 4))
    ratio = P_per_L / M10_PL
    print(f"[m10] 3D mid-section P/L = {P_per_L:.1f} W/m vs M10 axi "
          f"{M10_PL:.2f} W/m: ratio {ratio:.3f} (band 0.2--5; "
          f"nonlinear iters {info['iterations']}, converged "
          f"{info['converged']}) (simulation outputs)")
    assert info["converged"]
    assert 0.2 < ratio < 5.0, "order-of-magnitude check vs BIG-M10"


if __name__ == "__main__":
    test_bh_curve_anchors()
    test_linear_limit()
    test_saturation_direction()
    test_picard_convergence()
    test_m10_order_of_magnitude()
    print("ALL NONLINEAR BH VALIDATIONS PASSED")
