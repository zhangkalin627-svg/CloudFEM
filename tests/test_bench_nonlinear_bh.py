"""
BIG-M10: Nonlinear B-H + Curie point in the axisymmetric eddy framework
========================================================================

Extends the BIG-M8 axisymmetric eddy-current benchmark (CloudFEM,
utilde = r*A_phi, Neumann excitation H_z(a) = H0 = const) with the
nonlinear steel law of BIG-M4, in exactly the same calibration:

    mu_r(|H|, T) = 1 + mu_mag(|H|) * step(T)
      step(T)  : tanh smooth switch at the Curie point Tc = 770 degC
                 (width 15 K), 1 below Tc, 0 above.
      mu_mag   : M(H) = Ms * tanh(H / H_REF), Ms = Bsat/mu0, Bsat = 1.9 T,
                 H_REF = Ms/200 so the low-field mu_mag -> 200;
                 mu_mag(H) = M(H)/H (secant permeability).
    sigma(T)   : tanh transition 6e6 -> 1e6 S/m (centre 550 degC, width
                 150 K) -- copied verbatim from BIG-M4.
    cp(T)      : 480 + 0.18 (T-25) plus the Curie Gaussian peak
                 (sigma = 100 K, integral exactly 2e5 J/kg) -- BIG-M4.

Nonlinear solve: Picard (secant) iteration on the element-wise mu_r
field.  Each pass re-solves the linear axi problem with a callable
nu(x, y) (the assembler samples coefficients at element centroids, and
a tensor-grid lookup maps each centroid to its mesh cell), recovers
the element B phasor, forms |H| = |B| / (mu0 * mu_r_secant), updates
mu_r from the magnetisation curve, and relaxes (factor 0.5).  The
iteration stops on the total-power change; iteration counts and final
residuals are printed.

Case A (isothermal, 20 degC): nonlinear mu vs the two linear references
mu_r = 1 and mu_r = 200 at the same excitation (a = 20 mm, f = 2 kHz,
H0 = 2.0e4 A/m, sigma = sigma(20 degC) = 5.996e6 S/m).  The linear
cases are also checked against the Bessel analytic solution.

Case B (transient, loosely coupled EM/thermal): the same cylinder at
f = 10 kHz with H0 = 1.49e5 A/m held constant -- the value BIG-M4's
surface-impedance iteration produced at t = 0 for P_surf = 3 MW/m^2,
but here imposed as a constant-H0 (constant ampere-turn) excitation
instead of M4's constant absorbed power.  sigma(T) and mu_r(|H|, T)
are updated every 0.1 s (Picard, warm-started); the heat solve is
CloudFEM transient_heat_axi (implicit Euler, dt = 0.01 s), adiabatic
surface (same short-time assumption as M4), k = 30 W/(m K) const.

Geometry: r-z slab of height Lz = a with a geometrically graded radial
mesh (growth 1.18, surface layer ~0.069 mm); the exact solution is
z-invariant, so the slab reproduces the infinite cylinder.  All powers
are per unit length (total slab power / Lz).

Run:  python3 tests/test_bench_nonlinear_bh.py   (from the repo root)
Integrated into the CloudFEM suite by BIG-M11; physics unchanged
from the standalone BIG-M10 script.  The frozen-profile check in
case A and the verdict block in main() were added by BIG-M11 to
make the benchmark criteria explicit (no computed value changes).
Writes big_m10_caseB.csv (t, T_surf, T_axis, P/L, delta_eff, mu_r,s).
"""

import os
import sys

import numpy as np
from scipy.special import jv, erf

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from cloudfem.mesh.tri_mesh import TriMesh
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, total_joule_power_axi,
    joule_heating_density_axi, flux_density_axi, transient_heat_axi, MU0)

# ------------------------------------------------------------ geometry etc.
A_RAD = 0.02                  # cylinder radius [m]
LZ = A_RAD                    # r-z slab height [m]
NZ = 4                        # axial divisions (z-invariant solution)
GROWTH = 1.18                 # radial geometric growth ratio

# ------------------------------------------------- material laws (BIG-M4)
TC = 770.0                    # Curie temperature [degC]
W_STEP = 15.0                 # mu switch smooth width [K]
BSAT = 1.9                    # saturation polarisation [T]
MS = BSAT / MU0               # saturation magnetisation [A/m]
MU_MAG0 = 200.0               # low-field magnetic part of mu_r
H_REF = MS / MU_MAG0          # tanh scale of M(H) [A/m]

RHO = 7850.0                  # density [kg/m^3]
K_TH = 30.0                   # thermal conductivity [W/(m K)], constant

SIG_HI = 6.0e6                # conductivity, cold [S/m]
SIG_LO = 1.0e6                # conductivity, hot [S/m]
T_SIG = 550.0                 # sigma transition centre [degC]
W_SIG = 150.0                 # sigma transition width [K]

CP_PEAK_INT = 2.0e5           # Gaussian cp-peak integral [J/kg]
CP_PEAK_SIG = 100.0           # Gaussian cp-peak sigma [K]


def step_T(T):
    """Smooth Curie switch: 1 below Tc, 0 above Tc (BIG-M4 verbatim)."""
    return 0.5 * (1.0 - np.tanh((T - TC) / W_STEP))


def sigma_T(T):
    """Electrical conductivity [S/m], 6e6 -> 1e6 (BIG-M4 verbatim)."""
    s = 0.5 * (1.0 - np.tanh((T - T_SIG) / W_SIG))
    return SIG_LO + (SIG_HI - SIG_LO) * s


def cp_T(T):
    """Apparent heat capacity [J/(kg K)] incl. Curie peak (BIG-M4)."""
    base = 480.0 + 0.18 * (T - 25.0)
    amp = CP_PEAK_INT / (CP_PEAK_SIG * np.sqrt(2.0 * np.pi))
    peak = amp * np.exp(-0.5 * ((T - TC) / CP_PEAK_SIG) ** 2)
    return base + peak


def cp_integral(T):
    """Integral of cp from 25 degC to T [J/kg] (energy balance)."""
    T0 = 25.0
    base = 480.0 * (T - T0) + 0.09 * ((T - 25.0) ** 2 - (T0 - 25.0) ** 2)
    z = (T - TC) / (CP_PEAK_SIG * np.sqrt(2.0))
    z0 = (T0 - TC) / (CP_PEAK_SIG * np.sqrt(2.0))
    peak = CP_PEAK_INT * 0.5 * (erf(z) - erf(z0))
    return base + peak


def mu_mag_of_H(Hmag):
    """M(H)/H for M = Ms tanh(H/H_REF); -> MU_MAG0 as H -> 0."""
    h = np.asarray(Hmag, dtype=float) / H_REF
    ratio = np.where(h > 1.0e-6,
                     np.tanh(h) / np.maximum(h, 1.0e-300),
                     1.0 - h * h / 3.0)
    return MU_MAG0 * ratio


def mu_r_of_HT(Hmag, T):
    """mu_r = 1 + mu_mag(|H|) * step(T)  (BIG-M4 law)."""
    return 1.0 + mu_mag_of_H(Hmag) * step_T(T)


# ------------------------------------------------------------------- mesh
def graded_slab_mesh(h_surf_target):
    """Tensor r-z mesh, geometric radial grading toward r = a.

    Same construction as BIG-M8: layer thicknesses from the surface
    inward form a geometric progression with ratio GROWTH summing
    exactly to A_RAD.  Returns (mesh, r_nodes, z_nodes).
    """
    q = GROWTH
    m = 1
    while h_surf_target * (q ** m - 1.0) / (q - 1.0) < A_RAD:
        m += 1
    h0 = A_RAD * (q - 1.0) / (q ** m - 1.0)
    thick = h0 * q ** np.arange(m)
    edges_from_surf = np.concatenate(([0.0], np.cumsum(thick)))
    r_nodes = A_RAD - edges_from_surf[::-1]
    nr = len(r_nodes) - 1
    z_nodes = np.linspace(0.0, LZ, NZ + 1)

    nodes = np.array([(r, z) for z in z_nodes for r in r_nodes])

    def idx(i, j):
        return j * (nr + 1) + i

    elements = []
    for j in range(NZ):
        for i in range(nr):
            n0, n1 = idx(i, j), idx(i + 1, j)
            n2, n3 = idx(i, j + 1), idx(i + 1, j + 1)
            elements.append((n0, n1, n3))
            elements.append((n0, n3, n2))

    boundaries = {
        "left": np.array([idx(0, j) for j in range(NZ + 1)]),
        "right": np.array([idx(nr, j) for j in range(NZ + 1)]),
        "bottom": np.array([idx(i, 0) for i in range(nr + 1)]),
        "top": np.array([idx(i, NZ) for i in range(nr + 1)]),
    }
    mesh = TriMesh(nodes, np.array(elements), boundaries)
    return mesh, r_nodes, z_nodes


class CellLookup:
    """Maps points (element centroids) to tensor-grid cells.

    Cell values are stored flat with index jz * nr + ir.  The FEM
    assembler evaluates coefficient callables at element centroids,
    which always lie strictly inside their own cell, so the lookup is
    exact for every assembled coefficient.
    """

    def __init__(self, mesh, r_nodes, z_nodes):
        self.r_nodes = r_nodes
        self.z_nodes = z_nodes
        self.nr = len(r_nodes) - 1
        self.nz = len(z_nodes) - 1
        cents = mesh.nodes[mesh.elements].mean(axis=1)
        ir = np.clip(np.searchsorted(r_nodes, cents[:, 0]) - 1,
                     0, self.nr - 1)
        jz = np.clip(np.searchsorted(z_nodes, cents[:, 1]) - 1,
                     0, self.nz - 1)
        self.elem_cell = jz * self.nr + ir
        self.elem_ir = ir
        # cell-centre radius (mean of its elements' centroid radii)
        self.cell_rc = np.zeros(self.nr)
        for i in range(self.nr):
            self.cell_rc[i] = cents[ir == i, 0].mean()

    def fn(self, cell_values):
        """Return a callable f(x, y) -> value of the containing cell."""
        vals = np.asarray(cell_values, dtype=float)
        nr, r_nodes, z_nodes = self.nr, self.r_nodes, self.z_nodes

        def f(x, y):
            i = min(max(np.searchsorted(r_nodes, x) - 1, 0), nr - 1)
            j = min(max(np.searchsorted(z_nodes, y) - 1, 0),
                    len(z_nodes) - 2)
            return float(vals[j * nr + i])

        return f

    def elem_to_cell(self, elem_values):
        """Average an element-wise array onto cells."""
        out = np.zeros(self.nr * self.nz)
        cnt = np.zeros(self.nr * self.nz)
        np.add.at(out, self.elem_cell, elem_values)
        np.add.at(cnt, self.elem_cell, 1.0)
        return out / np.maximum(cnt, 1.0)

    def cell_to_elem(self, cell_values):
        return np.asarray(cell_values)[self.elem_cell]


# ------------------------------------------------------------- EM helpers
def h_magnitude_elem(mesh, utilde, mu_elem):
    """|H| phasor amplitude per element from B and the secant mu."""
    B = flux_density_axi(mesh, utilde)
    H = B / (MU0 * np.maximum(mu_elem, 1.0)[:, None])
    return np.sqrt(np.abs(H[:, 0]) ** 2 + np.abs(H[:, 1]) ** 2)


def delta_eff_of(mesh, lk, H_elem, H0):
    """1/e penetration depth of |H| from the surface [m].

    Radial profile = |H| averaged over all cells at the same radius;
    reference level is the outermost cell's value, depth is measured
    inward from r = a and log-interpolated.
    """
    prof = np.zeros(lk.nr)
    for i in range(lk.nr):
        prof[i] = H_elem[lk.elem_ir == i].mean()
    depth = A_RAD - lk.cell_rc          # depth of each cell centre
    order = np.argsort(depth)
    d_sorted = depth[order]
    p_sorted = prof[order]
    target = p_sorted[0] / np.e
    below = np.nonzero(p_sorted < target)[0]
    if len(below) == 0:
        return float(d_sorted[-1]), prof
    k = below[0]
    if k == 0:
        return float(d_sorted[0]), prof
    d1, d2 = d_sorted[k - 1], d_sorted[k]
    l1, l2 = np.log(p_sorted[k - 1]), np.log(p_sorted[k])
    lt = np.log(target)
    frac = (lt - l1) / (l2 - l1) if l2 != l1 else 0.0
    return float(d1 + frac * (d2 - d1)), prof


def picard_em(mesh, lk, T_cell, sig_cell, omega, H0, mu_cell,
              tol_P=1.0e-3, max_it=60, relax=0.5):
    """Picard iteration for mu_r(|H|, T) at fixed T and sigma fields.

    T_cell, sig_cell, mu_cell are per-cell arrays.  Returns
    (utilde, mu_cell, P_per_length, n_iter, rel_dP, max_rel_dmu).
    """
    sig_fn = lk.fn(sig_cell)
    T_elem = lk.cell_to_elem(T_cell)
    P_prev = None
    rel_dP = np.inf
    rel_dmu = np.inf
    utilde = None
    P_L = None
    for it in range(1, max_it + 1):
        nu_fn = lk.fn(1.0 / (MU0 * mu_cell))
        utilde = solve_eddy_current_axi(
            mesh, nu=nu_fn, sigma=sig_fn, omega=omega,
            bc={"left": 0.0}, neumann={"right": 2.0 * np.pi * H0})
        P_L = total_joule_power_axi(mesh, utilde, sig_fn, omega) / LZ
        mu_elem = lk.cell_to_elem(mu_cell)
        Hmag = h_magnitude_elem(mesh, utilde, mu_elem)
        mu_tgt = lk.elem_to_cell(mu_r_of_HT(Hmag, T_elem))
        rel_dmu = float(np.max(np.abs(mu_tgt - mu_cell) /
                               np.maximum(mu_cell, 1.0)))
        mu_new = mu_cell + relax * (mu_tgt - mu_cell)
        if P_prev is not None:
            rel_dP = abs(P_L - P_prev) / max(P_L, 1e-300)
            if rel_dP < tol_P and rel_dmu < 1.0e-2:
                mu_cell = mu_new
                break
        P_prev = P_L
        mu_cell = mu_new
    return utilde, mu_cell, P_L, it, rel_dP, rel_dmu


# ------------------------------------------------------- analytic (linear)
def analytic_PL(mu_r, sigma, omega, H0):
    """P/L [W/m] for the linear problem, two independent routes.

    Route 1: Gauss-Legendre volume integral of (sigma/2)|E_phi|^2.
    Route 2: Poynting flux at r = a.  Returns (P_vol, P_poy, delta).
    """
    mu = MU0 * mu_r
    delta = np.sqrt(2.0 / (omega * mu * sigma))
    k = (1.0 - 1.0j) / delta

    def e_phi(r):
        return (H0 * k / sigma) * jv(1, k * r) / jv(0, k * A_RAD)

    xg, wg = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, A_RAD, 801)
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        total += np.sum(w * 0.5 * sigma * np.abs(e_phi(r)) ** 2
                        * 2.0 * np.pi * r)
    p_vol = float(total)
    p_poy = float(-np.pi * A_RAD * np.real(e_phi(A_RAD) * np.conj(H0)))
    return p_vol, p_poy, delta


def linear_solve(mesh, lk, mu_r, sigma, omega, H0):
    """Linear axi solve at constant mu_r, sigma. Returns (P_L, deff)."""
    utilde = solve_eddy_current_axi(
        mesh, nu=1.0 / (MU0 * mu_r), sigma=sigma, omega=omega,
        bc={"left": 0.0}, neumann={"right": 2.0 * np.pi * H0})
    P_L = total_joule_power_axi(mesh, utilde, sigma, omega) / LZ
    mu_elem = np.full(mesh.n_elements, float(mu_r))
    Hmag = h_magnitude_elem(mesh, utilde, mu_elem)
    deff, _ = delta_eff_of(mesh, lk, Hmag, H0)
    return P_L, deff


# ------------------------------------------------------------------ case A
def case_A():
    print("=" * 72)
    print("CASE A  isothermal 20 degC: nonlinear mu vs linear references")
    F = 2000.0
    omega = 2.0 * np.pi * F
    H0 = 2.0e4
    T_A = 20.0
    sigma = float(sigma_T(T_A))
    print(f"  a = {A_RAD*1e3:.0f} mm, f = {F:.0f} Hz, H0 = {H0:.3e} A/m, "
          f"T = {T_A:.0f} degC, sigma = {sigma:.6e} S/m")

    mesh, r_nodes, z_nodes = graded_slab_mesh(0.08e-3)
    lk = CellLookup(mesh, r_nodes, z_nodes)
    print(f"  mesh: nr = {lk.nr}, nz = {lk.nz}, "
          f"elements = {mesh.n_elements}, nodes = {mesh.n_nodes}")

    ncell = lk.nr * lk.nz
    T_cell = np.full(ncell, T_A)
    sig_cell = np.full(ncell, sigma)

    # ---- linear references, FEM + analytic
    lin = {}
    for mu_r in (1.0, 200.0):
        P_fem, deff = linear_solve(mesh, lk, mu_r, sigma, omega, H0)
        P_an, P_poy, delta = analytic_PL(mu_r, sigma, omega, H0)
        agree = abs(P_an - P_poy) / P_an
        err = (P_fem - P_an) / P_an
        lin[mu_r] = (P_fem, deff, delta, P_an)
        print(f"  linear mu_r = {mu_r:6.1f}: P_FEM = {P_fem:11.2f} W/m, "
              f"P_analytic = {P_an:11.2f} (routes agree {agree:.1e}), "
              f"FEM err {err*100:+.2f}%, delta_nom = {delta*1e3:.3f} mm, "
              f"delta_eff(FEM) = {deff*1e3:.3f} mm")

    # ---- nonlinear Picard, start from the low-field mu
    mu_init = np.full(ncell, 1.0 + MU_MAG0 * float(step_T(T_A)))
    utilde, mu_cell, P_nl, iters, rel_dP, rel_dmu = picard_em(
        mesh, lk, T_cell, sig_cell, omega, H0, mu_init,
        tol_P=1.0e-4, max_it=80, relax=0.5)
    Hmag = h_magnitude_elem(mesh, utilde, lk.cell_to_elem(mu_cell))
    deff_nl, prof = delta_eff_of(mesh, lk, Hmag, H0)
    mu_surf = float(mu_cell[lk.nr - 1])          # outermost radial cell
    B_surf = MU0 * mu_surf * H0
    print(f"  nonlinear Picard: {iters} iterations, final rel dP = "
          f"{rel_dP:.2e}, max rel dmu = {rel_dmu:.2e}")
    print(f"  nonlinear: P = {P_nl:11.2f} W/m, mu_r,surf = {mu_surf:.1f} "
          f"(B_surf ~ {B_surf:.2f} T), delta_eff = {deff_nl*1e3:.3f} mm")
    for mu_r in (1.0, 200.0):
        d = (P_nl - lin[mu_r][0]) / lin[mu_r][0] * 100.0
        print(f"    vs linear mu_r = {mu_r:6.1f} (P = {lin[mu_r][0]:.2f}): "
              f"{d:+.1f}%")
    # mu_r radial profile (surface -> axis), a few sample radii
    mu_prof = np.array([mu_cell[i] for i in range(lk.nr)])
    print("  mu_r profile (depth[mm] : mu_r): " + ", ".join(
        f"{(A_RAD - lk.cell_rc[i])*1e3:.2f}:{mu_prof[i]:.0f}"
        for i in range(lk.nr - 1, -1, -4)))

    # ---- mesh independence of the nonlinear result
    mesh2, rn2, zn2 = graded_slab_mesh(0.04e-3)
    lk2 = CellLookup(mesh2, rn2, zn2)
    ncell2 = lk2.nr * lk2.nz
    _, _, P_nl2, iters2, _, _ = picard_em(
        mesh2, lk2, np.full(ncell2, T_A), np.full(ncell2, sigma),
        omega, H0, np.full(ncell2, 1.0 + MU_MAG0 * float(step_T(T_A))),
        tol_P=1.0e-4, max_it=80, relax=0.5)
    print(f"  mesh check: finer mesh (nr = {lk2.nr}, {iters2} Picard "
          f"iters) P = {P_nl2:.2f} W/m, change "
          f"{(P_nl2 - P_nl)/P_nl*100:+.2f}%")

    # ---- frozen-profile self-consistency (added by BIG-M11): one
    # linear solve with the converged mu_r field held fixed.  Its power
    # must reproduce the Picard power (the Picard loop's last power was
    # evaluated with the previous iterate's mu field, so this checks
    # that the fixed point is genuinely self-consistent).
    u_fr = solve_eddy_current_axi(
        mesh, nu=lk.fn(1.0 / (MU0 * mu_cell)), sigma=sigma, omega=omega,
        bc={"left": 0.0}, neumann={"right": 2.0 * np.pi * H0})
    P_frozen = total_joule_power_axi(mesh, u_fr, sigma, omega) / LZ
    frozen_rel = abs(P_frozen - P_nl) / P_nl
    print(f"  frozen-profile check: P_frozen = {P_frozen:.2f} W/m vs "
          f"Picard P = {P_nl:.2f} W/m, rel diff = {frozen_rel:.2e}")
    return dict(P_nl=P_nl, mu_surf=mu_surf, deff=deff_nl, iters=iters,
                lin=lin, rel_dP=rel_dP, rel_dmu=rel_dmu,
                P_frozen=P_frozen, frozen_rel=frozen_rel)


# ------------------------------------------------------------------ case B
def case_B():
    print("=" * 72)
    print("CASE B  transient constant-H0 heating across the Curie point")
    F = 10000.0
    omega = 2.0 * np.pi * F
    H0 = 1.49e5
    T_INIT = 25.0
    T_END = 8.0
    EM_DT = 0.1
    HEAT_DT = 0.01
    print(f"  a = {A_RAD*1e3:.0f} mm, f = {F:.0f} Hz, H0 = {H0:.3e} A/m "
          f"(const), T_init = {T_INIT:.0f} degC, t_end = {T_END:.0f} s")

    mesh, r_nodes, z_nodes = graded_slab_mesh(0.08e-3)
    lk = CellLookup(mesh, r_nodes, z_nodes)
    nr = lk.nr
    ncell = lk.nr * lk.nz
    print(f"  mesh: nr = {nr}, nz = {lk.nz}, elements = "
          f"{mesh.n_elements}, nodes = {mesh.n_nodes}")

    # diagnostic nodes: surface (r=a) and axis (r=0) at mid-height
    def idx(i, j):
        return j * (nr + 1) + i

    n_surf = idx(nr, NZ // 2)
    n_axis = idx(0, NZ // 2)

    T = np.full(mesh.n_nodes, T_INIT)
    T_cell = np.full(ncell, T_INIT)
    sig_cell = sigma_T(T_cell)
    mu_cell = 1.0 + MU_MAG0 * step_T(T_cell)

    rec = []          # t, T_surf, T_axis, P_L, delta_eff, mu_surf
    iters_all = []
    E_in = 0.0
    utilde = None
    P_L = None
    mu_surf = None
    deff = None
    n_em = int(round(T_END / EM_DT))
    for n in range(n_em + 1):
        t = n * EM_DT
        utilde, mu_cell, P_L, iters, rel_dP, rel_dmu = picard_em(
            mesh, lk, T_cell, sig_cell, omega, H0, mu_cell,
            tol_P=2.0e-3, max_it=40, relax=0.5)
        iters_all.append(iters)
        Hmag = h_magnitude_elem(mesh, utilde, lk.cell_to_elem(mu_cell))
        deff, _ = delta_eff_of(mesh, lk, Hmag, H0)
        mu_surf = float(mu_cell[nr - 1])
        rec.append((t, float(T[n_surf]), float(T[n_axis]), P_L,
                    deff, mu_surf))
        if n == n_em:
            break
        # heat block with Q and cp frozen at this EM state
        Q_elem, _ = joule_heating_density_axi(
            mesh, utilde, lk.fn(sig_cell), omega)
        Q_cell = lk.elem_to_cell(Q_elem)
        rhocp_cell = RHO * cp_T(T_cell)
        _, hist = transient_heat_axi(
            mesh, k=K_TH, rho_cp=lk.fn(rhocp_cell), Q=lk.fn(Q_cell),
            bc=None, t_end=EM_DT, dt=HEAT_DT, T0=T)
        T = hist[-1]
        E_in += P_L * LZ * EM_DT
        T_cell = lk.elem_to_cell(T[mesh.elements].mean(axis=1))
        sig_cell = sigma_T(T_cell)

    rec = np.array(rec)
    # enthalpy rise (adiabatic box -> must match E_in)
    T_elem = T[mesh.elements].mean(axis=1)
    cents = mesh.nodes[mesh.elements].mean(axis=1)
    areas = mesh.element_areas()
    enth = float(np.sum(RHO * cp_integral(T_elem) * 2.0 * np.pi
                        * cents[:, 0] * areas))
    bal_err = abs(E_in - enth) / E_in * 100.0

    # Curie crossing of the surface (linear interpolation of records)
    t_cross = None
    for i in range(1, len(rec)):
        if rec[i - 1, 1] < TC <= rec[i, 1]:
            f_ = (TC - rec[i - 1, 1]) / (rec[i, 1] - rec[i - 1, 1])
            t_cross = rec[i - 1, 0] + f_ * (rec[i, 0] - rec[i - 1, 0])
            break

    def rate(t0, t1):
        i0 = int(round(t0 / EM_DT))
        i1 = int(round(t1 / EM_DT))
        return (rec[i1, 1] - rec[i0, 1]) / (rec[i1, 0] - rec[i0, 0])

    rate_early = rate(0.0, 0.5)
    rate_after = rate(t_cross + 0.5, t_cross + 1.5) \
        if t_cross is not None else float("nan")
    i_pre = int(np.searchsorted(rec[:, 0], t_cross)) - 1 \
        if t_cross is not None else None
    P_pre = rec[i_pre, 3] if i_pre is not None else float("nan")
    i_post = int(np.searchsorted(rec[:, 0], t_cross + 1.0)) \
        if t_cross is not None else None
    P_post = rec[i_post, 3] if i_post is not None else float("nan")
    P_max = float(rec[:, 3].max())
    i_Pmax = int(rec[:, 3].argmax())

    print(f"  Picard iterations per EM update: mean "
          f"{np.mean(iters_all):.1f}, max {max(iters_all)}")
    print("  t[s]  T_surf[C]  T_axis[C]    P/L[W/m]  delta_eff[mm]  "
          "mu_r,surf")
    for tt in range(0, int(T_END) + 1):
        i = int(round(tt / EM_DT))
        print(f"  {rec[i,0]:4.0f}  {rec[i,1]:9.1f}  {rec[i,2]:9.1f}  "
              f"{rec[i,3]:11.1f}  {rec[i,4]*1e3:13.3f}  {rec[i,5]:10.1f}")
    print(f"  initial state: P/L = {rec[0,3]:.1f} W/m, mu_r,surf = "
          f"{rec[0,5]:.1f}, delta_eff = {rec[0,4]*1e3:.3f} mm")
    print(f"  P/L peaks at {P_max:.1f} W/m (t = {rec[i_Pmax,0]:.1f} s), "
          f"pre-crossing {P_pre:.1f}, 1 s after crossing {P_post:.1f} "
          f"W/m ({(P_post-P_pre)/P_pre*100:+.1f}%)")
    print(f"  Curie crossing (surface): t = "
          f"{t_cross:.2f} s" if t_cross is not None else
          "  Curie crossing (surface): never")
    print(f"  surface heating rate: 0-0.5 s = {rate_early:.1f} K/s, "
          f"crossing+0.5..1.5 s = {rate_after:.1f} K/s")
    print(f"  final: T_surf = {rec[-1,1]:.1f} C, T_axis = {rec[-1,2]:.1f} "
          f"C, delta_eff = {rec[-1,4]*1e3:.3f} mm (cold "
          f"{rec[0,4]*1e3:.3f} mm, x{rec[-1,4]/rec[0,4]:.1f})")
    print(f"  energy balance: in = {E_in:.6e} J (per slab), enthalpy = "
          f"{enth:.6e} J, err = {bal_err:.3f} %")

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "big_m10_caseB.csv")
    np.savetxt(out_csv,
               np.column_stack((rec[:, 0], rec[:, 1], rec[:, 2],
                                rec[:, 3], rec[:, 4] * 1e3, rec[:, 5])),
               delimiter=",",
               header="t_s,T_surf_C,T_axis_C,P_per_L_W_m,"
                      "delta_eff_mm,mu_r_surf",
               comments="")
    print(f"  time series written to {out_csv}")
    return dict(t_cross=t_cross, rec=rec, E_in=E_in, enth=enth,
                bal_err=bal_err)


def main():
    print("BIG-M10  nonlinear B-H + Curie in the axi eddy framework")
    print(f"  material law (BIG-M4): Bsat = {BSAT} T, low-field mu_mag "
          f"= {MU_MAG0:.0f}, H_REF = {H_REF:.1f} A/m, Tc = {TC:.0f} degC")
    resA = case_A()
    case_B()
    # ---- BIG-M11 benchmark criteria (case A quantities, unchanged):
    # both linear references within 5 % of the Bessel analytic power,
    # Picard iteration converged, frozen-profile self-consistency
    # < 0.1 %.
    lin_errs = {mu: (v[0] - v[3]) / v[3] for mu, v in resA["lin"].items()}
    ok_lin = all(abs(e) < 0.05 for e in lin_errs.values())
    ok_pic = resA["iters"] < 80 and resA["rel_dP"] < 1.0e-3
    ok_frz = resA["frozen_rel"] < 1.0e-3
    print("=" * 72)
    print("VERDICT: linear-ref FEM errors "
          + ", ".join(f"mu_r={mu:g}: {e * 100:+.2f}%"
                      for mu, e in lin_errs.items())
          + f" (criterion < 5%) -> {'PASS' if ok_lin else 'FAIL'}")
    print(f"VERDICT: Picard converged ({resA['iters']} iters, final "
          f"rel dP = {resA['rel_dP']:.2e}) -> "
          f"{'PASS' if ok_pic else 'FAIL'}")
    print(f"VERDICT: frozen-profile rel diff = "
          f"{resA['frozen_rel']:.2e} (criterion < 0.1%) -> "
          f"{'PASS' if ok_frz else 'FAIL'}")
    assert ok_lin, "linear reference error >= 5%"
    assert ok_pic, "Picard iteration did not converge"
    assert ok_frz, "frozen-profile self-consistency >= 0.1%"
    print("BIG-M10 BENCHMARK PASSED")


if __name__ == "__main__":
    main()
