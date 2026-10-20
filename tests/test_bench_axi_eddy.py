"""BIG-M8: CloudFEM axisymmetric eddy-current benchmark.

Configuration
-------------
Solid conducting cylinder (radius a) coaxial inside an infinitely long
solenoid producing a uniform axial field. The tangential field at the
cylinder surface is fixed by the solenoid surface current, so the exact
boundary condition for the workpiece problem is

    H_z(a) = H0        (tangential H continuous, no surface current
                        on the workpiece; bore field is uniform H0)

This is the canonical "cylinder in uniform axial AC field" problem with
closed-form solution (phasor convention e^{+i omega t}):

    H_z(r) = H0 * J0(k r) / J0(k a),      k = (1 - i) / delta
    E_phi(r) = -(1/sigma) dH_z/dr = (H0 * k / sigma) * J1(k r) / J0(k a)
    delta = sqrt(2 / (omega * mu * sigma))

Analytic reference power per unit length, three independent routes:
  1. Volume integral  P/L = int_0^a (sigma/2) |E_phi|^2 2 pi r dr
     (composite Gauss-Legendre on the Bessel profile; semi-analytic,
     fully independent of the FEM code).
  2. Poynting flux at r = a:  P/L = -pi * a * Re{E_phi(a) * conj(H0)}.
  3. Kelvin-function closed form with x = sqrt(2) a / delta:
     P/L = (pi a H0^2 sqrt(2) / (sigma delta))
           * (ber ber' + bei bei') / (ber^2 + bei^2)   at x.
The script asserts the three routes agree to <1e-8 relative before the
FEM numbers are compared against them.

FEM side: CloudFEM axisymmetric solver (utilde = r * A_phi) on an r-z
slab of height Lz. The exact solution is z-invariant and satisfies
d(utilde)/dz = 0 on the slab faces, so the slab reproduces the infinite
cylinder exactly; P/L = total slab power / Lz. The solver's Neumann
condition on r = a with g = 2*pi*H0 is exactly the weak-form statement
of H_z(a) = H0 (the same excitation used in tests/test_axi_validation.py);
Dirichlet utilde = 0 on the axis. Meshes are tensor grids with geometric
radial grading toward the surface (growth ratio 1.18), nz = 4.

Parameters: a = 20 mm, sigma = 2.0e6 S/m, mu = mu0 (mu_r = 1),
f = 2000 Hz  ->  delta = 7.958 mm, a/delta = 2.513, H0 = 5000 A/m.

Run:  python3 tests/test_bench_axi_eddy.py   (from the repo root)
Integrated into the CloudFEM suite by BIG-M11; physics unchanged
from the standalone BIG-M8 script.
"""

import os
import sys

import numpy as np
from scipy.special import jv, ber, bei, berp, beip

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from cloudfem.mesh.tri_mesh import TriMesh
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, total_joule_power_axi, MU0)

# ---------------------------------------------------------------- parameters
A = 0.02            # cylinder radius [m]
SIGMA = 2.0e6       # conductivity [S/m], constant
MU = MU0            # mu_r = 1
FREQ = 2000.0       # [Hz]
H0 = 5000.0         # solenoid field amplitude [A/m]
LZ = A              # r-z slab height [m] (solution is z-invariant)
NZ = 4              # axial divisions (field does not vary with z)
GROWTH = 1.18       # radial geometric growth ratio, surface -> axis
LEVELS_HS = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 24.0,
             1.0 / 48.0)  # target h_surf / delta

OMEGA = 2.0 * np.pi * FREQ
DELTA = np.sqrt(2.0 / (OMEGA * MU * SIGMA))
K = (1.0 - 1.0j) / DELTA


# ------------------------------------------------------- analytic references
def e_phi(r):
    """Analytic azimuthal electric field phasor [V/m]."""
    return (H0 * K / SIGMA) * jv(1, K * r) / jv(0, K * A)


def power_volume_integral():
    """Route 1: composite Gauss-Legendre integral of (sigma/2)|E|^2."""
    xg, wg = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, A, 801)
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        total += np.sum(w * 0.5 * SIGMA * np.abs(e_phi(r)) ** 2
                        * 2.0 * np.pi * r)
    return float(total)


def power_poynting():
    """Route 2: time-averaged Poynting flux through r = a."""
    return float(-np.pi * A * np.real(e_phi(A) * np.conj(H0)))


def power_kelvin():
    """Route 3: Kelvin ber/bei closed form at x = sqrt(2) a / delta."""
    x = np.sqrt(2.0) * A / DELTA
    num = ber(x) * berp(x) + bei(x) * beip(x)
    den = ber(x) ** 2 + bei(x) ** 2
    return float(np.pi * A * H0 ** 2 * np.sqrt(2.0) / (SIGMA * DELTA)
                 * num / den)


# ---------------------------------------------------------------- graded mesh
def graded_slab_mesh(h_surf):
    """Tensor r-z mesh, geometric radial grading toward r = a.

    Layer thicknesses from the surface inward form an exact geometric
    progression h_surf, h_surf*q, h_surf*q^2, ... (q = GROWTH) summing
    exactly to A; h_surf is the largest value <= the requested target
    that makes the progression close exactly. Boundary tags follow
    rect_mesh: left (r=0), right (r=a), bottom (z=0), top (z=Lz).
    """
    q = GROWTH
    # smallest layer count m whose progression reaches A with h = target
    m = 1
    while h_surf * (q ** m - 1.0) / (q - 1.0) < A:
        m += 1
    h0 = A * (q - 1.0) / (q ** m - 1.0)        # exact first-layer thickness
    thick = h0 * q ** np.arange(m)
    edges_from_surf = np.concatenate(([0.0], np.cumsum(thick)))
    r_nodes = A - edges_from_surf[::-1]        # ascending: 0 .. A
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
    h_min = float(thick[0])
    return TriMesh(nodes, np.array(elements), boundaries), nr, h_min


# ------------------------------------------------------------------ FEM solve
def fem_power_per_length(h_surf):
    mesh, nr, h_min = graded_slab_mesh(h_surf)
    utilde = solve_eddy_current_axi(
        mesh, nu=1.0 / MU, sigma=SIGMA, omega=OMEGA,
        bc={"left": 0.0}, neumann={"right": 2.0 * np.pi * H0})
    p_total = total_joule_power_axi(mesh, utilde, SIGMA, OMEGA)
    return p_total / LZ, mesh.n_elements, mesh.n_nodes, nr, h_min


def main():
    print("BIG-M8  axisymmetric cylinder in uniform axial AC field")
    print(f"  a = {A*1e3:.1f} mm, sigma = {SIGMA:.3e} S/m, mu_r = 1, "
          f"f = {FREQ:.0f} Hz, H0 = {H0:.0f} A/m")
    print(f"  delta = {DELTA*1e3:.3f} mm, a/delta = {A/DELTA:.3f}")

    p_vol = power_volume_integral()
    p_poy = power_poynting()
    p_kel = power_kelvin()
    d1 = abs(p_poy - p_vol) / p_vol
    d2 = abs(p_kel - p_vol) / p_vol
    print(f"  analytic P/L: volume integral = {p_vol:.6f} W/m, "
          f"Poynting = {p_poy:.6f} (rel diff {d1:.2e}), "
          f"Kelvin = {p_kel:.6f} (rel diff {d2:.2e})")
    assert d1 < 1e-8 and d2 < 1e-8, "analytic routes disagree"
    p_ref = p_vol

    print("  level  h_surf/delta   nr  elements   nodes   "
          "P_FEM [W/m]   rel err")
    rows = []
    for lev, frac in enumerate(LEVELS_HS, start=1):
        p_fem, ne, nn, nr, h_min = fem_power_per_length(frac * DELTA)
        err = (p_fem - p_ref) / p_ref
        rows.append((lev, h_min / DELTA, nr, ne, nn, p_fem, err))
        print(f"  {lev:5d}  {h_min/DELTA:12.4f}  {nr:3d}  {ne:8d}  "
              f"{nn:6d}   {p_fem:11.4f}   {err*100:+.3f}%")

    errs = [abs(r[6]) for r in rows]
    rates = [np.log(errs[i] / errs[i + 1]) /
             np.log(rows[i][1] / rows[i + 1][1]) for i in range(len(rows) - 1)]
    print(f"  observed convergence rates (in h_surf): "
          f"{[f'{r:.2f}' for r in rates]}")
    print(f"  finest-level error = {rows[-1][6]*100:+.3f}%  "
          f"(criterion: |err| < 5%)")
    assert errs[-1] < 0.05, "finest mesh misses the 5% criterion"
    print("BIG-M8 BENCHMARK PASSED")


if __name__ == "__main__":
    main()
