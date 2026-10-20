"""BIG-M6 self-check benchmark: cantilever beam static tip deflection.

Benchmark problem
-----------------
Cantilever beam along x, one end clamped, transverse end load:

    L = 0.5 m (x), square cross-section B x H = 0.05 m x 0.05 m (y, z),
    so L/H = 10.  Steel: E = 200 GPa, nu = 0.3.
    Total transverse force P = 1000 N in -z at the free end.

The load is applied as a *uniform traction* t_z = -P/(B*H) on the
xmax face (resultant exactly P): no point-load singularity and no
local dent in the readout.  The xmin face is fully clamped
(ux = uy = uz = 0).  The readout is the mean of u_z over the free-end
face nodes (the cross-section average deflection; it suppresses the
end-face warping that a single-node readout would pick up).

Reference solutions (beam theory, independent of the solver):

    Euler-Bernoulli:  delta_EB = P L^3 / (3 E I),  I = B H^3 / 12
                      = 4.000000e-04 m  (0.4000 mm)
    Timoshenko adds the shear deflection  delta_s = P L / (k G A),
    k = 5/6, G = E / (2 (1 + nu)):  delta_T = 4.031200e-04 m.
    At L/H = 10 the shear share is 0.780 % of delta_EB.

Symmetry half model
-------------------
The exact solution is symmetric about the mid-plane y = B/2, so the
convergence levels solve the half beam y in [0, B/2] with u_y = 0 on
the symmetry plane ymin; the traction value is unchanged (its
resultant on the half face is P/2).  A full-model run at the level-1
resolution cross-checks the half model (same y resolution B/4).

Four levels + Richardson extrapolation (BIG-M6 upgrade)
---------------------------------------------------------
The original three levels used only ny = 2-3 cells across the half
width.  Level L4 raises the cross-section resolution to ny = 5 on the
half model (10 cells across the full width) and nz = 32 through the
height, so every cross-section edge carries >= 5 cells.  L4 keeps
nx = 272 (slightly coarser than L3's 320): the direct solver's memory
on the development machine caps the usable size near ~170 k DOF, and
the volume-mean element size h = (V/N_tets)^(1/3) still decreases
monotonically (4.02 -> 2.21 -> 1.59 -> 1.34 mm).  Cross-section
resolution, not nx, was the documented gap.

Richardson extrapolation fits delta(h) = delta* + C h^p on the three
finest levels (observed order p from the data, not assumed) and
compares delta* with both beam theories.  The L1-L3 window fit is
reported as an error band.  Result (measured, see table): both
windows put the extrapolated limit between the two beam theories
(L2-L4: delta* = 0.39973 mm, p = 2.56; L1-L3: delta* = 0.40104 mm,
p = 1.96; window spread 0.0013 mm).  Extrapolation improves the
Timoshenko comparison of the raw L4 value (-1.29 % -> -0.84 % /
-0.52 %), but the limit does NOT reach the Timoshenko value
0.40312 mm: the residual 0.5-0.8 % gap is larger than the window
spread (0.33 %), so it is a model difference between the 3D
elasticity solution (finite L/H = 10, fully clamped end, uniform
end traction) and the Timoshenko formula (shear coefficient
k = 5/6, 1D assumptions), not remaining mesh error.  The L4
discretisation deficit itself is delta* - delta_L4 ~= 0.5-0.8 %.

Expected behaviour / honesty notes
----------------------------------
P1 constant-strain tetrahedra are overstiff in bending (parasitic
shear), so the FEM deflections converge from below.  At L/H = 10 the
Timoshenko shear share (+0.78 %) partially offsets the remaining
discretisation deficit: the finest level sits -0.52 % from the
Euler-Bernoulli value (the benchmark criterion, |err| < 1 %) and
-1.29 % from the Timoshenko value, which is the fairer measure of
the remaining discretisation error.  Both errors are printed and
asserted; nothing is hidden by the choice of reference.

Measured on the development machine (times are indicative):

    level  mesh (nx, ny, nz)   tets     DOF     tip [mm]   err EB    err Tim
    L1     (100, 2, 8) half     9600     8181   0.37563    -6.09 %   -6.82 %
    L1f    (100, 4, 8) full    19200    13635   0.37618    -5.95 %   -6.68 %
    L2     (200, 3, 16) half   57600    41004   0.39314    -1.72 %   -2.48 %
    L3     (320, 3, 27) half  155520   107856   0.39690    -0.77 %   -1.54 %
    L4     (272, 5, 32) half  261120   162162   0.39791    -0.52 %   -1.29 %

    Richardson (L2-L4 window): observed order p = 2.56,
    delta* = 0.39973 mm (err vs Timoshenko -0.84 %, vs EB -0.07 %);
    L1-L3 window gives delta* = 0.40104 mm (spread 0.0013 mm).

Checks
------
1. Traction resultant: the assembled load vector of the level-1 half
   model sums to exactly -P/2 in z (traction resultant check).
2. Symmetry cross-check: half vs full model at level 1 agree to < 1 %.
3. Mesh convergence: |err vs EB| decreases monotonically L1 -> L4.
4. Benchmark criterion: |err vs EB| at L4 < 1 % (BIG-M6), and
   |err vs Timoshenko| at L4 < 2.5 % (discretisation-error record).
5. Richardson: extrapolated delta* is closer to the Timoshenko
   value than the raw L4 deflection, and within 1 % of it.
"""

import time

import numpy as np
from scipy.optimize import brentq

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.physics.elastic3d import (
    assemble_elasticity_3d, solve_elasticity_3d)

# benchmark setup
E_STEEL = 200e9
NU_STEEL = 0.3
L_BEAM = 0.5            # m, along x
B_BEAM = 0.05           # m, width (y)
H_BEAM = 0.05           # m, height (z)
P_LOAD = 1000.0         # N, resultant transverse force at the free end

I_BEAM = B_BEAM * H_BEAM ** 3 / 12.0
A_BEAM = B_BEAM * H_BEAM
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))
DELTA_EB = P_LOAD * L_BEAM ** 3 / (3.0 * E_STEEL * I_BEAM)
DELTA_TIM = DELTA_EB + P_LOAD * L_BEAM / ((5.0 / 6.0) * G_STEEL * A_BEAM)

# (nx, ny, nz) of the half model at each convergence level; L4 raises
# the cross-section resolution to ny = 5 / nz = 32 (>= 5 cells on every
# cross-section edge of the half model, 10 across the full width)
LEVELS = ((100, 2, 8), (200, 3, 16), (320, 3, 27), (272, 5, 32))


def solve_beam(nx, ny, nz, half=True):
    """Solve the benchmark on one mesh.

    Return (tip, n_tets, n_dof, seconds).  tip is the mean u_z over
    the free-end face nodes (signed, -z load gives a negative value);
    half=True models y in [0, B/2] with the symmetry condition
    u_y = 0 on ymin.
    """
    ytop = B_BEAM / 2.0 if half else B_BEAM
    mesh = box_mesh(nx, ny, nz, xlim=(0.0, L_BEAM), ylim=(0.0, ytop),
                    zlim=(0.0, H_BEAM))
    bc = {"xmin": (0.0, 0.0, 0.0)}
    if half:
        bc["ymin"] = (None, 0.0, None)
    t0 = time.time()
    ux, uy, uz = solve_elasticity_3d(
        mesh, E_STEEL, NU_STEEL, bc=bc,
        traction={"xmax": (0.0, 0.0, -P_LOAD / A_BEAM)})
    dt = time.time() - t0
    tip = float(uz[mesh.boundaries["xmax"]].mean())
    return tip, mesh.n_elements, 3 * mesh.nodes.shape[0], dt


def test_traction_resultant():
    """Assembled z-load of the level-1 half model must total -P/2."""
    nx, ny, nz = LEVELS[0]
    mesh = box_mesh(nx, ny, nz, xlim=(0.0, L_BEAM),
                    ylim=(0.0, B_BEAM / 2.0), zlim=(0.0, H_BEAM))
    A, rhs = assemble_elasticity_3d(
        mesh, E_STEEL, NU_STEEL,
        traction={"xmax": (0.0, 0.0, -P_LOAD / A_BEAM)})
    fz = float(rhs[2::3].sum())
    fx = float(rhs[0::3].sum())
    fy = float(rhs[1::3].sum())
    return fz, fx, fy


def richardson(d1, d2, d3, h1, h2, h3):
    """Three-level Richardson fit delta(h) = delta* + C h^p.

    The observed order p is solved from the three deflections
    themselves (no assumed order); returns (p, delta*).
    """
    ratio = (d1 - d2) / (d2 - d3)

    def f(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - ratio
    p = brentq(f, 0.05, 12.0)
    dstar = d3 + (d3 - d2) / ((h2 / h3) ** p - 1.0)
    return p, dstar


def main():
    print(f"[setup] L = {L_BEAM} m, B x H = {B_BEAM} x {H_BEAM} m "
          f"(L/H = {L_BEAM / H_BEAM:.0f}), E = {E_STEEL:.3e}, "
          f"nu = {NU_STEEL}, P = {P_LOAD:.0f} N")
    print(f"[reference] delta_EB = {DELTA_EB:.6e} m, "
          f"delta_Timoshenko = {DELTA_TIM:.6e} m "
          f"(shear share {100.0 * (DELTA_TIM - DELTA_EB) / DELTA_EB:.3f} %)")

    fz, fx, fy = test_traction_resultant()
    print(f"[traction] assembled resultant (half model): "
          f"Fx = {fx:.3e}, Fy = {fy:.3e}, Fz = {fz:.6e} N "
          f"(expect -P/2 = {-P_LOAD / 2.0:.6e})")
    assert abs(fz + P_LOAD / 2.0) < 1e-9 * P_LOAD, \
        "traction resultant in z is not -P/2"
    assert abs(fx) < 1e-9 * P_LOAD and abs(fy) < 1e-9 * P_LOAD, \
        "traction must have no x/y resultant"

    # symmetry cross-check at level 1 (same y resolution B/4)
    tip_h, nt_h, _, _ = solve_beam(*LEVELS[0], half=True)
    tip_f, nt_f, _, _ = solve_beam(LEVELS[0][0], 2 * LEVELS[0][1],
                                   LEVELS[0][2], half=False)
    rel = abs(tip_h - tip_f) / abs(tip_f)
    print(f"[symmetry] level-1 half tip = {tip_h:.6e} m "
          f"({nt_h} tets), full tip = {tip_f:.6e} m ({nt_f} tets), "
          f"rel diff = {rel:.3e}")
    assert rel < 0.01, "half/full symmetry cross-check failed"

    # mesh convergence of the half model (four levels)
    rows = []
    vol_half = L_BEAM * (B_BEAM / 2.0) * H_BEAM
    for i, (nx, ny, nz) in enumerate(LEVELS):
        tip, nt, ndof, dt = solve_beam(nx, ny, nz, half=True)
        err_eb = (abs(tip) - DELTA_EB) / DELTA_EB
        err_tim = (abs(tip) - DELTA_TIM) / DELTA_TIM
        h = (vol_half / nt) ** (1.0 / 3.0)
        rows.append((nx, ny, nz, nt, ndof, h, tip, err_eb, err_tim))
        print(f"[L{i + 1}] mesh = ({nx}, {ny}, {nz}) half, "
              f"tets = {nt}, DOF = {ndof}, h = {1e3 * h:.3f} mm, "
              f"solve = {dt:.1f} s, tip = {abs(tip):.6e} m, "
              f"err EB = {100.0 * err_eb:+.3f} %, "
              f"err Timoshenko = {100.0 * err_tim:+.3f} %")

    errs = [abs(r[7]) for r in rows]
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), \
        f"tip-deflection error not monotone under refinement: {errs}"
    assert errs[-1] < 0.01, \
        f"BIG-M6 criterion failed: finest-level EB error {errs[-1]:.3%}"
    assert abs(rows[-1][8]) < 0.025, \
        "finest-level Timoshenko error outside the documented 2.5 % band"

    # Richardson extrapolation on the three finest levels; the coarser
    # L1-L3 window is the error-band marker
    d = [abs(r[6]) for r in rows]
    hh = [r[5] for r in rows]
    p_fin, dstar_fin = richardson(d[1], d[2], d[3], hh[1], hh[2], hh[3])
    p_old, dstar_old = richardson(d[0], d[1], d[2], hh[0], hh[1], hh[2])
    print(f"[richardson] L2-L4 window: observed order p = {p_fin:.3f}, "
          f"delta* = {dstar_fin:.6e} m "
          f"(err EB = {100.0 * (dstar_fin - DELTA_EB) / DELTA_EB:+.3f} %, "
          f"err Timoshenko = "
          f"{100.0 * (dstar_fin - DELTA_TIM) / DELTA_TIM:+.3f} %)")
    print(f"[richardson] L1-L3 window: p = {p_old:.3f}, "
          f"delta* = {dstar_old:.6e} m "
          f"(window spread = {1e3 * abs(dstar_fin - dstar_old):.5f} mm)")
    assert abs(dstar_fin - DELTA_TIM) < abs(d[3] - DELTA_TIM), \
        "Richardson extrapolation does not improve on the raw L4 value"
    assert abs(dstar_fin - DELTA_TIM) / DELTA_TIM < 0.01, \
        "extrapolated deflection not within 1 % of Timoshenko"

    print("ALL CANTILEVER BENCHMARK VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
