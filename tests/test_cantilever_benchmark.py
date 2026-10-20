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

Expected behaviour / honesty notes
----------------------------------
P1 constant-strain tetrahedra are overstiff in bending (parasitic
shear), so the FEM deflections converge from below.  At L/H = 10 the
Timoshenko shear share (+0.78 %) partially offsets the remaining
discretisation deficit: the finest level sits -0.77 % from the
Euler-Bernoulli value (the benchmark criterion, |err| < 1 %) and
-1.54 % from the Timoshenko value, which is the fairer measure of
the remaining discretisation error.  Both errors are printed and
asserted; nothing is hidden by the choice of reference.

Measured on the development machine (times are indicative):

    level  mesh (nx, ny, nz)   tets    tip [mm]   err EB    err Tim
    L1     (100, 2, 8) half     9600   0.37563    -6.09 %   -6.82 %
    L1f    (100, 4, 8) full    19200   0.37618    -5.95 %   -6.68 %
    L2     (200, 3, 16) half   57600   0.39314    -1.72 %   -2.48 %
    L3     (320, 3, 27) half  155520   0.39690    -0.77 %   -1.54 %

Checks
------
1. Traction resultant: the assembled load vector of the level-1 half
   model sums to exactly -P/2 in z (traction resultant check).
2. Symmetry cross-check: half vs full model at level 1 agree to < 1 %.
3. Mesh convergence: |err vs EB| decreases monotonically L1 -> L3.
4. Benchmark criterion: |err vs EB| at L3 < 1 % (BIG-M6), and
   |err vs Timoshenko| at L3 < 2.5 % (discretisation-error record).
"""

import time

import numpy as np

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

# (nx, ny, nz) of the half model at each convergence level
LEVELS = ((100, 2, 8), (200, 3, 16), (320, 3, 27))


def solve_beam(nx, ny, nz, half=True):
    """Solve the benchmark on one mesh; return (tip, n_tets, seconds).

    tip is the mean u_z over the free-end face nodes (signed, -z load
    gives a negative value); half=True models y in [0, B/2] with the
    symmetry condition u_y = 0 on ymin.
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
    return tip, mesh.n_elements, dt


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
    tip_h, nt_h, _ = solve_beam(*LEVELS[0], half=True)
    tip_f, nt_f, _ = solve_beam(LEVELS[0][0], 2 * LEVELS[0][1],
                                LEVELS[0][2], half=False)
    rel = abs(tip_h - tip_f) / abs(tip_f)
    print(f"[symmetry] level-1 half tip = {tip_h:.6e} m "
          f"({nt_h} tets), full tip = {tip_f:.6e} m ({nt_f} tets), "
          f"rel diff = {rel:.3e}")
    assert rel < 0.01, "half/full symmetry cross-check failed"

    # mesh convergence of the half model
    rows = []
    for i, (nx, ny, nz) in enumerate(LEVELS):
        tip, nt, dt = solve_beam(nx, ny, nz, half=True)
        err_eb = (abs(tip) - DELTA_EB) / DELTA_EB
        err_tim = (abs(tip) - DELTA_TIM) / DELTA_TIM
        rows.append((nx, ny, nz, nt, tip, err_eb, err_tim))
        print(f"[L{i + 1}] mesh = ({nx}, {ny}, {nz}) half, "
              f"tets = {nt}, solve = {dt:.1f} s, "
              f"tip = {abs(tip):.6e} m, "
              f"err EB = {100.0 * err_eb:+.3f} %, "
              f"err Timoshenko = {100.0 * err_tim:+.3f} %")

    errs = [abs(r[5]) for r in rows]
    assert errs[1] < errs[0] and errs[2] < errs[1], \
        f"tip-deflection error not monotone under refinement: {errs}"
    assert errs[2] < 0.01, \
        f"BIG-M6 criterion failed: finest-level EB error {errs[2]:.3%}"
    assert abs(rows[2][6]) < 0.025, \
        "finest-level Timoshenko error outside the documented 2.5 % band"

    print("ALL CANTILEVER BENCHMARK VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
