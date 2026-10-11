# CloudFEM

An open-source, cloud-native multiphysics finite-element framework — a
practical step toward a COMSOL-like simulation environment that runs
entirely on commodity Linux cloud machines.

> **Status: v0.2.** 2D triangular P1 finite elements, steady/transient
> heat conduction, time-harmonic eddy currents (planar + axisymmetric),
> sequential EM-thermal (induction heating) coupling with
> temperature-dependent conductivity. Validated against analytical
> solutions. Roadmap below.

## Why this exists

Commercial multiphysics software (COMSOL) is expensive and tied to a
local workstation. CloudFEM is built to run on a plain cloud VM with
only NumPy/SciPy/Matplotlib — no proprietary dependencies, no licence
server — and to grow module by module toward broader multiphysics
coverage.

## Quick start

```bash
git clone <repo-url>
cd cloudfem
export PYTHONPATH=.
python3 tests/test_validation.py        # analytical validations
python3 examples/induction_heating.py   # showcase: steel billet in coil
```

Requirements: Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`.

## What is validated (not just "it ran")

| Test | Reference | Result |
|---|---|---|
| Steady heat, 1D slab + uniform source | Analytical `T = Q/2k·x(L−x)` | rel. L2 error ≈ 2e-14 |
| Skin depth, semi-infinite conductor @10 kHz | Analytical `A₀e^{−(1+i)x/δ}` | L2 ≈ 7e-4, phase err < 0.015 rad, Q-profile err < 1e-4 |
| Mesh convergence (manufactured solution) | Theory: P1 → O(h²) | measured 1.96 / 1.98 |
| Planar induction heating energy balance | `∫ρcₚΔT = P·t` | 0.00 % error |
| **v0.2** Axisymmetric cylinder in axial AC field | Bessel `H₀·J₀(αr)/J₀(αa)` | L2 ≈ 4e-3 |
| **v0.2** Axisymmetric heat convergence | Theory: P1 → O(h²) | measured 2.05 / 2.05 |

Run all: `python3 tests/test_validation.py && python3 tests/test_axi_validation.py`

## Layout

```
cloudfem/
  mesh/        triangular mesh generation, refinement
  fem/         P1 assembly (real + complex), implicit-Euler time stepper
  physics/
    heat.py    steady / transient conduction
    em.py      time-harmonic eddy currents (A-formulation), Joule heating
    coupled.py sequential EM → thermal induction heating
tests/         analytical validations (must stay green)
examples/      induction_heating.py + output plots
docs/          roadmap
```

## Roadmap

- [x] v0.2: Axisymmetric EM (ũ=rAφ) + heat, Neumann BCs, σ(T) coupling
- [ ] Unstructured Delaunay meshing of arbitrary polygons (CAD import: DXF/STEP)
- [ ] Nonlinear B–H curve (magnetostatics with saturation)
- [ ] Linear elasticity / thermal stress
- [ ] Web UI: geometry sketch → mesh → solve → plots in the browser
- [ ] 3D (tetrahedral) support

## Licence

MIT — see `LICENSE`.
