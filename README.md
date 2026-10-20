# CloudFEM

An open-source, cloud-native multiphysics finite-element framework — a
practical step toward a COMSOL-like simulation environment that runs
entirely on commodity Linux cloud machines.

> **Status: v1.1.** 2D triangular P1 finite elements (structured +
> unstructured Delaunay meshes), tetrahedral P1 elements for 3D
> domains (structured box meshes plus unstructured Delaunay meshes
> of spheres, cylinders and L-shaped blocks), steady/transient heat
> conduction (2D and 3D; the 3D
> transient solver uses a consistent capacity matrix with
> theta-method time integration and Dirichlet/Neumann/Robin
> boundary conditions), time-harmonic eddy currents
> (planar + axisymmetric), sequential EM-thermal (induction heating)
> coupling with temperature-dependent conductivity, linear elasticity
> with thermal-stress coupling (2D plane stress/strain and 3D
> tetrahedral, including a transient temperature-history stress
> chain), 3D structural dynamics (consistent/lumped mass
> matrices, modal analysis), nonlinear magnetostatics with B-H
> saturation, and a minimal browser Web UI (draw polygon → mesh →
> solve steady heat → contour plot).
> Validated against analytical solutions. Roadmap below.

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
| **v0.3** Disk heat convergence (unstructured) | Analytical `Q/4k·(R²−r²)` | measured 2.01 / 2.09, qmin > 0.5 |
| **v0.4** Constrained block, uniform heating (plane strain) | Analytical `−EαΔT/((1+ν)(1−2ν))` | rel. err 2.6e-16 |
| **v0.4** Elasticity convergence (manufactured, plane stress) | Theory: P1 → O(h²) | measured 1.94 / 1.97 |
| **v0.4** Disk thermal stress (plane stress) | Analytical `σ_r, σ_θ` | rel. L2 err 2.6 % |
| **v0.5** Nonlinear magnetostatics, manufactured solution | Theory: P1 → O(h²) | measured 1.96 / 1.98 (Newton, 5 iters) |
| **v0.5** C-core inductor saturation curve | Linear at low NI, bends at knee | B(30kA·t) = 1.8 % of linear extrapolation |
| **v0.6** Web UI: rectangle vs analytic linear profile | T(x) = T_hot(1−x/L) | rel. L2 err < 2 % |
| **v0.6** Web UI: L-shape mesh quality + max principle | qmin floor, 0 ≤ T ≤ 100 | qmin > 0.1 (slivers filtered), bounds hold |
| **v0.7** Tetra P1: affine solution on unit cube (linear exactness) | P1 reproduces affine T exactly | L2 err = 1.4e-15 |
| **v0.7** Tetra P1: manufactured sin(πx)sin(πy)sin(πz), f = 3π²T | Theory: P1 → O(h²) | measured 1.72 / 1.86 |
| **v0.7** 3D steel cube, uniform source, 0.1 m | slab bound Q·L²/8k = 25.0 °C | centre rise 11.18 °C < bound |
| **v0.8** 3D uniaxial tension cube (surface traction) | Analytical σ_xx = 100 MPa, u = σx/E | rel. err ≈ 1e-14 (affine-exact) |
| **v0.8** Fully constrained cube, uniform heating ΔT = 50 | Analytical `−EαΔT/(1−2ν)` = −300 MPa (hydrostatic) | rel. err 2.0e-16, shear ≈ 0, u ≈ 0 |
| **v0.8** Free-expansion cube, uniform heating ΔT = 50 | Analytical u = αΔT·x, σ = 0 | rel. err ≈ 1e-13 |
| **v0.8** 3D elasticity convergence (manufactured sin(πx)) | Theory: P1 → O(h²) | measured 1.74 / 1.88 |
| **v0.8** Heated steel cube demo (base clamped), ΔT_max = 222.5 °C | Bound E·α·ΔT_max = 534 MPa | peak von Mises 329.6 MPa < bound |
| **v0.9** Tet mass matrix vs independent Gauss quadrature | Exact P1 integrals `V/20·(1+δᵢⱼ)` | rel. err 3.7e-17 |
| **v0.9** Total mass conservation (consistent + lumped) | Analytical ρ·V, all 3 components | rel. err 3.5e-16 |
| **v0.9** Free cube rigid-body modes (6) + M-orthonormality | λ₁₋₆ = 0, λ₇ > 0, ΦᵀMΦ = I | max ratio 6.1e-16, orth. err 6.7e-16 |
| **v0.9** Cantilever beam f₁ (40×5×5 tet mesh) | Euler-Bernoulli 81.54 Hz | 86.93 Hz (+6.6 %, P1 overstiff, converges from above) |
| **v0.9** Fixed-free rod, 1st longitudinal mode | `(1/4L)√(E/ρ)` = 1261.9 Hz | 1272.0 Hz (+0.80 %) |
| **v1.0** Tet capacity matrix total + SPD | Analytical `ρcₚ·V`, exact P1 integrals | rel. err 1.5e-16, min eig > 0 |
| **v1.0** 1D slab transient (both faces at Ts) | Eigenfunction series, Fo = 0.05–0.3 | max rel. err BE 5.2e-3, CN 8.3e-4 |
| **v1.0** Steady-state limit of transient solver | v0.7 steady solver, same cube + source | rel. diff 9.8e-12 |
| **v1.0** Robin steady state, 1D bar | Analytical `T∞·Bi·(x/L)/(1+Bi)` | rel. diff 1.2e-11 |
| **v1.0** Neumann energy balance (uniform flux, all faces) | `E(t) = E₀ + q·A·t`, mean T(1) = 28 exactly | drift 1.3e-14 (BE + CN) |
| **v1.0** Adiabatic relaxation: energy + max principle | Energy conserved; bounds exact for dt ≥ h²/(6α) | drift ≤ 2.6e-15; undershoot ≤ 3.1 K at dt = h²/(6α)/2 (stated 5 K tol.) |
| **v1.0** Temporal convergence order (manufactured decay) | Theory: BE O(Δt), CN O(Δt²) (step-halving differences) | measured BE 0.91 / 0.96, CN 1.99 / 2.00 |
| **v1.0** Transient chain: uniform source, fully constrained | `−EαΔT(t)/(1−2ν)`, T(t) = T₀ + Qt/ρcₚ exact | stress rel. err ≤ 5.7e-14, von Mises ≈ 0 |
| **v1.0** Quench demo (steel cube, oil, free body) | Peak stress at intermediate t, < E·α·ΔT scale 1980 MPa | peak von Mises 365.2 MPa at t = 9.0 s, decays to 107 MPa |
| **v1.1** Unstructured sphere/cylinder volume convergence | Analytic `4πR³/3`, `πR²H`; faceting → O(h²) | sphere errs 6.4e-2/3.2e-2/1.6e-2, rates 2.00/2.04; cylinder 2.6e-2/1.2e-2/6.4e-3, rates 2.02/1.96 |
| **v1.1** L-block volume (planar faces) | Analytic union volume 5.0 | rel. err 1.8e-16 / 0.0 |
| **v1.1** Unstructured mesh quality (4 test meshes) | Radius-ratio q = 3r/R; floor + sliver fraction | qmin 0.031–0.045, median 0.77–0.80, frac(q<0.05) ≤ 0.11 % |
| **v1.1** Hollow-cylinder radial conduction (unstructured) | Analytical log profile `T = 100·ln(R/r)/ln(R/ri)` | rel. RMS 4.9e-3 / 2.6e-3, rate 1.83 |
| **v1.1** Sphere, uniform source (unstructured) | Analytical parabola `q(R²−r²)/(6k)` | rel. RMS 3.7e-3 / 1.9e-3, rate 1.59 |
| **v1.1** Sphere thermal stress via v0.8 solver (unstructured mesh) | Analytical `−EαΔT/(1−2ν)`, u = 0 | rel. err 2.0e-15, shear ≈ 8e-16 |
| **v1.1** Steel cylinder billet demo (unstructured) | Infinite-cylinder bound `QR²/(4k)` = 62.5 °C | centre rise 48.79 °C < bound |

Run all: `python3 tests/test_validation.py && python3 tests/test_axi_validation.py && python3 tests/test_unstructured.py && python3 tests/test_elastic.py && python3 tests/test_tet.py && python3 tests/test_elastic3d.py && python3 tests/test_modal3d.py && python3 tests/test_transient3d.py && python3 tests/test_unstructured3d.py`

## Layout

```
cloudfem/
  mesh/        triangular + tetrahedral mesh generation, refinement;
               unstructured3d.py Delaunay tet meshes (sphere /
               cylinder / L block, v1.1)
  fem/         P1 assembly (real + complex), implicit-Euler time stepper,
               tet_heat.py 3D transient heat (theta method, v1.0)
  physics/
    heat.py    steady / transient conduction
    em.py      time-harmonic eddy currents (A-formulation), Joule heating
    coupled.py sequential EM → thermal induction heating
    elastic.py linear elasticity (plane stress/strain), P1 vector FEM
    elastic3d.py 3D linear elasticity + thermal stress, tet P1 FEM
    modal3d.py 3D structural dynamics: mass matrices + modal analysis
    thermomech.py sequential thermal → structural (thermal stress, 2D+3D,
               transient 3D chain with peak summary, v1.0)
tests/         analytical validations (must stay green)
examples/      induction_heating.py + output plots, heat_3d.py (v0.7),
               thermal_stress_3d.py (v0.8), modal_beam_3d.py (v0.9),
               transient_thermomech_3d.py (v1.0),
               unstructured_cylinder_3d.py (v1.1)
docs/          roadmap
```

## Roadmap

- [x] v0.2: Axisymmetric EM (ũ=rAφ) + heat, Neumann BCs, σ(T) coupling
- [x] v0.3: Unstructured Delaunay meshing (arbitrary polygons, disks)
- [x] v0.4: Linear elasticity / thermal stress (plane stress+strain,
      von Mises, sequential EM → thermal → structural chain)
- [x] v0.5: Nonlinear B-H curve (2D magnetostatics with saturation,
      Newton solver with line search, C-core inductor demo)
- [x] v0.6: Web UI — browser sketch → Delaunay mesh → steady heat
      solve → contour plot (stdlib-only server, validated: analytic
      rectangle L2 < 2 %, L-shape qmin > 0.1, max principle holds)
- [x] v0.7: 3D tetrahedral P1 elements — box meshes (Kuhn 6-tet
      subdivision), steady heat conduction, manufactured-solution
      convergence 1.72/1.86 (→ 2.0), affine exactness 1.4e-15,
      steel-cube demo with mid-plane slices + 3D scatter
- [x] v0.8: 3D linear elasticity + thermal stress — tet P1 vector FEM
      (body/surface-traction/thermal loads, von Mises), sequential
      3D heat → structural coupling; validated: uniaxial tension
      affine-exact ≈1e-14, constrained heating −EαΔT/(1−2ν) to
      2.0e-16, free expansion ≈1e-13, manufactured-solution
      convergence 1.74/1.88; heated-cube demo (base clamped)
- [x] v0.9: 3D structural dynamics — consistent + lumped tet-P1 mass
      matrices (exact integrals, Gauss-quadrature checked to 3.7e-17,
      total mass conserved to 3.5e-16), modal analysis via
      shift-invert sparse eigensolver at a slightly negative shift
      (free bodies handled: six rigid modes with |λ|/λ₇ = 6.1e-16,
      M-orthonormal shapes); validated: cantilever f₁ +6.6 % vs
      Euler-Bernoulli (P1 overstiff, mesh sweep +29 % → +6.6 %),
      fixed-free rod longitudinal f₁ +0.80 % vs (1/4L)√(E/ρ);
      cantilever modal demo with mode-shape plots
- [x] v1.0: 3D transient heat conduction + transient thermal-stress
      chain (milestone) — consistent tet-P1 heat-capacity matrix
      (exact integrals, total heat ρcₚ·V to 1.5e-16, SPD), theta-method
      time integration (implicit Euler + Crank–Nicolson, both
      A-stable, system matrix LU-factorised once per run) with
      time-constant Dirichlet / Neumann (inward flux) / Robin
      (convection) boundary conditions in fem/tet_heat.py;
      transient_thermal_stress_3d chains a temperature history
      through the v0.8 stress solver and summarises the peak-stress
      instant. Validated: 1D-slab eigenfunction series (BE ≤ 5.2e-3,
      CN ≤ 8.3e-4), steady-state limit vs the v0.7 solver (9.8e-12),
      Robin 1D steady profile (1.2e-11), Neumann energy balance
      (drift 1.3e-14), adiabatic energy conservation (≤ 2.6e-15)
      with the consistent-mass maximum principle characterised
      (exact for dt ≥ h²/(6α); ≤ 3.1 K undershoot at half that step
      on a 100 K range), temporal orders BE 0.91/0.96 and CN
      1.99/2.00 by step-halving differences, and the coupled chain
      against −EαΔT(t)/(1−2ν) with T(t) = T₀ + Qt/ρcₚ exact
      (≤ 5.7e-14). Quench demo: 50 mm steel cube, oil convection,
      free body — peak von Mises 365.2 MPa at t = 9.0 s decaying to
      107 MPa as the cube homogenises; the early consistent-mass
      surface overshoot (~12 K on an 825 K jump) is documented in
      the demo, not hidden
- [x] v1.1: 3D unstructured tetrahedral meshing — breaks the box-only
      restriction of the 3D solvers. mesh/unstructured3d.py follows
      the v0.3 2D recipe in 3D: true-surface sampling (Fibonacci
      lattice on the sphere, ring stacks on the cylinder, face grids
      on the two-box L union with interface samples dropped),
      jittered-grid interior, Delaunay, centroid filter, quality-
      aware Laplacian smoothing, retriangulation, and bounded
      sliver-vertex removal (radius-ratio q < 0.02; boundary vertices
      never removed). Two measured pitfalls are designed around:
      interior points stand 0.35h off the surface (closer, and the
      wall layer fills with paper-thin tetrahedra), and cylinder
      rings share one angular phase (alternating phases form
      near-coplanar "diamond" quadruples with q ~ 0.006). Output is
      a plain TetMesh, so the steady/transient heat, elasticity and
      modal solvers consume it unchanged. Validated: volume
      convergence at rates 2.00/2.04 (sphere) and 2.02/1.96
      (cylinder), L-block volume exact (planar faces), qmin
      0.031-0.045 with median ~0.78 on the test meshes, hollow-
      cylinder log-profile conduction (rel. RMS 4.9e-3 -> 2.6e-3,
      rate 1.83), sphere parabolic source profile (3.7e-3 ->
      1.9e-3, rate 1.59), and the fully constrained sphere heating
      reproduced to 2.0e-15 through the v0.8 stress solver.
      Limitations: curved surfaces are planar-faceted (volume error
      O(h^2), cylinder dominated by the circumferential polygon
      factor), uniform h only (thin walls need h < wall/3), and the
      L block's re-entrant edge is resolved by the centroid test
      alone (O(h) stair-step, no leaks)

## Licence

MIT — see `LICENSE`.
