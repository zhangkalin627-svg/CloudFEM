# CloudFEM

An open-source, cloud-native multiphysics finite-element framework — a
practical step toward a COMSOL-like simulation environment that runs
entirely on commodity Linux cloud machines.

> **Status: v1.9.** 2D triangular P1 finite elements (structured +
> unstructured Delaunay meshes), tetrahedral P1 elements for 3D
> domains (structured box meshes plus unstructured Delaunay meshes
> of spheres, cylinders and L-shaped blocks), plus second-order
> Tet10 (10-node) tetrahedral elements for steady/transient heat
> conduction (v1.6: Neumann/Robin boundary terms), 3D linear
> elasticity (v1.5; higher-order accuracy
> at equal mesh size — see the roadmap entry) and 3D time-harmonic
> eddy currents (v1.7; nodal A-V formulation) plus 3D transient
> eddy currents and coil terminal-inductance extraction (v1.8;
> theta method on the same nodal A-V solver) plus 3D nonlinear
> B-H (saturation) phasor eddy currents (v1.9; secant-mu Picard
> iteration on the same solver), steady/transient heat
> conduction (2D and 3D; the 3D
> transient solver uses a consistent capacity matrix with
> theta-method time integration and Dirichlet/Neumann/Robin
> boundary conditions), time-harmonic eddy currents
> (planar + axisymmetric in 2D; 3D with lowest-order Nedelec edge
> elements in an A-V formulation), sequential EM-thermal (induction heating)
> coupling with temperature-dependent conductivity (in 3D the full
> EM → transient-thermal → thermal-stress flagship chain with
> explicit sigma(T) updates between EM re-solves, v1.4), linear elasticity
> with thermal-stress coupling (2D plane stress/strain and 3D
> tetrahedral, including a transient temperature-history stress
> chain), 3D structural dynamics (consistent/lumped mass
> matrices, modal analysis, and Newmark transient time integration
> with Rayleigh damping plus a modal-superposition solver),
> nonlinear magnetostatics with B-H
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
| **v1.2** SDOF step + harmonic response (Newmark core, 1×1) | Closed form `(F₀/k)(1−cos ωt)`, harmonic analogue | step err 9.1e-4 @ Δt = T/200, harmonic err 9.9e-5 @ T/400 |
| **v1.2** Temporal order, average acceleration | Theory: O(Δt²) | measured 1.997 (Δt = T/25…T/400) |
| **v1.2** Linear-acceleration (β = 1/6) stability limit | Theory: stable iff `ωΔt ≤ 2√3` | amplitude 1.000 at ωΔt = 3.0 and 2√3, ×2.8e7 growth at 4.0; average-acceleration bounded at 4.0 |
| **v1.2** Cantilever step response vs exact modal history | 12 v0.9 modes in closed form + static (mode-acceleration) correction | direct Newmark RMS rel 6.9e-4, max 2.1e-3 @ Δt = T₁/400; modal superposition RMS 8.4e-4 |
| **v1.2** Mesh energy conservation (mode-1 IC, undamped) | `½vᵀMv + ½uᵀKu` constant | drift 2.2e-12 over 5 periods (round-off level) |
| **v1.2** Static limit, Rayleigh damping ζ = 0.10 at ω₁, ω₂ | v0.8 static solve `K⁻¹F` | rel err 1.4e-4 at t = 8/(ζω₁), 637 steps |
| **v1.2** Cantilever step-load demo (1 kN, undamped) | Overshoot → 2× static deflection | peak 0.299 mm vs 2× static 0.307 mm; period within 2 % of T₁ |
| **v1.3** Whitney edge elements: discrete exact sequence | `curl(grad φ) = 0` for any nodal φ | max \|B\| ~ 1e-15 (round-off level) |
| **v1.3** Conducting sphere in uniform AC field, a/δ = 0.50 (160 Hz) | Spherical-Bessel analytic loss / centre field | P rel. err −7.9 %, centre-field err +0.02 % (h = 0.40 a) |
| **v1.3** Conducting sphere, a/δ = 3.02 (5.76 kHz) | Spherical-Bessel analytic loss / centre field | P rel. err +13.7 %, centre-field err +9.3 % (skin layer h/δ ≈ 1.2) |
| **v1.3** Cylinder in axial AC field, 100 Hz (R/δ = 0.4) | v0.2 axisymmetric solver, same R/σ/f | total-loss rel. diff −7.2 % (quasi-static tier) |
| **v1.3** Cylinder at 2.5 kHz (R/δ = 2, L/D = 2) | v0.2 axisymmetric (infinite cylinder) | total-loss rel. diff +36.8 % — finite-length end effect + h ≈ δ skin resolution, documented in tests/test_em3d.py |
| **v1.3** Low-frequency limits on the sphere | Theory: P ∝ f², P ∝ σ (δ ≫ a) | P(160)/P(40) = 15.1-16.0 (f² → 16), P(σ)/P(σ/10) = 9.88-9.98 (→ 10), ranges over two runs |
| **v1.3** Gauge regularisation sensitivity | Loss at gauge_eps = 1e-5 vs 1e-7 | rel. change <= 7.7e-07 (two runs: 2.2e-16, 7.7e-07) |
| **v1.4** 3D induction chain energy account | EM energy accumulated over the run vs discrete enthalpy gain of the adiabatic billet (an algebraic identity of the discrete chain when the transfer is conservative) | rel. err 3.3e-16 |
| **v1.4** sigma(T) feedback, hot point | Surface ~340 C: frozen-sigma peak ΔT 382 K vs sigma(T) 322 K (hot skin turns transparent, deposition deepens); power 11.53 → 11.41 kW across the turnover | measured in tests/test_induction3d.py |
| **v1.4** 3D chain vs 2D axisymmetric chain | Same billet and sigma(T) law, independent solvers: cold-power ratio 1.75 (3D end effect + skin resolution — the v1.3 band); mean-rise ratio 1.79 vs 1.82 from the power/volume expectation | −1.4 % deviation |
| **v1.5** Tet10 heat convergence, manufactured sin·sin·sin | Least-squares orders over n = 2, 4, 6, 8 cells | L2 order 2.90 (expect 3), H1 order 1.86 (expect 2); Tet4 on the same grids: 1.61 / 0.84 (rising to 2 / 1) |
| **v1.5** Tet10 patch tests | Quadratic heat/elastic fields with compensating source/body force on an unstructured sphere mesh | Tet10 exact to ~1e-15 (heat) / ~1e-19 (elasticity); Tet4 2.3e-02 / 8.8e-06 |
| **v1.5** Tet10 cantilever (BIG-M6 geometry, full model) | Euler-Bernoulli / Timoshenko tip deflection | (50, 2, 4) mesh, 2400 tets: −0.033 % vs EB, −0.81 % vs Timoshenko; Tet4 (100, 4, 8) at the same 4545 nodes: −5.95 % vs EB |
| **v1.5** Tet10 transient heat | Cube cooling vs product-of-slabs series; adiabatic heat conservation; temporal orders (v1.0 protocol) | centre err −2.2e-03; drift 6.4e-16; BE 0.91/0.95, CN 1.94/2.38 |
| **v1.6** Tet10 transient Neumann/Robin BCs | Adiabatic box + one-face heat flux: energy-balance drift; slab vs analytic (2/3)x Robin steady profile; max-principle spot check | Neumann drift 3.553e-15 / 1.561e-15 / 1.295e-15 (const / time-varying / space-varying); Robin steady inf-err 2.530e-08 (Tet4 2.404e-08); CN sub-threshold overshoot +2.240e-03 as in v1.0 |
| **v1.7** Tet10 nodal A-V 3D time-harmonic eddy current | Manufactured-solution convergence; conducting sphere vs spherical-Bessel analytic loss; D1 cylinder vs axisymmetric reference; power reconciliation | MMS L2(A) rel err 2.362e-01 / 4.102e-02 / 2.547e-02, orders 2.53 / 2.14 (below scalar-P2 3.0 — known nodal curl-curl suboptimality); sphere P err −13.61 % (a/δ = 0.5) / −3.04 % (a/δ = 3.02 skin-dominated); D1 cylinder (h/δ = 0.22) 3D vs axi −12.25 % (< 15 % criterion); power closure 5.59e-03 / 2.55e-03, in tests/test_tet10_em.py |
| **v1.8** Tet10 nodal A-V 3D transient eddy current + coil terminal inductance | DC-step long-time limit vs low-frequency phasor; step-halving temporal orders; sine (CN) phasor vs v1.7 phasor solver; inductance energy route vs circuit (Z) route; D1 coil L vs EFD analytic formula | DC-step long-time limit rel. diff 4.708e-08; sine CN phasor rel. L2 diff 0.036 % (step-halving orders: BE 1.78 on the step problem, CN 2.005 on smooth sine); L energy-vs-Z routes rel. diff 2.98 % (< 5 % criterion); D1 coil L vs EFD analytic ratio 1.501 (stated 0.3–3 band); full regression `run_all` 6/6 PASS + BIG-D3 SKIP, rc = 0, in tests/test_tet10_em_transient.py + tests/test_inductance.py |
| **v1.9** Tet10 nodal A-V 3D nonlinear B-H (saturation) phasor eddy current | Linear limit (mu_mag0=0 / iron=None) vs the v1.7 linear solver; saturation bracketing between constant mu_r=1 and mu_r=201 linear solves; Picard convergence history; BIG-M10 axi nonlinear P/L order-of-magnitude check (stated non-tight) | linear-limit a-field rel. diff 0.000e+00 (bitwise, both paths); nonlinear P 468910.5 W strictly between 150757.5 W (mu_r=1) and 531896.3 W (mu_r=201); iron max|B| 2.718 T vs 4.425 T linear extrapolation; Picard 40 iters, rel_dB 1.62e-02 peak -> 9.53e-04 < 1e-3; M10 ratio 0.372 (stated 0.2–5 band); full regression `run_all` 6/6 PASS + BIG-D3 SKIP, rc = 0, in tests/test_tet10_em_nonlinear.py |
| **BIG-M6** Cantilever beam static tip deflection (L/H = 10, end traction, symmetry half model, 4 mesh levels + Richardson extrapolation) | Euler–Bernoulli `PL³/(3EI)` = 0.4000 mm | −6.09 % / −1.72 % / −0.77 % / −0.52 % (finest < 1 % criterion); vs Timoshenko −1.29 % at finest (shear share 0.78 %); Richardson limit 0.3997–0.4010 mm — between EB and Timoshenko, residual 0.5–0.8 % gap is a 3D-vs-beam model difference, not mesh error, in tests/test_cantilever_benchmark.py |
| **BIG-M7** 1-D rod thermal stress (real heat -> thermal-stress chain on one tet mesh, 4 mesh levels) | Closed form `sigma = -E alpha <dT>` = -240.00 MPa | slab-mean sigma_xx err 3.233 % / 2.175 % / 1.948 % / 1.664 % (finest < 2 % criterion); end reactions +/-24.40 kN vs 24.00 kN analytic, in tests/test_bench_thermal_stress.py |
| **BIG-M8** Axisymmetric eddy current: solid cylinder in uniform axial AC field (a/delta = 2.51, 5 mesh levels) | Bessel closed form, three independent routes agree < 1e-8: P/L = 79.8105 W/m | FEM P/L err +4.894 % -> +0.520 % (finest < 5 % criterion), in tests/test_bench_axi_eddy.py |
| **BIG-M9** Full induction-heating chain (axi EM -> transient heat, sigma(T) feedback, 20 mm bar, 60 s) | Kelvin closed form, cold state: P/L = 13114 W/m; discrete energy account | t = 0 power err +0.84 % (< 2 % criterion); energy-balance residual -0.10 % (< 1 %); surface 706.8 C at 60 s, in tests/test_bench_induction_chain.py |
| **BIG-M10** Nonlinear B-H + Curie point in the axi eddy framework (Picard iteration on mu_r(|H|, T), BIG-M4 material law) | Linear references vs Bessel analytic power; frozen-profile self-consistency | lin-ref err +0.51 % (mu_r = 1) / +3.05 % (mu_r = 200), both < 5 %; nonlinear P = 10112 W/m (-23.3 % vs the mu_r = 200 linear case); frozen-profile rel diff 3.8e-05 (< 0.1 %); surface Curie crossing at 2.15 s under constant H0, in tests/test_bench_nonlinear_bh.py |
| **BIG-D2** Cylinder skin effect at 50 kHz vs the classical planar formula (mu_r = 1 main, mu_r = 200 contrast) | Bessel/Kelvin analytic P/L, three routes agree ~1e-14 | FEM P/L err +0.515 % (mu_r = 1) / +0.632 % (mu_r = 200), finest < 1 % criterion; delta_eff err +1.31 %, in tests/test_bench_skin50k.py |
| **BIG-M11** Benchmark run-all driver | - | `python3 tests/run_all_benchmarks.py` runs BIG-M6..M10 + BIG-D2 in sequence and prints a per-benchmark error / PASS-FAIL summary (BIG-D3 skipped: external realworld_sim dependency) |

Run all: `python3 tests/test_validation.py && python3 tests/test_axi_validation.py && python3 tests/test_unstructured.py && python3 tests/test_elastic.py && python3 tests/test_tet.py && python3 tests/test_elastic3d.py && python3 tests/test_modal3d.py && python3 tests/test_transient3d.py && python3 tests/test_unstructured3d.py && python3 tests/test_dynamics3d.py && python3 tests/test_em3d.py && python3 tests/test_induction3d.py && python3 tests/test_cantilever_benchmark.py && python3 tests/test_bench_thermal_stress.py && python3 tests/test_bench_axi_eddy.py && python3 tests/test_bench_induction_chain.py && python3 tests/test_bench_nonlinear_bh.py && python3 tests/test_bench_skin50k.py && python3 tests/test_tet10.py`

Benchmarks only (BIG-M6..M10 + BIG-D2, one command with a summary table): `python3 tests/run_all_benchmarks.py`

## Layout

```
cloudfem/
  mesh/        triangular + tetrahedral mesh generation, refinement;
               unstructured3d.py Delaunay tet meshes (sphere /
               cylinder / L block, v1.1; sphere / cylinder embedded
               in an air box, v1.3),
               tet10_mesh.py Tet10 meshes: Tet4 -> Tet10 edge-midpoint
               converter + box meshes (v1.5)
  fem/         P1 assembly (real + complex), implicit-Euler time stepper,
               tet_heat.py 3D transient heat (theta method, v1.0),
               tet_edge.py lowest-order Nedelec (Whitney) edge
               elements on tetrahedra (v1.3),
               tet10.py Tet10 (quadratic) heat conduction: steady +
               transient (theta method), exact mass matrix (v1.5)
  physics/
    heat.py    steady / transient conduction
    em.py      time-harmonic eddy currents (A-formulation), Joule heating
    em3d.py    3D time-harmonic eddy currents: edge-element A-V
               formulation, background-field / current excitation,
               Joule heating for the EM → thermal chain (v1.3)
    induction3d.py 3D induction-heating flagship chain: EM (v1.3)
               → transient heat (v1.0) → thermal stress (v0.8)
               with explicit sigma(T) updates and a discrete
               energy account (v1.4)
    coupled.py sequential EM → thermal induction heating
    elastic.py linear elasticity (plane stress/strain), P1 vector FEM
    elastic3d.py 3D linear elasticity + thermal stress, tet P1 FEM
    elastic_tet10.py 3D linear elasticity + thermal stress, Tet10
               (quadratic) elements (v1.5)
    modal3d.py 3D structural dynamics: mass matrices + modal analysis
    dynamics3d.py 3D transient dynamics: Newmark time integration
               (average / linear acceleration), Rayleigh damping,
               modal superposition (v1.2)
    thermomech.py sequential thermal → structural (thermal stress, 2D+3D,
               transient 3D chain with peak summary, v1.0)
tests/         analytical validations (must stay green)
examples/      induction_heating.py + output plots, heat_3d.py (v0.7),
               thermal_stress_3d.py (v0.8), modal_beam_3d.py (v0.9),
               transient_thermomech_3d.py (v1.0),
               unstructured_cylinder_3d.py (v1.1),
               transient_beam_3d.py (v1.2),
               eddy_current_3d.py (v1.3: 3D eddy currents in a
               billet + EM → transient-heat chain),
               induction_heating_3d.py (v1.4: flagship full chain —
               billet heated toward the Curie temperature with
               sigma(T), power history and peak thermal stress)
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
- [x] v1.2: 3D transient structural dynamics — Newmark-beta time
      integration of M u'' + C u' + K u = F(t) on the v0.8 stiffness
      and v0.9 mass matrices (physics/dynamics3d.py), structured
      like the v1.0 theta-method heat solver (effective stiffness
      Khat = K + gamma/(beta dt) C + M/(beta dt^2) factorised once,
      exact landing on t_end). Named parameter sets: "average"
      (beta = 1/4, gamma = 1/2; second order, unconditionally
      stable, conserves the quadratic mechanical energy to
      round-off) and "linear" (beta = 1/6; second order but
      conditionally stable, omega dt <= 2 sqrt(3) -- on a mesh this
      binds at the *highest* modal frequency, so it is a comparison
      / teaching variant, a caveat verified numerically rather than
      just stated). Rayleigh damping C = alpha M + beta_r K,
      consistent or lumped mass, initial displacement / velocity,
      loads as constant nodal vectors or callables F(t) (step,
      harmonic); homogeneous Dirichlet only -- nonzero prescribed
      motion is rejected with an error, not silently ignored. A
      modal-superposition solver (same Newmark core in modal
      coordinates) is included. Validated: SDOF step / harmonic
      closed forms (9.1e-4 / 9.9e-5), temporal order 1.997,
      stability boundary reproduced (amplitude 1.000 at
      omega dt = 2 sqrt(3), x2.8e7 growth at 4.0), cantilever step
      response against the exact 12-mode history with
      mode-acceleration static correction (RMS 6.9e-4 of peak; the
      residual is the (omega_j dt)^2/12 period elongation of the
      highest retained modes, diagnosed mode by mode), mesh energy
      drift 2.2e-12 over 5 periods, and the static limit under
      Rayleigh damping (1.4e-4 vs the v0.8 solve). Demo:
      cantilever step load, tip overshoot 0.299 mm approaching the
      undamped 2x-static value 0.307 mm
- [x] v1.3: 3D time-harmonic eddy currents — lowest-order Nedelec
      (Whitney) edge elements in an A-V formulation
      (physics/em3d.py, fem/tet_edge.py): edge circulations for A
      everywhere, nodal V in conducting regions only (one node per
      connected component pinned), e^{iwt} convention and
      Q = sigma |E|^2 / 2 output identical to the 2D EM modules, so
      the v1.0 transient heat solver consumes the 3D Joule field
      unchanged and the 3D EM -> thermal -> structural flagship
      chain is closed. Gauge in air: Whitney forms are elementwise
      divergence-free, so an elementwise grad-div penalty vanishes
      identically; the air region is instead regularised with a
      small artificial conductivity (gauge_eps = 1e-6 of sigma_max)
      in the mass term only, damping exactly the gradient null
      space (measured loss sensitivity between gauge_eps 1e-5 and
      1e-7: relative change <= 7.7e-07 over repeated runs).
      Excitation: uniform applied field via tangential-A Dirichlet
      data A0 = (1/2) B x r on the outer boundary of the new
      sphere/cylinder-in-a-box meshes (mesh/unstructured3d.py), or
      an imposed current density J_s. Joule loss is integrated
      exactly for the affine Whitney field (centroid sampling of
      |E|^2 under-reads the loss by ~10 % at h ~ R/2, measured).
      Validated: discrete exact sequence (curl of a discrete
      gradient ~ 1e-15); conducting sphere against the
      spherical-Bessel analytic solution (loss -7.9 % at
      a/delta = 0.50 with the centre field at +0.02 %, +13.7 % at
      a/delta = 3.02 where h/delta ~ 1.2, both at h = 0.40 a);
      finite cylinder against the v0.2 axisymmetric solver
      (-7.2 % in the quasi-static 100 Hz tier; +36.8 % at 2.5 kHz,
      attributed to the L/D = 2 billet end effect plus h ~ delta
      skin resolution and documented as a band, not hidden);
      low-frequency f^2 and sigma scalings and gauge sensitivity
      in tests/test_em3d.py. Demo: steel billet in an axial AC
      field, 201 W of Joule heating chained into a 30 s transient
      heat run. Limitations: linear materials, uniform mesh only
      (skin layers thinner than h are under-resolved), far-field
      boundary at finite distance ((a/L)^3 perturbation), direct
      sparse solve keeps edge counts in the tens of thousands,
      omega -> 0 magnetostatics out of scope
- [x] v1.4: 3D induction-heating flagship chain
      (physics/induction3d.py): v1.3 eddy currents → v1.0 transient
      heat on the extracted billet submesh → v0.8 thermal stress
      snapshots, driven as one sequential chain with explicit
      sigma(T) updates (the 2D law sigma0/(1 + 0.004 (T - 20)),
      re-solved every em_every heat steps, frozen inside each EM
      interval; mu = mu0 throughout, the 2D chain's convention).
      Validation (tests/test_induction3d.py): the discrete energy
      account closes at 3.3e-16 (EM energy in vs enthalpy gain --
      an identity when the cross-mesh transfer is conservative, so
      any wiring error breaks it); cross-check against the
      independent 2D axisymmetric chain at the same working point
      (cold-power ratio 1.75 from the 3D end effect + h/delta ~ 1.55
      skin resolution, mean-rise ratio within 1.4 % of the
      power/volume expectation, pointwise ratios bracketing it);
      sigma(T) feedback measured in two regimes (mild: power drifts
      +0.3 % as sigma falls; hot: across the turnover the power
      declines and the surface peak is tamed, 322 vs 382 K
      frozen-sigma). Demo: R20 x L80 mm billet at 180 kA/m,
      2.5 kHz, 12 s -- surface 755 C approaching the Curie
      temperature, core 442 C, power 16.6 → 14.1 kW, peak von
      Mises 469 MPa at t = 6 s. Limitations: mu constant (no
      magnetic Curie transition), adiabatic billet surface,
      explicit (first-order) sigma update, absolute powers inherit
      the v1.3 h/delta error band
- [x] v1.5: Second-order Tet10 (10-node) tetrahedral elements
      (mesh/tet10_mesh.py, fem/tet10.py, physics/elastic_tet10.py):
      quadratic shape functions in barycentric form, 5-point Keast
      quadrature (degree-3 exact) for stiffness/loads, and an
      exactly integrated consistent mass matrix (closed-form
      barycentric moments). Covers steady heat conduction,
      transient heat (theta method, Dirichlet BCs), and 3D linear
      elasticity with body force, 6-node-face tractions and thermal
      loads; meshes come from a Tet4 -> Tet10 edge-midpoint
      converter (corner indices preserved) or directly as box
      meshes. Validation (tests/test_tet10.py): manufactured-
      solution convergence orders L2 2.90 / H1 1.86 (Tet4 on the
      same grids: 1.61 / 0.84); quadratic patch fields reproduced
      to round-off on an unstructured sphere mesh where Tet4 is
      not exact; on the BIG-M6 cantilever the Tet10 (50, 2, 4) mesh
      reaches -0.033 % vs Euler-Bernoulli at the same node count
      where Tet4 (100, 4, 8) sits at -5.95 %; transient checks
      (series solution, exact adiabatic heat conservation, BE/CN
      temporal orders) match the v1.0 P1 results. Limitations:
      affine element maps only (midpoints at geometric midpoints,
      no curved edges); transient solver is Dirichlet-only (no
      Neumann/Robin face terms yet); no Tet10 support in the EM
      edge-element chain or the unstructured-mesh generators
      (converter output works, but boundary sets of converted
      meshes place midpoints on chords, not on curved surfaces).
- [x] v1.6: Tet10 transient heat Neumann/Robin boundary conditions
      (fem/tet10.py: transient_heat_tet10 gains neumann/robin dicts
      with the same names and semantics as tet_heat; new public
      assemble_boundary_terms_tet10; 6-point Dunavant quadrature
      (4th-order exact) on the 6-node faces; time-varying Neumann
      flux supported via callable(x, y, z, t); backward compatible —
      neumann=robin=None reproduces v1.5 bit-for-bit). Validation
      (tests/test_tet10_bc.py): energy-balance drift 3.553e-15 /
      1.561e-15 / 1.295e-15 (constant / time-varying / space-varying
      flux), net-zero pair 8.069e-16; Robin steady slab vs analytic
      (2/3)x, Tet10 inf-err 2.530e-08 (Tet4 2.404e-08, same order,
      dominated by BE time residual); uniform Robin equilibrium
      6.106e-15; max-principle holds above the v1.0 dt threshold,
      CN overshoots +2.240e-03 below it (documented, not hidden).
      Limitations: time-varying Robin not supported; steady
      solver still Dirichlet-only; maximum principle not
      unconditional with the consistent mass matrix (as in v1.0).
- [x] v1.7: Tet10 nodal A-V 3D time-harmonic eddy-current solver
      (cloudfem/physics/em3d_tet10.py): 10-node quadratic tets,
      A-V formulation `curl(1/μ curl A) + iωσA + σ∇V = J_s` with
      the continuity equation `∇·(iωσ(A+∇V)) = 0` as the second
      block (avoids explicit Coulomb gauge in air); exact element
      integrals on affine tets (5-point Keast, degree-3 exact +
      closed-form P2 mass); air-region gauge via gauge_eps
      artificial conductivity in the mass term only (physical
      output always uses true σ); Joule power exact for the
      discrete E field; source complex power `S = ½∫J_s·E* dV`
      gives the coil terminal impedance `Z = 2S/|I|²`. Validation
      (tests/test_tet10_em.py): manufactured-solution
      convergence orders 2.53 / 2.14 (below scalar-P2 3.0 — known
      suboptimality of the nodal curl-curl operator, documented);
      conducting sphere vs spherical-Bessel analytic: power
      −13.61 % (a/δ = 0.5) / −3.04 % (a/δ = 3.02, skin-dominated);
      D1 cylinder (h/δ = 0.22, R/δ = 0.4) 3D vs axisymmetric
      reference −12.25 % (< 15 % criterion); power reconciliation
      (moment identity vs independent quadrature) 5.59e-03,
      EM → heat energy link 2.55e-03; regression trio rc = 0.
      Limitations: D1 coil case at h/δ = 1.12 is only a wide-band
      consistency check (+175.76 %, under-resolved skin; a finer
      probe at h/δ = 0.84 gives +249.8 % non-monotonic —
      pre-asymptotic at this density), so the tight 3D-vs-axi coil
      benchmark needs h/δ ≪ 1 (follow-up work); linear materials,
      single-frequency phasor solve, sparse direct solver
      (~4 DOF per mesh node); affine element maps only.
- [x] v1.8: 3D transient eddy current (theta-method) on the Tet10 nodal
      A-V solver + coil terminal-inductance extraction with EFD analytic
      calibration — scope: docs/v1.8_scope.md
      (cloudfem/physics/em3d_tet10.py adds
      transient_eddy_current_tet10: monolithic theta discretisation
      (BE/CN) of the time-domain A-V system
      `curl(ν curl A) + σ(∂A/∂t + ∇V) = J_s(t)`,
      `∇·(σ(∂A/∂t + ∇V)) = 0`, system matrix LU-factorised once,
      separable source J_s(x)·time_scale(t) assembled once and only
      rescaled per step; the transient Joule power uses the
      instantaneous real field, σ|E|² with no ½ factor. New
      cloudfem/physics/inductance.py: magnetic_energy_tet10 in the
      exact form W_m = Re(a*·K·a)/4 (replacing a centroid-B
      estimate that can be off by ~18× on wire-like near-singular
      fields), terminal_inductance_tet10 (L = 4W_m/|I|²,
      R_ac = 2P/|I|², Z = 2S/|I|², peak-phasor convention) and
      inductance_active_coil_efd (the EFD active-coil formula
      L ≈ μ0·μr·N²·A_m/l_coil). Also fixes a v1.7 sign bug:
      source complex power is now S = −½∫J_s*·E dV (v1.7 returned
      +½∫J_s·E*, whose real part is minus the Joule loss, so the
      terminal Z had a negative real part). Validation
      (tests/test_tet10_em_transient.py + tests/test_inductance.py;
      all simulation output, independently re-measured): DC-step
      (BE) long-time limit vs the low-frequency phasor solve
      4.708e-08; sine drive (CN) phasor vs the v1.7 phasor solver
      0.036 %, step-halving temporal orders BE 1.78 (step
      problem) / CN 2.005 (smooth sine); complex-power regression
      Re(S) vs P 7.82 % and Im(S) vs 2ωW_m 3.07 % (the 5-point
      quadrature error scale of the discontinuous coil current —
      the exact W_m identity itself checks to < 1e-12);
      inductance energy route vs Z-imaginary route 2.98 %
      (< 5 %); D1 coil L vs the EFD analytic value, ratio 1.501
      (inside the stated 0.3–3 band, single-turn mapping with
      finite length + truncated domain as documented); full
      regression run_all 6/6 PASS + BIG-D3 SKIP, rc = 0.
      Limitations: CN is unreliable on step sources — the
      gauge-regularised stiff modes do not decay under CN
      (R(z) → −1, non-decaying ringing; the CN and BE solutions
      differ by 25.1 % on the step problem), so CN is for smooth
      time_scale only and step / piecewise sources must use BE
      (L-stable), a partition the tests assert explicitly; the
      manufactured-solution orders 2.53 / 2.14 of the nodal
      curl-curl operator remain suboptimal — restoring the
      optimal order needs second-order Nédélec edge elements,
      deferred beyond v1.8; linear materials, separable sources
      only, sparse direct solver, affine element maps only.
- [x] v1.9: 3D nonlinear B-H (saturation) phasor eddy current on the
      Tet10 nodal A-V solver — secant-mu Picard iteration with the
      BIG-M4/M10 analytic magnetisation law
      — scope: docs/v1.9_scope.md
      (cloudfem/physics/em3d_tet10.py adds bh_secant_mu_r, the
      secant mu_r(|H|) of the analytic magnetisation law
      M(H) = Ms*tanh(H/H_REF) — a stated model assumption, not
      measured material data, isothermal so the M10 Curie switch
      is identically 1 — and solve_eddy_current_nonlinear_tet10:
      per-element nu(|B|) updated through the secant relation with
      relaxation 0.5 (as in M10), the sigma blocks assembled once
      and K(nu_e) reassembled each round; the convergence norm is
      taken over the iron elements only — a whole-domain norm was
      measured to mask the nonlinear region's motion, because the
      air-region |B| is set by the applied field. Validation
      (tests/test_tet10_em_nonlinear.py; all simulation output,
      independently re-measured): linear limit bitwise identical
      to the v1.7 linear solver on both the mu_mag0=0 and
      iron=None paths (rel. diff 0.000e+00); strong-drive
      nonlinear power 468910.5 W strictly bracketed by the
      constant-mu_r=1 (150757.5 W) and mu_r=201 (531896.3 W)
      linear solves on the same mesh; iron max|B| 2.718 T,
      clamped below the mu_r=201 linear extrapolation 4.425 T,
      with a surface/deep mu_r gradient of 6.1 vs 200.1; Picard
      converged in 40 iterations (rel_dB peak 1.62e-02 -> final
      9.53e-04 < 1e-3, P monotone decreasing); BIG-M10 axi
      order-of-magnitude check ratio 0.372 inside the stated
      0.2–5 band (non-tight: finite-length demagnetisation,
      drive and mesh differences as documented); full regression
      run_all 6/6 PASS + BIG-D3 SKIP, rc = 0, and the v1.7/v1.8
      numbers reproduce item by item. Limitations: Picard
      converges slowly under strong saturation (~0.92x/round in
      the tail at relax=0.5; no Anderson/Aitken acceleration);
      the secant/effective-mu phasor approximation generates no
      harmonics and no hysteresis; the material law is an
      analytic model, not measured data; no temperature coupling
      or Curie point (v2.0 candidate, joining the v1.8 transient
      EM -> thermal chain); the test mesh is coarse (h/delta ~ 49),
      so absolute values are coarse-mesh simulation output and a
      resolution-convergence study is a compute-queue task, not
      part of this version; the manufactured-solution orders
      2.53 / 2.14 of the nodal curl-curl operator remain
      suboptimal (second-order Nedelec edge elements deferred).
- [x] v2.0: 3D nonlinear B-H + thermal coupling on the Tet10
      nodal A-V solver — mu_r(|B|, T) with the BIG-M4/M10 Curie
      switch and sigma(T) law, staggered coupling to Tet10
      transient heat conduction (merged 2026-10-23, d7a746b;
      time-convergence: em_every=1 family C1 vs C3 T +0.51%/P +0.76%,
      t_cross |Δ|=0.005s ≤ 0.01s)
      — scope: docs/v2.0_scope.md; report: examples/convergence/CONVERGENCE_REPORT.md
- [ ] v2.1: cp(T) 居里焓峰（候选，未合 main；分支 v2.1-cp-enthalpy）——
      体积热容 = 基线 + 居里高斯焓峰，焓形式能量对账；峰关闭逐位复现 v2.0。
      已知限制：默认工况（em_every=5，EM 间隔 0.1s）下温度在一个 EM 间隔内
      越过居里窗口，焓峰不生效；需 em_every=1（或单区间温升 ≲ cp_peak_w）
      峰才起作用。未写明该限制前不称功能完成；本轮仅代码质量与数值行为
      核查，非物理验收。
      — scope: docs/v2.1_scope.md

## Licence

MIT — see `LICENSE`.
