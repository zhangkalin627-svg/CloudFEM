"""3D nonlinear B-H + thermal coupling on the Tet10 A-V solver (v2.0).

Extends the v1.9 phasor eddy-current solver
(:mod:`cloudfem.physics.em3d_tet10`) with the temperature dependence
of the BIG-M4/M10 steel law and a staggered coupling to Tet10
transient heat conduction:

    per EM interval (temperature field frozen):
        T_e -> sigma(T_e), secant mu_r(|B|, T_e)   (Picard, as v1.9)
            -> phasor A-V solve -> element Joule density Q_e
    per heat step (Q frozen):
        theta-method Tet10 heat conduction on the workpiece submesh

so the induction-heating Curie transition runs in 3D: the surface
layer crosses Tc first, its mu_r collapses towards 1, the skin depth
grows and the Joule power redistributes towards the core.

Material laws -- the BIG-M4/M10 analytic *model* laws, copied in
form from tests/test_bench_nonlinear_bh.py (identical there and in
the standalone BIG-M10 script).  Documented model assumptions, NOT
measured material data:

    Curie switch   step(T) = 0.5 * (1 - tanh((T - Tc) / W_STEP)),
                   Tc = 770 degC, W_STEP = 15 K
    magnetisation  mu_r(|H|, T) = 1 + mu_mag(|H|) * step(T) with
                   mu_mag(|H|) = MU_MAG0 * tanh(h) / h,
                   h = |H| / H_REF, Ms = Bsat / MU0, Bsat = 1.9 T,
                   MU_MAG0 = 200, H_REF = Ms / MU_MAG0 -- i.e.
                   1 + (bh_secant_mu_r(|H|) - 1) * step(T) with the
                   v1.9 secant curve, so at constant low temperature
                   (step == 1.0 exactly in float64 at 20 degC) the
                   target mu is bitwise the v1.9 one.
    conductivity   sigma(T) = SIG_LO + (SIG_HI - SIG_LO) * s(T),
                   s(T) = 0.5 * (1 - tanh((T - T_SIG) / W_SIG)),
                   SIG_HI = 6e6, SIG_LO = 1e6 S/m, T_SIG = 550 degC,
                   W_SIG = 150 K.  (sigma(20 degC) = 5.995738e6 S/m,
                   not exactly 6e6; BIG-M10 case A uses the same
                   sigma(20 degC) convention isothermally.)  This is
                   NOT the linear-resistivity law
                   induction3d.steel_sigma_of_T -- the two laws live
                   in different drivers and must not be mixed.
    thermal        k = 30 W/(m K) constant; volumetric heat capacity
                   c(T) = rho_cp * (1 + cp_slope * (T - 20))
                   + cp_peak_h * exp(-(T - t_curie)^2/(2*cp_peak_w^2))
                   [J/(m^3 K)] (:func:`steel_cp_T`).  Defaults
                   (cp_slope = 0, cp_peak_h = 0) give the constant
                   rho_cp = 7850 * 480 = 3.768e6 J/(m^3 K), i.e. the
                   v2.0 thermal law bitwise.  With the peak enabled
                   the Gaussian centred at t_curie is the documented
                   analytic model of the magnetic-transition
                   enthalpy (model assumption, not measured data);
                   Curie-crossing *times* then include the
                   transition-enthalpy delay.  The capacity matrix is
                   rebuilt from the element temperatures at each EM
                   interval start and frozen over the interval (the
                   same staggered granularity as sigma(T)/mu_r(T));
                   the per-interval discrete identity
                   ones^T C_iv dT_iv = P_iv * dt_iv holds to linear
                   solver precision (ones^T K = 0, adiabatic).

Coupling (loosely coupled / staggered, the BIG-M10 case-B protocol):
the EM problem is re-solved every ``em_every`` heat steps with the
temperature frozen at the interval start (Picard warm-started from
the previous interval's nu field) and Q is held constant over the
interval.  ``stagger_iter > 1`` adds fixed-point passes over each
interval: re-solve the EM problem at the estimated interval-end
temperature and re-advance the heat from the interval start, until
the interval-end temperature change falls below ``stagger_tol``
or the pass count is exhausted.

The workpiece is the set of elements marked by the ``iron``
predicate; the heat solve runs on that Tet10 submesh with all faces
adiabatic (the same short-time assumption as BIG-M10 case B and the
v1.4 chain: no heat loss to air/coil).  The element Joule source is
assembled exactly as an element-constant load (the assemble_tet10
f-assembly with f == Q_e), so with the adiabatic boundary the
discrete identity ones^T C_iv dT = dt * P holds per EM interval
(ones^T K = 0; C_iv frozen per interval, constant-C v2.0 as the
special case) and the accumulated EM energy matches the discrete
enthalpy gain to linear-solver precision; the driver reports both
sides plus the analytic-enthalpy diagnostic
ones^T M_1 (H(T) - H(T0)) with H from :func:`steel_enthalpy_T`.

Limitations: secant / effective-mu fundamental-harmonic phasor
approximation as in v1.9 (no harmonics, no hysteresis); constant
thermal properties (see above); the EM solver here is specific to
the steel-workpiece setting -- non-iron elements are air (sigma = 0,
nu = 1/MU0); meshes should stay small (the phasor system is solved
once per Picard iteration).
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import erf
from scipy.spatial import cKDTree

from ..fem.tet10 import (assemble_mass_tet10, assemble_tet10,
                         element_operators, quadrature)
from ..mesh.tet10_mesh import Tet10Mesh
from .em import MU0
from .em3d_tet10 import (_assemble_curl_curl_tet10_nu,
                         _solve_phasor_with_K_tet10, assemble_tet10_av,
                         bh_secant_mu_r, element_flux_density_tet10,
                         total_joule_power_tet10)

__all__ = [
    "curie_step_T", "steel_sigma_T", "bh_secant_mu_r_HT",
    "steel_cp_T", "steel_enthalpy_T",
    "solve_eddy_current_nonlinear_tet10_T",
    "induction_heating_3d_tet10",
    "TC_CURIE", "W_STEP_CURIE", "SIG_HI", "SIG_LO", "T_SIG", "W_SIG",
]

# BIG-M4/M10 material constants (defaults; all overridable).
TC_CURIE = 770.0          # Curie temperature [degC]
W_STEP_CURIE = 15.0       # Curie switch width [K]
SIG_HI = 6.0e6            # conductivity, cold [S/m]
SIG_LO = 1.0e6            # conductivity, hot [S/m]
T_SIG = 550.0             # sigma transition centre [degC]
W_SIG = 150.0             # sigma transition width [K]

_THETA_HEAT = {"be": 1.0, "implicit_euler": 1.0,
               "cn": 0.5, "crank_nicolson": 0.5}


def curie_step_T(T, t_curie=TC_CURIE, w_step=W_STEP_CURIE):
    """Curie switch of BIG-M4/M10: 1 below Tc, 0 above Tc.

    ``0.5 * (1 - tanh((T - Tc) / W_STEP))`` verbatim in form.
    Scalar or array T [degC].
    """
    return 0.5 * (1.0 - np.tanh((np.asarray(T, dtype=float)
                                 - float(t_curie)) / float(w_step)))


def steel_sigma_T(T, sig_hi=SIG_HI, sig_lo=SIG_LO, t_sig=T_SIG,
                  w_sig=W_SIG):
    """Electrical conductivity [S/m] of the BIG-M4/M10 tanh law.

    6e6 S/m cold -> 1e6 S/m hot, centre 550 degC, width 150 K
    (defaults).  NOT induction3d.steel_sigma_of_T (a different,
    linear-resistivity law).  Scalar or array T [degC].
    """
    s = 0.5 * (1.0 - np.tanh((np.asarray(T, dtype=float)
                              - float(t_sig)) / float(w_sig)))
    return float(sig_lo) + (float(sig_hi) - float(sig_lo)) * s


def bh_secant_mu_r_HT(Hmag, T, bsat=1.9, mu_mag0=200.0,
                      t_curie=TC_CURIE, w_step=W_STEP_CURIE):
    """Secant mu_r(|H|, T) = 1 + mu_mag(|H|) * step(T)  (BIG-M10).

    mu_mag(|H|) is the v1.9 secant magnetic part
    ``bh_secant_mu_r(|H|) - 1`` (the BIG-M4/M10 tanh magnetisation
    law, a documented analytic model curve, not measured data);
    step(T) is :func:`curie_step_T`.  At temperatures where
    step(T) == 1 (e.g. 20 degC, exactly in float64) this reduces
    bitwise to :func:`bh_secant_mu_r`; far above Tc the magnetic
    part vanishes and mu_r -> 1.
    """
    return (1.0 + (bh_secant_mu_r(Hmag, bsat=bsat, mu_mag0=mu_mag0)
                   - 1.0) * curie_step_T(T, t_curie=t_curie,
                                         w_step=w_step))


def steel_cp_T(T, rho_cp=3.768e6, cp_slope=0.0, t_ref=20.0,
               cp_peak_h=0.0, cp_peak_w=25.0, t_curie=TC_CURIE):
    """Volumetric heat capacity [J/(m^3 K)] with Curie enthalpy peak.

    c(T) = rho_cp * (1 + cp_slope * (T - t_ref))
           + cp_peak_h * exp(-(T - t_curie)^2 / (2 * cp_peak_w^2))

    The Gaussian peak centred at ``t_curie`` is the documented
    analytic model of the magnetic-transition enthalpy (its area
    ``cp_peak_h * cp_peak_w * sqrt(2*pi)`` is the transition
    enthalpy per unit volume); peak height/width are model
    assumptions, NOT measured data.  Defaults (cp_slope = 0,
    cp_peak_h = 0) give the constant ``rho_cp``, i.e. the v2.0
    thermal law bitwise.  Scalar or array T [degC].
    """
    T = np.asarray(T, dtype=float)
    c = float(rho_cp) * (1.0 + float(cp_slope) * (T - float(t_ref)))
    h = float(cp_peak_h)
    w = float(cp_peak_w)
    if h != 0.0 and w > 0.0:
        z = (T - float(t_curie)) / w
        c = c + h * np.exp(-0.5 * z * z)
    return c


def steel_enthalpy_T(T, t_ref=20.0, rho_cp=3.768e6, cp_slope=0.0,
                     cp_peak_h=0.0, cp_peak_w=25.0,
                     t_curie=TC_CURIE):
    """Volumetric enthalpy [J/m^3] relative to ``t_ref``.

    H(T) = integral_{t_ref}^{T} c(s) ds with c(s) from
    :func:`steel_cp_T`, in closed form: the baseline integrates to
    ``rho_cp * ((T - t_ref) + cp_slope * (T - t_ref)^2 / 2)`` and the
    Gaussian peak to ``cp_peak_h * cp_peak_w * sqrt(pi/2) *
    (erf((T - t_curie)/(w*sqrt(2))) - erf((t_ref - t_curie)/
    (w*sqrt(2))))``.  Scalar or array T [degC].
    """
    T = np.asarray(T, dtype=float)
    dT = T - float(t_ref)
    H = float(rho_cp) * (dT + 0.5 * float(cp_slope) * dT * dT)
    h = float(cp_peak_h)
    w = float(cp_peak_w)
    if h != 0.0 and w > 0.0:
        sq2 = np.sqrt(2.0)
        arg_T = (T - float(t_curie)) / (w * sq2)
        arg_ref = (float(t_ref) - float(t_curie)) / (w * sq2)
        H = H + h * w * np.sqrt(np.pi / 2.0) * (erf(arg_T) - erf(arg_ref))
    return H


def _centroid_lookup(centroids, values):
    """Callable(x, y, z) -> values of the nearest centroid.

    Exact for queries made at the centroids themselves (distance 0),
    which is where the A-V assembler evaluates element fields (the
    induction3d pattern).
    """
    tree = cKDTree(np.asarray(centroids, dtype=float))
    vals = np.asarray(values, dtype=float)

    def fn(x, y, z):
        _, idx = tree.query([x, y, z])
        return float(vals[idx])

    return fn


def _iron_mask(mesh, iron):
    if iron is None:
        raise ValueError("iron predicate is required: it marks the "
                         "steel elements whose mu and sigma follow "
                         "the temperature-dependent BIG-M4/M10 laws")
    cen = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    return np.array([bool(iron(x, y, z)) for x, y, z in cen]), cen


def solve_eddy_current_nonlinear_tet10_T(mesh, T_elem, omega,
                                         B_applied=None, J_s=None,
                                         boundary=None, a_bc=None,
                                         gauge_eps=1e-6,
                                         permc_spec=None, iron=None,
                                         bsat=1.9, mu_mag0=200.0,
                                         t_curie=TC_CURIE,
                                         w_step=W_STEP_CURIE,
                                         sig_hi=SIG_HI, sig_lo=SIG_LO,
                                         t_sig=T_SIG, w_sig=W_SIG,
                                         mu_init=None, nu_init=None,
                                         tol=1e-3, max_iter=50,
                                         relax=0.5):
    """Nonlinear B-H phasor eddy-current solve at frozen temperature.

    The v1.9 Picard solver (:func:`solve_eddy_current_nonlinear_
    tet10`) with the element temperature field ``T_elem`` [degC,
    per element] folded into the BIG-M4/M10 laws: on the iron
    elements the target secant permeability is
    :func:`bh_secant_mu_r_HT` and the conductivity is
    :func:`steel_sigma_T` (assembled per element); non-iron
    elements are air (sigma = 0, nu = 1/MU0).  Amplitudes are
    phasor peak amplitudes, as in v1.9.

    With ``T_elem`` uniformly at a low temperature (step(T) == 1)
    and sigma == steel_sigma_T(T) held constant, this reproduces
    :func:`solve_eddy_current_nonlinear_tet10` run with that
    constant sigma -- the same assembly and iteration path, so the
    agreement is to solver precision (bitwise in practice).

    Additional parameters beyond the v1.9 solver's: ``T_elem``;
    ``t_curie``, ``w_step`` (Curie switch); ``sig_hi``, ``sig_lo``,
    ``t_sig``, ``w_sig`` (conductivity law); ``nu_init`` -- a
    per-element reluctivity array used as the Picard start
    (warm start from a previous solve; overrides ``mu_init`` on
    the iron elements).

    Returns (a, v, info) with info as in v1.9 plus ``Q_elem``
    (element mean Joule density [W/m^3] of the final solve),
    ``sigma_elem`` and ``T_elem``.
    """
    if omega <= 0:
        raise ValueError("omega must be positive (phasor eddy-current "
                         "solve; omega = 0 magnetostatics is out of scope)")
    if B_applied is None and J_s is None:
        raise ValueError("no excitation: pass B_applied (background "
                         "field) or J_s (imposed current density)")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not (0.0 < relax <= 1.0):
        raise ValueError("relax must be in (0, 1]")
    T_elem = np.asarray(T_elem, dtype=float)
    if T_elem.shape != (mesh.n_elements,):
        raise ValueError(f"T_elem must have shape ({mesh.n_elements},), "
                         f"got {T_elem.shape}")
    iron_mask, cen = _iron_mask(mesh, iron)
    if not iron_mask.any():
        raise ValueError("iron predicate selects no elements")

    # Per-element conductivity from the frozen temperature field.
    sig_e = np.zeros(mesh.n_elements)
    sig_e[iron_mask] = steel_sigma_T(T_elem[iron_mask], sig_hi=sig_hi,
                                     sig_lo=sig_lo, t_sig=t_sig,
                                     w_sig=w_sig)
    sigma_fn = _centroid_lookup(cen, sig_e)
    # sigma blocks depend on T but not on nu: assemble once (the K
    # built by assemble_tet10_av with nu = 1 is discarded, exactly
    # as in the v1.9 solver).
    _, M, Gs, S, v_nodes, sig_e, conducting = assemble_tet10_av(
        mesh, nu=1.0, sigma=sigma_fn, gauge_eps=gauge_eps)
    if not conducting.any():
        raise ValueError("no conducting elements (sigma > 0) in mesh")

    nu_e = np.full(mesh.n_elements, 1.0 / MU0)
    if nu_init is not None:
        nu_e = np.asarray(nu_init, dtype=float).copy()
        if nu_e.shape != (mesh.n_elements,):
            raise ValueError("nu_init must be a per-element array")
    else:
        mu_r_init = (1.0 + float(mu_mag0)) if mu_init is None \
            else float(mu_init)
        nu_e[iron_mask] = 1.0 / (MU0 * mu_r_init)

    hist_rel, hist_P = [], []
    B_amp_prev = None
    converged = False
    a = v = B = Q = None
    it = 0
    for it in range(1, max_iter + 1):
        K = _assemble_curl_curl_tet10_nu(mesh, nu_e)
        a, v = _solve_phasor_with_K_tet10(
            mesh, K, M, Gs, S, v_nodes, omega, B_applied, J_s,
            boundary, a_bc, permc_spec)
        B = element_flux_density_tet10(mesh, a)
        B_amp = np.sqrt(np.sum(np.abs(B) ** 2, axis=1))
        P, Q = total_joule_power_tet10(mesh, a, v, sigma_fn, omega)
        hist_P.append(P)
        rel = None
        if B_amp_prev is not None:
            # Iron-only |B| norm, as in v1.9: the air-region |B| is
            # set by the applied field and would mask the nonlinear
            # region's motion.
            denom = max(float(np.linalg.norm(B_amp[iron_mask])), 1e-300)
            rel = float(np.linalg.norm(B_amp[iron_mask]
                                       - B_amp_prev[iron_mask]) / denom)
            hist_rel.append(rel)
            if rel < tol:
                converged = True
                break
        if it == max_iter:
            break
        B_amp_prev = B_amp
        H_amp = B_amp[iron_mask] * nu_e[iron_mask]
        mu_tgt = bh_secant_mu_r_HT(H_amp, T_elem[iron_mask], bsat=bsat,
                                   mu_mag0=mu_mag0, t_curie=t_curie,
                                   w_step=w_step)
        nu_tgt = 1.0 / (MU0 * mu_tgt)
        nu_e[iron_mask] = ((1.0 - relax) * nu_e[iron_mask]
                           + relax * nu_tgt)

    mu_r_elem = 1.0 / (MU0 * nu_e)
    info = {"iterations": it, "converged": converged,
            "rel_dB": np.array(hist_rel), "P": np.array(hist_P),
            "mu_r_elem": mu_r_elem, "nu_elem": nu_e.copy(),
            "B_elem": B, "Q_elem": Q, "sigma_elem": sig_e.copy(),
            "T_elem": T_elem.copy(), "iron_mask": iron_mask,
            "relax": relax, "tol": tol}
    return a, v, info


def _extract_submesh_tet10(mesh, elem_mask):
    """Extract the masked Tet10 elements as a standalone Tet10Mesh.

    Returns (submesh, used_nodes, elem_indices); submesh element i
    is full-mesh element elem_indices[i] (mask order preserved).
    Boundary tags are not carried over (the coupled driver's
    workpiece is adiabatic on every face).
    """
    elem_idx = np.where(np.asarray(elem_mask, dtype=bool))[0]
    els = mesh.elements[elem_idx]
    used, inv = np.unique(els, return_inverse=True)
    sub = Tet10Mesh(mesh.nodes[used], inv.reshape(-1, 10),
                    n_corners=int(np.sum(used < mesh.n_corners)))
    return sub, used, elem_idx


def _element_source_load(sub, Q_sub):
    """Nodal load of an element-constant heat source Q_sub [W/m^3].

    Identical formula to the f-assembly of fem.tet10.assemble_tet10
    with f constant per element, so ones^T F == sum_e Q_e * vol_e
    exactly (the weights sum to 1 and the shape functions to 1).
    """
    _, wq = quadrature()
    F = np.zeros(sub.n_nodes)
    for e, el in enumerate(sub.elements):
        el = np.asarray(el, dtype=int)
        _, N, _, vol = element_operators(sub, el)
        fe = vol * float(Q_sub[e]) * (N * wq[:, None]).sum(axis=0)
        F[el] += fe
    return F


def induction_heating_3d_tet10(mesh, omega, B_applied=None, J_s=None,
                               boundary=None, a_bc=None, iron=None,
                               T0=20.0, t_end=1.0, dt=0.01,
                               em_every=5, k_th=30.0, rho_cp=3.768e6,
                               method="be", stagger_iter=1,
                               stagger_tol=None, em_tol=1e-3,
                               em_max_iter=50, em_relax=0.5,
                               gauge_eps=1e-6, permc_spec=None,
                               diag=False, **law_kwargs):
    """Staggered EM -> thermal induction heating of a steel workpiece.

    Drives :func:`solve_eddy_current_nonlinear_tet10_T` and Tet10
    transient heat conduction on the workpiece submesh (see the
    module docstring for the coupling protocol, the material laws
    and the energy-accounting convention).  ``law_kwargs`` forwards
    the
    material-law parameters (bsat, mu_mag0, t_curie, w_step,
    sig_hi, sig_lo, t_sig, w_sig) to the EM solver, and carries
    the v2.1 cp(T) law parameters (cp_slope, cp_peak_h,
    cp_peak_w; popped before the EM call): volumetric heat
    capacity c(T) = rho_cp * (1 + cp_slope * (T - 20))
    + cp_peak_h * exp(-(T - t_curie)^2/(2*cp_peak_w^2))
    [J/(m^3 K)] (:func:`steel_cp_T`).  Defaults (cp_slope = 0,
    cp_peak_h = 0) give the constant rho_cp, i.e. the v2.0
    thermal law bitwise; with the peak enabled the capacity
    matrix is rebuilt from the element temperatures at each EM
    interval start and frozen over the interval.

    Parameters (beyond the EM solver's)
    ----------
    T0 : uniform initial temperature [degC].
    t_end, dt : total time and heat step [s]; the step actually
        used is t_end / ceil(t_end / dt), as in
        transient_heat_tet10.
    em_every : heat steps per EM interval (EM re-solved at each
        interval start, temperature frozen there).
    k_th, rho_cp : constant workpiece thermal conductivity
        [W/(m K)] and volumetric heat capacity [J/(m^3 K)].
    method : heat time integration, "be" (default) or "cn".
    stagger_iter, stagger_tol : fixed-point passes per EM interval
        (module docstring); 1 = pure loosely-coupled update.
    diag : v2.4 diagnostic switch (default False = off).  When True,
        each per-interval history record additionally carries
        ``stagger_dT_seq`` (list of per-pass dT in pass order;
        length = actual passes, the first pass entry is None
        "missing" -- never 0-filled), ``frac_above_Tc_vol`` (volume
        fraction of iron elements with T > t_curie at the interval-end
        temperature field) and ``mu_r_agg`` (min / volume-weighted
        mean / max of the secant mu_r used by that interval's last
        EM pass, sampled at that pass's input temperature field).
        The pre-existing ``frac_above_Tc`` key is the element-count
        fraction and is kept unchanged; the volume-fraction
        diagnostic uses the distinct key ``frac_above_Tc_vol`` so
        the two conventions never mix.  Diagnostic recording is
        read-only observation: no solver array is written by the
        diagnostic path, so diag on/off does not alter any
        computed number (D1 criterion).

    Returns a dict with ``times`` / ``T_hist`` (workpiece nodal
    temperatures at the EM interval ends, T_hist[0] at t = 0),
    ``T_elem_hist`` (per-iron-element temperatures, same times),
    ``history`` (per-interval records: t_end, P, em_iterations,
    em_converged, stagger_passes, T_elem_mean/max, frac_above_Tc,
    mu_r_iron_mean/min; plus stagger_dT_seq, frac_above_Tc_vol,
    mu_r_agg when diag=True), ``E_in`` / ``enthalpy_gain`` /
    ``balance_rel`` (energy account; with cp(T) enabled the
    account is the discrete enthalpy form sum_iv ones^T C_iv
    dT_iv, and ``enthalpy_gain_phys`` carries the analytic
    enthalpy diagnostic), ``cp_varying``, ``submesh``,
    ``iron_idx``, ``used_nodes`` and the final ``a``, ``v``,
    ``em_info``.
    """
    if iron is None:
        raise ValueError("iron predicate is required (it defines "
                         "the workpiece)")
    if method not in _THETA_HEAT:
        raise ValueError(f"unknown method {method!r}; "
                         f"use one of {sorted(_THETA_HEAT)}")
    if em_every < 1:
        raise ValueError("em_every must be >= 1")
    if stagger_iter < 1:
        raise ValueError("stagger_iter must be >= 1")
    theta = _THETA_HEAT[method]

    t_curie = float(law_kwargs.get("t_curie", TC_CURIE))
    # v2.1 cp(T) law parameters (popped: not forwarded to the EM solver).
    cp_slope = float(law_kwargs.pop("cp_slope", 0.0))
    cp_peak_h = float(law_kwargs.pop("cp_peak_h", 0.0))
    cp_peak_w = float(law_kwargs.pop("cp_peak_w", 25.0))
    if cp_peak_h != 0.0 and cp_peak_w <= 0.0:
        raise ValueError(
            "cp_peak_w must be positive when cp_peak_h != 0 "
            f"(got cp_peak_h={cp_peak_h}, cp_peak_w={cp_peak_w})")
    cp_varying = (cp_slope != 0.0) or (cp_peak_h != 0.0
                                      and cp_peak_w > 0.0)

    iron_mask, _ = _iron_mask(mesh, iron)
    sub, used_nodes, iron_idx = _extract_submesh_tet10(mesh, iron_mask)
    vols_sub = sub.element_volumes()

    C1 = assemble_mass_tet10(sub, rho_cp=1.0)
    K_th, _ = assemble_tet10(sub, k=k_th, f=0.0)
    ones_sub = np.ones(sub.n_nodes)

    n_steps = int(np.ceil(t_end / dt - 1e-12))
    if n_steps < 1:
        raise ValueError("t_end must be positive")
    h_step = t_end / n_steps

    def _make_solver(C_mat):
        """Factorized theta-stepper for one capacity matrix.

        Identical operations to the v2.0 inline code; the closure
        is rebuilt per EM interval when cp(T) varies.
        """
        A_sys = C_mat + theta * h_step * K_th
        R_sys = C_mat - (1.0 - theta) * h_step * K_th
        solve = spla.factorized(A_sys.tocsc())

        def advance(T_start, F_Q, n_sub):
            T = T_start.copy()
            for _ in range(n_sub):
                T = np.asarray(solve(R_sys @ T + h_step * F_Q)).ravel()
            return T

        return advance

    def elem_temps(T_nodes):
        return T_nodes[sub.elements].mean(axis=1)

    def _capacity_matrix(T_nodes):
        """Capacity matrix from element temperatures (v2.1).

        Element-level (centroid) granularity, the same convention as
        sigma(T)/mu_r(T) in the EM solver.
        """
        sub_cen = sub.nodes[sub.elements[:, :4]].mean(axis=1)
        c_e = steel_cp_T(elem_temps(T_nodes), rho_cp=rho_cp,
                         cp_slope=cp_slope, cp_peak_h=cp_peak_h,
                         cp_peak_w=cp_peak_w, t_curie=t_curie)
        return assemble_mass_tet10(
            sub, rho_cp=_centroid_lookup(sub_cen, c_e))

    if not cp_varying:
        # v2.0 path: constant capacity matrix, factorized once.
        # Bitwise the v2.0 driver when cp_slope == 0 and the peak is
        # off (steel_cp_T reduces to the constant rho_cp exactly).
        C_const = assemble_mass_tet10(sub, rho_cp=rho_cp)
        advance = _make_solver(C_const)
    else:
        C_const = None
        advance = None  # rebuilt per EM interval below
    T = np.full(sub.n_nodes, float(T0))
    times = [0.0]
    T_hist = [T.copy()]
    T_elem_hist = [elem_temps(T)]
    history = []
    E_in = 0.0
    enthalpy_gain_disc = 0.0  # v2.1: sum of ones^T C_iv dT_iv
    nu_warm = None
    a = v = em_info = None

    n_intervals = int(np.ceil(n_steps / em_every))
    for iv in range(n_intervals):
        n_sub = min(em_every, n_steps - iv * em_every)
        T_start = T.copy()
        if cp_varying:
            # v2.1: capacity matrix from the interval-start
            # temperatures, frozen over the interval (staggered
            # granularity, as sigma(T)/mu_r(T)).
            C_iv = _capacity_matrix(T_start)
            advance_iv = _make_solver(C_iv)
        else:
            C_iv = C_const
            advance_iv = advance
        T_est = T_start.copy()
        T_trial = None
        dT_stagger = None
        dT_seq = []  # v2.4 diag: per-pass dT record (read-only)
        for s in range(stagger_iter):
            T_elem = np.full(mesh.n_elements, float(T0))
            T_elem[iron_idx] = elem_temps(T_est)
            a, v, em_info = solve_eddy_current_nonlinear_tet10_T(
                mesh, T_elem, omega, B_applied=B_applied, J_s=J_s,
                boundary=boundary, a_bc=a_bc, gauge_eps=gauge_eps,
                permc_spec=permc_spec, iron=iron, tol=em_tol,
                max_iter=em_max_iter, relax=em_relax,
                nu_init=nu_warm, **law_kwargs)
            nu_warm = em_info["nu_elem"]
            F_Q = _element_source_load(sub, em_info["Q_elem"][iron_idx])
            T_new = advance_iv(T_start, F_Q, n_sub)
            if T_trial is not None:
                dT_stagger = float(np.max(np.abs(T_new - T_trial)))
            if diag:
                # First pass has no predecessor: record an explicit
                # missing marker (None), never a 0 fill.
                dT_seq.append(None if T_trial is None else dT_stagger)
            T_trial = T_new
            T_est = T_trial
            if (stagger_tol is not None and dT_stagger is not None
                    and dT_stagger < stagger_tol):
                break
        T = T_trial
        P_used = float(em_info["P"][-1])
        E_in += P_used * (n_sub * h_step)
        if cp_varying:
            # Discrete enthalpy increment of this interval:
            # ones^T C_iv (T_end - T_start) == P_iv * dt_iv exactly
            # (ones^T K = 0), to linear-solver precision.
            enthalpy_gain_disc += float(ones_sub @ (C_iv @ (T - T_start)))
        Te = elem_temps(T)
        mu_iron = em_info["mu_r_elem"][iron_idx]
        rec = {
            "t_end": times[-1] + n_sub * h_step,
            "P": P_used,
            "em_iterations": em_info["iterations"],
            "em_converged": bool(em_info["converged"]),
            "stagger_passes": s + 1,
            "stagger_dT": dT_stagger,
            "T_elem_mean": float(Te.mean()),
            "T_elem_max": float(Te.max()),
            "frac_above_Tc": float(np.mean(Te > t_curie)),
            "mu_r_iron_mean": float(mu_iron.mean()),
            "mu_r_iron_min": float(mu_iron.min()),
            "mu_r_iron": mu_iron.copy(),
            "Q_iron": em_info["Q_elem"][iron_idx].copy(),
        }
        if diag:
            # v2.4 diagnostic records (read-only observation only;
            # nothing here writes back into solver state).  Sampling
            # points per CLOUD_CRITERIA.md section 0: mu_r_agg is
            # taken from the last pass's EM solve (i.e. at that
            # pass's INPUT temperature field T_est), while
            # frac_above_Tc_vol is taken from the interval-END
            # temperature field (the last pass's output) -- adjacent
            # in time but different states; record side by side,
            # never assert a time-ordered causal link.
            mur = np.asarray(mu_iron, dtype=float)
            vol_total = float(np.sum(vols_sub))
            vw_mean = float(np.sum(mur * vols_sub) / vol_total)
            rec["stagger_dT_seq"] = list(dT_seq)
            rec["frac_above_Tc_vol"] = float(
                np.sum(vols_sub[Te > t_curie]) / vol_total)
            rec["mu_r_agg"] = (float(mur.min()), vw_mean,
                               float(mur.max()))
        history.append(rec)
        times.append(times[-1] + n_sub * h_step)
        T_hist.append(T.copy())
        T_elem_hist.append(Te.copy())

    if not cp_varying:
        # v2.0 energy account, verbatim.
        enthalpy_gain = float(rho_cp * (ones_sub @ C1 @ (T - float(T0))))
        enthalpy_gain_phys = enthalpy_gain
    else:
        # v2.1 enthalpy-form account: the discrete identity
        # sum_iv ones^T C_iv dT_iv == E_in, to solver precision.
        # The analytic-enthalpy diagnostic is reported separately;
        # it differs by the interval-freezing (method) error.
        enthalpy_gain = enthalpy_gain_disc
        H_end = steel_enthalpy_T(T, t_ref=float(T0), rho_cp=rho_cp,
                                 cp_slope=cp_slope, cp_peak_h=cp_peak_h,
                                 cp_peak_w=cp_peak_w, t_curie=t_curie)
        enthalpy_gain_phys = float(ones_sub @ (C1 @ H_end))
    balance_rel = (E_in - enthalpy_gain) / E_in if E_in > 0 else 0.0
    return {"times": np.array(times), "T_hist": np.array(T_hist),
            "T_elem_hist": np.array(T_elem_hist), "history": history,
            "E_in": E_in, "enthalpy_gain": enthalpy_gain,
            "enthalpy_gain_phys": enthalpy_gain_phys,
            "cp_varying": bool(cp_varying),
            "balance_rel": balance_rel, "submesh": sub,
            "iron_idx": iron_idx, "used_nodes": used_nodes,
            "iron_volumes": vols_sub, "a": a, "v": v,
            "em_info": em_info}
