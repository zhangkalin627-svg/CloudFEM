"""BIG-M9: full induction-heating chain (axisymmetric EM -> transient heat).

Chain (all CloudFEM, read-only import):
  EM   : cloudfem.physics.axisym.solve_eddy_current_axi
         (utilde = r*A_phi, P1 triangles; same excitation path as BIG-M8:
         Dirichlet utilde=0 on the axis, Neumann g = 2*pi*H0 on r = a,
         i.e. the exact surface condition H_z(a) = H0 of an infinitely
         long solenoid). Joule density from joule_heating_density_axi.
  Heat : repo assembly (fem.poisson.assemble, fem.timestep.mass_matrix,
         fem.timestep.source_load_vector_elem) with axisymmetric weights
         2*pi*r, implicit Euler, plus a Robin boundary term on r = a
         (natural convection + surface radiation, radiation coefficient
         linearised on the previous step's surface temperature).

Configuration
-------------
Steel bar, DIAMETER 20 mm -> radius a = 10 mm (see report for the
calibre discussion vs BIG-M8, where a = 20 mm was the radius). The bar
is modelled as infinitely long (r-z slab, z-invariant fields), i.e. all
powers/energies are per unit length; the cylindrical surface r = a
loses heat by convection (h = 10 W/m^2/K) and radiation (eps = 0.8) to
T_inf = 20 C; end faces carry no flux by z-invariance (mid-section of
a long bar). Initial temperature 20 C everywhere, heating time 60 s.

Excitation: f = 10 kHz, H0 = 60 kA/m, mu = mu0 (mu_r = 1, constant).
Cold skin depth delta = 3.56 mm, a/delta = 2.81.

Materials (steel, engineering fits, stated in the report):
  sigma(T) = sigma0 / (1 + alpha_s*(T-20)), sigma0 = 2.0e6 S/m,
             alpha_s = 0.004 /K   (main case; control case freezes
             sigma = sigma0 and isolates the sigma(T) feedback only)
  cp(T)    = 450 + 0.25*(T-20)  J/(kg K)
  k(T)     = 48 - 0.025*(T-20)  W/(m K), floored at 15
  rho      = 7850 kg/m^3 (constant)

Coupling: explicit (staggered) -- each heat step first updates the
elementwise sigma/k/cp from the current temperature, re-solves the EM
problem, then advances the heat problem with that step's Joule source.

Run:  python3 tests/test_bench_induction_chain.py   (from the repo root)
Integrated into the CloudFEM suite by BIG-M11; physics unchanged
from the standalone BIG-M9 script.
Outputs: console log, big_m9_curves.csv (t, T_surf, T_core, P per case),
big_m9_profile.csv (radial T profile at t = 60 s, main case).
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.special import ber, bei, berp, beip

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from cloudfem.mesh.tri_mesh import TriMesh
from cloudfem.fem.poisson import assemble
from cloudfem.fem.timestep import mass_matrix, source_load_vector_elem
from cloudfem.physics.axisym import (
    solve_eddy_current_axi, joule_heating_density_axi,
    total_joule_power_axi, MU0, TWO_PI)

# ------------------------------------------------------------------ parameters
A = 0.010            # bar radius [m]  (diameter 20 mm)
LZ = A               # r-z slab height [m] (z-invariant solution)
NZ = 4               # axial divisions
GROWTH = 1.18        # radial geometric grading ratio (as BIG-M8)
FREQ = 10_000.0      # [Hz]
OMEGA = 2.0 * np.pi * FREQ
H0 = 60_000.0        # solenoid field amplitude [A/m]
MU = MU0             # mu_r = 1, constant

SIGMA0 = 2.0e6       # conductivity at 20 C [S/m]
ALPHA_S = 0.004      # sigma temperature coefficient [1/K]
RHO = 7850.0         # density [kg/m^3]
CP0, CP_B = 450.0, 0.25        # cp(T) = CP0 + CP_B*(T-20) [J/(kg K)]
K0, K_B = 48.0, -0.025         # k(T)  = K0 + K_B*(T-20)  [W/(m K)]
K_FLOOR = 15.0

T0 = 20.0            # initial temperature [C]
T_INF = 20.0         # ambient [C]
H_CONV = 10.0        # natural convection [W/(m^2 K)]
EPS_RAD = 0.8        # emissivity [-]
SB = 5.67e-8         # Stefan-Boltzmann [W/(m^2 K^4)]

T_END = 60.0         # heating time [s]
DT = 0.25            # heat time step [s]
H_SURF_FRAC = 1.0 / 16.0   # target surface layer / cold delta

DELTA0 = np.sqrt(2.0 / (OMEGA * MU * SIGMA0))


# ------------------------------------------------------------------ materials
def sigma_of_T(T):
    return SIGMA0 / (1.0 + ALPHA_S * (T - T0))


def cp_of_T(T):
    return CP0 + CP_B * (T - T0)


def k_of_T(T):
    return np.maximum(K0 + K_B * (T - T0), K_FLOOR)


def enthalpy_per_kg(T):
    """int_{T0}^{T} cp dT for the linear cp(T) fit [J/kg]."""
    d = T - T0
    return CP0 * d + 0.5 * CP_B * d ** 2


# ------------------------------------------------------------------ mesh
def graded_slab_mesh(h_surf):
    """Same generator as BIG-M8: tensor r-z mesh, geometric radial
    grading toward r = a closing exactly on A."""
    q = GROWTH
    m = 1
    while h_surf * (q ** m - 1.0) / (q - 1.0) < A:
        m += 1
    h0 = A * (q - 1.0) / (q ** m - 1.0)
    thick = h0 * q ** np.arange(m)
    edges_from_surf = np.concatenate(([0.0], np.cumsum(thick)))
    r_nodes = A - edges_from_surf[::-1]
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
    return TriMesh(nodes, np.array(elements), boundaries), nr, r_nodes


# ------------------------------------------------------------------ analytic
def analytic_power_cold():
    """Kelvin closed form, cold state (sigma = SIGMA0 uniform), per unit
    length -- independent reference for the t = 0 FEM power."""
    x = np.sqrt(2.0) * A / DELTA0
    num = ber(x) * berp(x) + bei(x) * beip(x)
    den = ber(x) ** 2 + bei(x) ** 2
    return float(np.pi * A * H0 ** 2 * np.sqrt(2.0) / (SIGMA0 * DELTA0)
                 * num / den)


# ------------------------------------------------------------------ chain
class Chain:
    def __init__(self, mesh):
        self.mesh = mesh
        cent = mesh.nodes[mesh.elements].mean(axis=1)   # (M, 2)
        self.cent = cent
        self.tree = cKDTree(cent)
        self.vol_e = TWO_PI * cent[:, 0] * mesh.element_areas()  # m^3/m... per slab
        self.sig_e = np.full(mesh.n_elements, SIGMA0)
        self.k_e = np.full(mesh.n_elements, k_of_T(T0))
        self.cp_e = np.full(mesh.n_elements, cp_of_T(T0))
        self._build_robin_base()

    def _elem_lookup(self, arr):
        def fn(x, y):
            _, idx = self.tree.query([x, y])
            return float(arr[idx])
        return fn

    def _build_robin_base(self):
        """Unit-h Robin matrix/rhs on r = a edges (axi weight 2*pi*a)."""
        mesh = self.mesh
        n = mesh.n_nodes
        rows, cols, data = [], [], []
        rvec = np.zeros(n)
        for i, j in mesh.boundary_edges():
            pi, pj = mesh.nodes[i], mesh.nodes[j]
            if abs(pi[0] - A) < 1e-12 and abs(pj[0] - A) < 1e-12:
                ell = float(np.linalg.norm(pj - pi))
                w = TWO_PI * A * ell
                for ii, jj, v in ((i, i, 2.0), (i, j, 1.0),
                                  (j, i, 1.0), (j, j, 2.0)):
                    rows.append(ii); cols.append(jj); data.append(w * v / 6.0)
                rvec[i] += w / 2.0
                rvec[j] += w / 2.0
        self.H_base = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        self.r_base = rvec

    def em_solve(self):
        mesh = self.mesh
        sig_fn = self._elem_lookup(self.sig_e)
        u = solve_eddy_current_axi(
            mesh, nu=1.0 / MU, sigma=sig_fn, omega=OMEGA,
            bc={"left": 0.0}, neumann={"right": TWO_PI * H0})
        Q_elem, _ = joule_heating_density_axi(mesh, u, sig_fn, OMEGA)
        P = total_joule_power_axi(mesh, u, sig_fn, OMEGA)
        return Q_elem, P / LZ   # per unit length [W/m]

    def run(self, sigma_feedback=True):
        mesh = self.mesh
        n_steps = int(round(T_END / DT))
        surf_nodes = mesh.boundaries["right"]
        axis_nodes = mesh.boundaries["left"]

        T = np.full(mesh.n_nodes, T0)
        self.sig_e[:] = SIGMA0
        self.k_e[:] = k_of_T(T0)
        self.cp_e[:] = cp_of_T(T0)

        k_axi = lambda x, y: TWO_PI * x * self._elem_lookup(self.k_e)(x, y)
        rc_axi = lambda x, y: TWO_PI * x * RHO * self._elem_lookup(self.cp_e)(x, y)

        times = [0.0]
        Ts_hist, Tc_hist, P_hist = [T0], [T0], []
        E_loss = 0.0
        P_first = None

        for step in range(n_steps):
            Te = T[mesh.elements].mean(axis=1)
            if sigma_feedback:
                self.sig_e = sigma_of_T(Te)
            self.k_e = k_of_T(Te)
            self.cp_e = cp_of_T(Te)

            Q_elem, P_L = self.em_solve()
            if P_first is None:
                P_first = P_L
            P_hist.append(P_L)

            rhs_q = source_load_vector_elem(
                mesh, Q_elem, weight=lambda r: TWO_PI * r)
            K, _ = assemble(mesh, k=k_axi)
            M = mass_matrix(mesh, rc_axi, lumped=True)

            # Robin coefficient from previous-step surface temperature
            Ts_prev = float(T[surf_nodes].mean())
            TsK, TiK = Ts_prev + 273.15, T_INF + 273.15
            h_rad = EPS_RAD * SB * (TsK + TiK) * (TsK ** 2 + TiK ** 2)
            h_eff = H_CONV + h_rad

            A_sys = (M / DT + K + h_eff * self.H_base).tocsc()
            rhs = M @ T / DT + rhs_q + h_eff * T_INF * self.r_base
            T = np.asarray(spla.spsolve(A_sys, rhs)).ravel()

            # surface loss over this step (end-of-step surface T)
            Ts_now = float(T[surf_nodes].mean())
            p_loss = (H_CONV * (Ts_now - T_INF)
                      + EPS_RAD * SB * ((Ts_now + 273.15) ** 4 - TiK ** 4)
                      ) * TWO_PI * A
            E_loss += p_loss * DT

            times.append((step + 1) * DT)
            Ts_hist.append(Ts_now)
            Tc_hist.append(float(T[axis_nodes].mean()))

        # enthalpy rise per unit length (slab volumes / LZ)
        Te_end = T[mesh.elements].mean(axis=1)
        dH = float(np.sum(RHO * self.vol_e * enthalpy_per_kg(Te_end))) / LZ
        E_in_L = float(np.sum(P_hist)) * DT   # P_hist already per unit length
        return {
            "times": np.array(times), "T_surf": np.array(Ts_hist),
            "T_core": np.array(Tc_hist), "P": np.array(P_hist),
            "T_final": T, "P_first": P_first, "P_last": P_hist[-1],
            "E_in": E_in_L, "E_loss": E_loss, "dH": dH,
        }


def radial_profile(mesh, T, r_nodes):
    """Nodal T averaged over z rows at each radial node station."""
    nr = len(r_nodes)
    prof = []
    for i, r in enumerate(r_nodes):
        col = [j * nr + i for j in range(NZ + 1)]
        prof.append(float(T[col].mean()))
    return np.array(prof)


def main():
    t_start = time.time()
    print("BIG-M9  full induction-heating chain (axi EM -> transient heat)")
    print(f"  bar: diameter 20 mm (a = {A*1e3:.0f} mm), long-bar axi model, "
          f"per-unit-length quantities")
    print(f"  f = {FREQ:.0f} Hz, H0 = {H0:.0f} A/m, mu_r = 1, "
          f"sigma0 = {SIGMA0:.3e} S/m, delta_cold = {DELTA0*1e3:.2f} mm, "
          f"a/delta = {A/DELTA0:.2f}")
    print(f"  heat: {T_END:.0f} s, dt = {DT} s, surface loss h = {H_CONV} "
          f"W/m2K + radiation eps = {EPS_RAD}, T_inf = {T_INF} C")

    mesh, nr, r_nodes = graded_slab_mesh(H_SURF_FRAC * DELTA0)
    print(f"  mesh: nr = {nr} radial layers, {mesh.n_elements} elements, "
          f"{mesh.n_nodes} nodes")

    p_ana = analytic_power_cold()
    chain = Chain(mesh)
    res = chain.run(sigma_feedback=True)
    err0 = (res["P_first"] - p_ana) / p_ana
    print(f"  t=0 power check: FEM P/L = {res['P_first']:.1f} W/m vs "
          f"analytic (Kelvin, cold) = {p_ana:.1f} W/m  ({err0*100:+.2f}%)")

    res_c = chain.run(sigma_feedback=False)

    def summary(tag, r):
        bal = r["E_in"] - r["dH"] - r["E_loss"]
        print(f"  [{tag}] P/L: {r['P_first']:.0f} -> {r['P_last']:.0f} W/m; "
              f"T_surf(60s) = {r['T_surf'][-1]:.1f} C, "
              f"T_core(60s) = {r['T_core'][-1]:.1f} C")
        print(f"  [{tag}] energy per m: in = {r['E_in']:.4e} J, "
              f"enthalpy = {r['dH']:.4e} J, loss = {r['E_loss']:.4e} J, "
              f"residual = {bal:.3e} J ({bal/r['E_in']*100:+.2f}%)")
        return bal / r["E_in"]

    print("  --- main case: sigma(T) feedback ---")
    bal_main = summary("main", res)
    print("  --- control: sigma frozen at sigma0 ---")
    bal_ctrl = summary("ctrl", res_c)

    # sampled histories (every 10 s)
    print("  t[s]   Tsurf_main  Tcore_main   P_main | Tsurf_ctrl  "
          "Tcore_ctrl   P_ctrl")
    for tt in range(0, int(T_END) + 1, 10):
        i = int(round(tt / DT))
        ip = min(i, len(res["P"]) - 1)
        print(f"  {tt:4d}   {res['T_surf'][i]:9.1f}  {res['T_core'][i]:9.1f}"
              f"  {res['P'][ip]:8.0f} | {res_c['T_surf'][i]:9.1f}  "
              f"{res_c['T_core'][i]:9.1f}  {res_c['P'][ip]:8.0f}")

    prof = radial_profile(mesh, res["T_final"], r_nodes)
    print("  radial profile at t = 60 s (main case):")
    print("    r[mm]      T[C]")
    for r, Tv in zip(r_nodes, prof):
        print(f"    {r*1e3:6.2f}  {Tv:8.1f}")

    # CSV outputs
    hdr = "t_s,Tsurf_main_C,Tcore_main_C,P_main_Wpm,Tsurf_ctrl_C," \
          "Tcore_ctrl_C,P_ctrl_Wpm"
    P_ext = np.concatenate(([res["P"][0]], res["P"]))
    Pc_ext = np.concatenate(([res_c["P"][0]], res_c["P"]))
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "big_m9_curves.csv"),
               np.column_stack([res["times"], res["T_surf"], res["T_core"],
                                P_ext, res_c["T_surf"], res_c["T_core"],
                                Pc_ext]),
               delimiter=",", header=hdr, comments="", fmt="%.4f")
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "big_m9_profile.csv"),
               np.column_stack([r_nodes * 1e3, prof]),
               delimiter=",", header="r_mm,T_C", comments="", fmt="%.4f")

    if "--check-fine" in sys.argv:
        mesh_f, nr_f, _ = graded_slab_mesh(DELTA0 / 32.0)
        res_f = Chain(mesh_f).run(sigma_feedback=True)
        dT = res_f["T_surf"][-1] - res["T_surf"][-1]
        print(f"  mesh check (h_surf = delta/32, nr = {nr_f}): "
              f"T_surf(60s) = {res_f['T_surf'][-1]:.1f} C "
              f"(base {res['T_surf'][-1]:.1f} C, diff {dT:+.1f} K), "
              f"T_core = {res_f['T_core'][-1]:.1f} C")

    print(f"  wall time: {time.time() - t_start:.0f} s")
    # BIG-M11 benchmark criteria: main-case energy-balance residual
    # < 1 % and t = 0 cold-state power vs the analytic Kelvin value
    # < 2 % (both quantities computed above, unchanged).
    ok_bal = abs(bal_main) < 0.01
    ok_cold = abs(err0) < 0.02
    print(f"VERDICT: energy-balance residual = {bal_main * 100:+.2f}% "
          f"(criterion < 1%), t=0 power error = {err0 * 100:+.2f}% "
          f"(criterion < 2%) -> "
          f"{'PASS' if ok_bal and ok_cold else 'FAIL'}")
    assert ok_bal, "main-case energy balance off (>= 1%)"
    assert ok_cold, "t=0 cold power off analytic by >= 2%"
    print("BIG-M9 BENCHMARK PASSED")


if __name__ == "__main__":
    main()
