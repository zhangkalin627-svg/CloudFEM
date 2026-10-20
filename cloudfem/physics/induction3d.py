"""3D sequential EM -> thermal -> structural coupling (v1.4).

The flagship chain, joining the previously separate 3D solvers into
one driver:

    applied field / source current
      -> v1.3 edge-element eddy currents (physics.em3d)
      -> element Joule heating Q = 0.5 * sigma * |E|^2
      -> v1.0 transient heat conduction on the workpiece (fem.tet_heat)
      -> v0.8 thermal stress at output snapshots (physics.thermomech)

with the electrical conductivity updated from the temperature field
between EM re-solves.

Material law for sigma(T) -- the convention of the 2D axisymmetric
chain (examples/induction_heating_axi.py), used unchanged::

    sigma(T) = sigma_ref / (1 + alpha_sig * (T - T_sigma_ref))

with sigma_ref the conductivity at T_sigma_ref (for carbon steel
the 2D chain uses sigma_ref = 2e6 S/m at 20 C, alpha_sig = 0.004 /K).
The reluctivity nu is *constant* (default 1/mu0): the Curie
transition of mu is not modelled, which is the same convention as
the 2D reference chain and is exact above the Curie point; below it,
a constant-mu model misestimates the cold coupling of real steel and
that limitation is inherited knowingly (see README v1.4 row).

Update strategy (explicit / quasi-iterative, as in the 2D chain):
sigma is frozen within each EM interval of `em_every` heat steps and
re-evaluated from the element temperatures (nodal means) at the
interval start. The Joule source is therefore piecewise constant in
time and the coupling is first order in the interval length. (The
interval-refinement error is *not* separately quantified in v1.4 --
the validation fixes em_every = 3 -- and is the first candidate
for a follow-up study.) With alpha_sig = 0 the EM problem is
time-independent and is solved exactly once.

Energy accounting: the workpiece surface is adiabatic in the heat
solve, so the discrete energy balance closes identically -- with the
element-constant source load of fem.tet_heat, ones^T C (T - T0)
increases by exactly P * dt_interval per interval, where P is the
EM-side total Joule power and C the consistent capacity matrix
(ones^T K = 0 for a pure Neumann problem). The driver reports both
sides (`E_in` accumulated from the EM solves, `enthalpy_gain` from
the final temperature field); their relative difference is a
regression check on the Q transfer (full-mesh elements -> workpiece
submesh), which is exact here because both solvers evaluate element
fields at the same centroids (the transfer callable is a
nearest-centroid lookup that hits its own element at distance 0;
the same pattern the v1.3 demo used).

Accuracy limitations inherited from the parts (not hidden):
  * v1.3 edge elements: at h ~ delta the cylinder loss carries the
    skin-resolution / end-effect error documented in
    tests/test_em3d.py (tens of percent at h/delta ~ 1.5).
  * Faceted workpiece surface: the mesh volume underestimates the
    analytic cylinder volume by O(h^2) (~4 % at h = 0.011, R = 0.02).
  * The heat solve runs on the workpiece alone: heat loss into the
    surrounding air/coil is neglected (adiabatic workpiece), exactly
    as in the 2D reference chain.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..fem.tet_edge import build_edges
from ..fem.tet_heat import assemble_mass_tet, transient_heat_tet
from ..mesh.tet_mesh import TetMesh
from .em import MU0
from .em3d import joule_heating_density_3d, solve_eddy_current_3d
from .thermomech import transient_thermal_stress_3d


def steel_sigma_of_T(T, sigma_ref, alpha_sig=0.004, T_ref=20.0):
    """sigma(T) = sigma_ref / (1 + alpha_sig * (T - T_ref)).

    The temperature law of the 2D axisymmetric chain; T may be a
    scalar or an array. alpha_sig = 0 recovers constant sigma.
    """
    return sigma_ref / (1.0 + alpha_sig * (np.asarray(T, dtype=float)
                                          - T_ref))


def thermal_enthalpy(mesh, T, rho_cp=1.0, T_ref=0.0):
    """Total heat content relative to T_ref: ones^T C (T - T_ref).

    C is the consistent P1 capacity matrix of fem.tet_heat, so this
    is the exact discrete enthalpy of the nodal field T.
    """
    C = assemble_mass_tet(mesh, rho_cp=rho_cp)
    dT = np.asarray(T, dtype=float) - T_ref
    return float(np.sum(C @ dT))


def extract_submesh(mesh, elem_mask):
    """Extract the masked elements as a standalone TetMesh.

    Returns (submesh, used_nodes, elem_indices): submesh node j is
    full-mesh node used_nodes[j], and submesh element i is full-mesh
    element elem_indices[i] (mask order preserved). Boundary tags
    are not carried over.
    """
    elem_mask = np.asarray(elem_mask, dtype=bool)
    elem_idx = np.where(elem_mask)[0]
    els = mesh.elements[elem_idx]
    used, inv = np.unique(els, return_inverse=True)
    sub = TetMesh(mesh.nodes[used], inv.reshape(-1, 4))
    return sub, used, elem_idx


def _centroid_lookup(centroids, values):
    """Callable(x, y, z) -> values of the nearest centroid.

    Exact for queries made at the centroids themselves (distance 0),
    which is where every solver in the chain evaluates element
    fields.
    """
    tree = cKDTree(np.asarray(centroids, dtype=float))
    values = np.asarray(values, dtype=float)

    def fn(x, y, z):
        _, idx = tree.query([x, y, z])
        return float(values[idx])

    return fn


def _pin_321(mesh):
    """Default rigid-body pins (3-2-1) at bounding-box corner nodes.

    Registers 'pin_a'/'pin_b'/'pin_c' one-node boundary sets on the
    mesh and returns the bc dict for thermal_stress_3d: the node
    nearest (xmin, ymin, zmin) fully fixed, the node nearest
    (xmax, ymin, zmin) fixed in y/z, the node nearest
    (xmin, ymax, zmin) fixed in z only -- the v1.0 quench-demo
    scheme, which removes rigid-body motion without constraining
    thermal expansion.
    """
    lo = mesh.nodes.min(axis=0)
    hi = mesh.nodes.max(axis=0)

    def nearest(point):
        d2 = np.sum((mesh.nodes - np.asarray(point)) ** 2, axis=1)
        return int(np.argmin(d2))

    mesh.boundaries["pin_a"] = np.array(
        [nearest((lo[0], lo[1], lo[2]))])
    mesh.boundaries["pin_b"] = np.array(
        [nearest((hi[0], lo[1], lo[2]))])
    mesh.boundaries["pin_c"] = np.array(
        [nearest((lo[0], hi[1], lo[2]))])
    return {"pin_a": (0.0, 0.0, 0.0),
            "pin_b": (None, 0.0, 0.0),
            "pin_c": (None, None, 0.0)}


def induction_heating_3d(mesh, sigma_ref, omega, B_applied=None,
                         J_s=None, nu=1.0 / MU0, alpha_sig=0.0,
                         T_sigma_ref=None, k=40.0, rho_cp=3.5e6,
                         T0=20.0, t_end=10.0, dt=0.5, em_every=1,
                         method="cn", E=None, nu_p=0.3,
                         alpha_th=12e-6, T_stress_ref=None,
                         stress_every=1, pins="default",
                         gauge_eps=1e-6, edge_data=None):
    """Run the 3D induction-heating chain EM -> heat (-> stress).

    Parameters
    ----------
    mesh : TetMesh covering workpiece + surrounding region (e.g.
        from mesh.unstructured3d.cylinder_in_box_mesh). Elements
        with sigma_ref > 0 at the centroid form the workpiece; the
        heat (and stress) solves run on that submesh alone.
    sigma_ref : float or callable(x, y, z), conductivity at
        T_sigma_ref [S/m] (the spatial material distribution).
    omega : angular frequency [rad/s].
    B_applied, J_s : excitation, as in em3d.solve_eddy_current_3d.
    nu : float or callable, reluctivity (constant in time).
    alpha_sig : temperature coefficient of sigma (module docstring);
        0 = constant sigma (EM solved once).
    T_sigma_ref : reference temperature of sigma_ref (default T0).
    k, rho_cp : workpiece thermal conductivity / volumetric heat
        capacity, floats or callables(x, y, z) (workpiece values).
    T0 : uniform initial temperature [C].
    t_end, dt : horizon and heat step (the step used is
        t_end / ceil(t_end / dt), as in fem.tet_heat).
    em_every : EM re-solve interval in heat steps (ignored when
        alpha_sig = 0: one solve is then exact).
    method : heat time integrator, "cn" or "be" (fem.tet_heat).
    E, nu_p, alpha_th : if E is given, thermal-stress snapshots are
        computed on the workpiece with Young's modulus E, Poisson
        ratio nu_p and expansion coefficient alpha_th.
    T_stress_ref : stress-free temperature (default T0).
    stress_every : stress snapshot interval in heat steps (the final
        step is always included).
    pins : "default" (3-2-1 rigid-body pins at the workpiece
        bounding-box corners, see _pin_321), None (no stress BC --
        only meaningful if the elasticity solve is otherwise
        constrained; the free thermal problem is singular), or a list
        of ((x, y, z), (ux, uy, uz)) pairs, each pinned at the
        workpiece node nearest the point, None = free component.
    gauge_eps, edge_data : passed to the EM solver.

    Returns
    -------
    dict with: times (S+1,), T_history (S+1, Nb) on the workpiece
    mesh, workpiece (the submesh), elem_idx (workpiece element
    indices in the full mesh), T_mean (S+1,) volume-averaged
    workpiece temperature, P_history (S,) Joule power applied during
    each step, em_times / P_em (power at each EM solve),
    sigma_min (smallest workpiece sigma at each EM solve),
    n_em_solves, E_in (accumulated EM energy [J]), enthalpy_gain
    (final - initial discrete enthalpy [J]), energy_rel_err, Q_elem
    (last Joule density on the full mesh), and -- if E is given --
    the transient_thermal_stress_3d entries (vm_max, u_max,
    peak_time, peak_vm_max, peak, history, with its own times).
    """
    if T_sigma_ref is None:
        T_sigma_ref = T0
    if T_stress_ref is None:
        T_stress_ref = T0
    if edge_data is None:
        edge_data = build_edges(mesh)

    cen = mesh.nodes[mesh.elements].mean(axis=1)
    vols_full = mesh.element_volumes()
    if callable(sigma_ref):
        sig_ref_e = np.array([sigma_ref(x, y, z) for x, y, z in cen])
    else:
        sig_ref_e = np.full(mesh.n_elements, float(sigma_ref))
    conductor = sig_ref_e > 0.0
    if not conductor.any():
        raise ValueError("no conducting elements: sigma_ref <= 0 "
                         "everywhere")
    workpiece, used_nodes, elem_idx = extract_submesh(mesh, conductor)
    vols_wp = vols_full[elem_idx]
    cen_wp = workpiece.nodes[workpiece.elements].mean(axis=1)

    n_steps = int(np.ceil(t_end / dt - 1e-12))
    if n_steps < 1:
        raise ValueError("t_end must be positive")
    h = t_end / n_steps
    if alpha_sig == 0.0:
        em_every = n_steps  # sigma constant: a single EM solve is exact
    em_every = max(1, int(em_every))

    T = np.full(workpiece.n_nodes, float(T0))
    times = [0.0]
    T_hist = [T.copy()]
    T_mean = [float(T0)]
    P_hist = []
    em_times, P_em, sigma_min = [], [], []
    E_in = 0.0
    Q_full = np.zeros(mesh.n_elements)

    step = 0
    while step < n_steps:
        m = min(em_every, n_steps - step)
        # --- EM solve with sigma(T) frozen at the interval start ---
        T_elem = T[workpiece.elements].mean(axis=1)
        sig_e = sig_ref_e.copy()
        sig_e[elem_idx] = steel_sigma_of_T(T_elem, sig_ref_e[elem_idx],
                                           alpha_sig, T_sigma_ref)
        # Element-wise sigma through an exact centroid lookup (a
        # plain float would only do for a spatially uniform, constant
        # sigma distributed over part of the mesh -- never the case
        # here, since air elements carry sigma = 0).
        sigma_fn = _centroid_lookup(cen, sig_e)
        a, v = solve_eddy_current_3d(mesh, nu=nu, sigma=sigma_fn,
                                     omega=omega, B_applied=B_applied,
                                     J_s=J_s, gauge_eps=gauge_eps,
                                     edge_data=edge_data)
        Q_full, _ = joule_heating_density_3d(mesh, a, v, sigma_fn,
                                             omega, edge_data=edge_data)
        P = float(np.sum(Q_full * vols_full))
        em_times.append(step * h)
        P_em.append(P)
        sigma_min.append(float(sig_e[elem_idx].min()))
        E_in += P * (m * h)

        # --- heat chunk on the workpiece with Q frozen --------------
        Q_wp = Q_full[elem_idx]
        heat_fn = _centroid_lookup(cen_wp, Q_wp)
        _, hist = transient_heat_tet(workpiece, k=k, rho_cp=rho_cp,
                                     f=heat_fn, t_end=m * h, dt=h,
                                     T0=T, method=method)
        for j in range(1, m + 1):
            T = hist[j]
            times.append((step + j) * h)
            T_hist.append(T.copy())
            Te = T[workpiece.elements].mean(axis=1)
            T_mean.append(float(np.sum(Te * vols_wp) / np.sum(vols_wp)))
            P_hist.append(P)
        step += m

    times = np.array(times)
    times[-1] = t_end
    T_hist = np.array(T_hist)
    enthalpy_gain = thermal_enthalpy(workpiece, T_hist[-1], rho_cp,
                                     T_ref=T0)
    res = {
        "times": times,
        "T_history": T_hist,
        "workpiece": workpiece,
        "elem_idx": elem_idx,
        "T_mean": np.array(T_mean),
        "P_history": np.array(P_hist),
        "em_times": np.array(em_times),
        "P_em": np.array(P_em),
        "sigma_min": np.array(sigma_min),
        "n_em_solves": len(em_times),
        "E_in": float(E_in),
        "enthalpy_gain": float(enthalpy_gain),
        "energy_rel_err": float(abs(E_in - enthalpy_gain) / E_in)
        if E_in > 0 else 0.0,
        "Q_elem": Q_full,
    }

    if E is not None:
        if pins == "default":
            bc_el = _pin_321(workpiece)
        elif pins is None:
            bc_el = None
        else:
            bc_el = {}
            for i, (point, dofs) in enumerate(pins):
                name = f"pin_{i}"
                d2 = np.sum((workpiece.nodes
                             - np.asarray(point, dtype=float)) ** 2,
                            axis=1)
                workpiece.boundaries[name] = np.array([int(np.argmin(d2))])
                bc_el[name] = dofs
        sel = list(range(0, n_steps + 1, max(1, int(stress_every))))
        if sel[-1] != n_steps:
            sel.append(n_steps)
        chain = transient_thermal_stress_3d(
            workpiece, times[sel], T_hist[sel], E, nu_p, alpha_th,
            T_ref=T_stress_ref, bc=bc_el)
        res.update({
            "stress_times": chain["times"],
            "vm_max": chain["vm_max"],
            "u_max": chain["u_max"],
            "peak_time": chain["peak_time"],
            "peak_vm_max": chain["peak_vm_max"],
            "peak": chain["peak"],
            "stress_history": chain["history"],
        })
    return res
