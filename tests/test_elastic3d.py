"""Validation of the 3D linear elasticity module.

Voigt order throughout: [xx, yy, zz, yz, xz, xy]; DOFs are interleaved
3*i + c (used here only implicitly, through (ux, uy, uz) tuples).
All analytical values below are derived from linear elasticity theory
(Lame parameters, thermal strain), never from the solver under test.

1. Uniaxial tension of a unit cube (traction tx = 1e8 Pa on xmax,
   ux = 0 on xmin, two extra pins consistent with the solution):
   P1 elements reproduce the affine solution exactly, so
       sigma_xx = 1e8 everywhere, all other components 0,
       ux(x = 1) = 1e8 / E = 5e-4,
       uy(0, 1, 0) = -nu * 5e-4.
2. Fully constrained block, uniform heating dT = 50 (all six faces fixed):
       sigma_xx = sigma_yy = sigma_zz
                = -E * alpha * dT / (1 - 2*nu) = -3.0e8,
       shear = 0, displacement = 0.
3. Free expansion, uniform heating dT = 50 (pins only, consistent with
   the free solution): u = alpha * dT * (x, y, z) everywhere, sigma = 0.
4. Affine patch test: an affine displacement prescribed on all six
   faces must be reproduced exactly at interior nodes, with constant
   strain eps0 = [0.01, -0.005, 0.003, 0.01, 0.004, 0.02] and constant
   stress sigma = lam * tr(eps) * I + 2 * mu * eps_tensor (Lame form,
   computed independently of constitutive_3d).
5. Mesh convergence (manufactured solution): u = (sin(pi x), 0, 0)
   with E = 1, nu = 0.3 satisfies -div sigma = b with the body force
   b = ((lam + 2*mu) * pi^2 * sin(pi x), 0, 0); Dirichlet data on all
   six faces is the exact solution. P1 must show O(h^2) in the nodal
   displacement L2 (RMS) error. (A polynomial manufactured solution
   such as u = (x^2, 0, 0) is nodally exact on uniform Kuhn meshes --
   its error sits at roundoff level -- so a trigonometric one is used
   to measure the true rate.)
6. Sequential thermal -> structural chain: solve_tet_poisson with
   f = 0 and T = 50 on all six faces gives a uniform T = 50 field;
   thermal_stress_3d with all faces fixed then gives the test-2
   stresses sigma_xx = sigma_yy = sigma_zz = -3.0e8.
"""

import numpy as np

from cloudfem.mesh.tet_mesh import box_mesh
from cloudfem.fem.tet_poisson import solve_tet_poisson
from cloudfem.physics.elastic3d import (
    solve_elasticity_3d, element_stress_strain_3d, von_mises_3d)
from cloudfem.physics.thermomech import thermal_stress_3d

# steel-ish properties
E_STEEL = 200e9
NU_STEEL = 0.3
ALPHA_STEEL = 12e-6

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _nearest_node(mesh, point):
    """Index of the mesh node closest to the (x, y, z) point."""
    d2 = np.sum((mesh.nodes - np.asarray(point, dtype=float)) ** 2, axis=1)
    return int(np.argmin(d2))


def _add_pin(mesh, name, point):
    """Register the single node nearest to point as boundary `name`.

    Same pattern as _pinned_disk_mesh in test_elastic.py: a custom
    one-node entry in mesh.boundaries, addressed by bc/traction dicts.
    Returns the node index.
    """
    idx = _nearest_node(mesh, point)
    mesh.boundaries[name] = np.array([idx])
    return idx


def _lame(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def test_uniaxial_tension():
    """Test 1: uniaxial tension, affine-exact for P1."""
    mesh = box_mesh(4, 4, 4)
    _add_pin(mesh, "pin_a", (0.0, 0.0, 0.0))
    node_b = _add_pin(mesh, "pin_b", (0.0, 1.0, 0.0))
    p = 1e8  # traction tx on xmax, Pa
    ux, uy, uz = solve_elasticity_3d(
        mesh, E_STEEL, NU_STEEL,
        bc={"xmin": (0.0, None, None),
            "pin_a": (None, 0.0, 0.0),
            "pin_b": (None, None, 0.0)},
        traction={"xmax": (p, 0.0, 0.0)})
    sig, _ = element_stress_strain_3d(mesh, (ux, uy, uz),
                                      E_STEEL, NU_STEEL)
    s_exact = p
    err_xx = np.abs(sig[:, 0] - s_exact).max() / s_exact
    err_other = np.abs(sig[:, 1:]).max() / s_exact
    err_vm = np.abs(von_mises_3d(sig) - s_exact).max() / s_exact
    ux_exact = p / E_STEEL  # 5e-4
    xmax = mesh.boundaries["xmax"]
    err_ux = np.abs(ux[xmax] - ux_exact).max() / ux_exact
    uy_exact = -NU_STEEL * ux_exact  # at (0, 1, 0): uy = -nu * eps_xx * y
    err_uy = abs(uy[node_b] - uy_exact) / abs(uy_exact)
    return err_xx, err_other, err_vm, err_ux, err_uy, s_exact, ux_exact


def test_constrained_heating():
    """Test 2: fully constrained block, uniform heating."""
    mesh = box_mesh(4, 4, 4)
    dT = 50.0
    T = np.full(mesh.n_nodes, dT)
    bc = {name: (0.0, 0.0, 0.0) for name in FACES}
    ux, uy, uz = solve_elasticity_3d(mesh, E_STEEL, NU_STEEL, bc=bc,
                                     T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    sig, _ = element_stress_strain_3d(mesh, (ux, uy, uz),
                                      E_STEEL, NU_STEEL,
                                      T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    s_exact = -E_STEEL * ALPHA_STEEL * dT / (1.0 - 2.0 * NU_STEEL)  # -3.0e8
    err_normal = np.abs(sig[:, :3] - s_exact).max() / abs(s_exact)
    err_shear = np.abs(sig[:, 3:]).max() / abs(s_exact)
    # hydrostatic stress state => von Mises must vanish
    err_vm = von_mises_3d(sig).max() / abs(s_exact)
    u_max = max(np.abs(ux).max(), np.abs(uy).max(), np.abs(uz).max())
    return err_normal, err_shear, err_vm, u_max, s_exact


def test_free_expansion():
    """Test 3: free thermal expansion, uniform heating."""
    mesh = box_mesh(4, 4, 4)
    dT = 50.0
    T = np.full(mesh.n_nodes, dT)
    _add_pin(mesh, "pin_origin", (0.0, 0.0, 0.0))
    _add_pin(mesh, "pin_x", (1.0, 0.0, 0.0))
    _add_pin(mesh, "pin_y", (0.0, 1.0, 0.0))
    ux, uy, uz = solve_elasticity_3d(
        mesh, E_STEEL, NU_STEEL,
        bc={"pin_origin": (0.0, 0.0, 0.0),
            "pin_x": (None, 0.0, 0.0),
            "pin_y": (None, None, 0.0)},
        T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    sig, _ = element_stress_strain_3d(mesh, (ux, uy, uz),
                                      E_STEEL, NU_STEEL,
                                      T=T, alpha=ALPHA_STEEL, T_ref=0.0)
    u_num = np.column_stack([ux, uy, uz])
    u_exact = ALPHA_STEEL * dT * mesh.nodes  # alpha*dT*(x, y, z)
    # characteristic displacement: alpha*dT * L with L = 1
    err_u = np.abs(u_num - u_exact).max() / (ALPHA_STEEL * dT)
    err_sig = np.abs(sig).max() / (E_STEEL * ALPHA_STEEL * dT)
    return err_u, err_sig


def test_affine_patch():
    """Test 4: affine patch test."""
    mesh = box_mesh(4, 4, 4)

    def ux_fn(x, y, z):
        return 0.01 * x + 0.02 * y + 0.001

    def uy_fn(x, y, z):
        return -0.005 * y + 0.01 * z

    def uz_fn(x, y, z):
        return 0.003 * z + 0.004 * x - 0.002

    bc = {name: (ux_fn, uy_fn, uz_fn) for name in FACES}
    ux, uy, uz = solve_elasticity_3d(mesh, E_STEEL, NU_STEEL, bc=bc)
    sig, eps = element_stress_strain_3d(mesh, (ux, uy, uz),
                                        E_STEEL, NU_STEEL)

    # expected strain, Voigt with engineering shear strains:
    # gamma_yz = duz/dy + duy/dz = 0.01, gamma_xz = dux/dz + duz/dx
    # = 0.004, gamma_xy = dux/dy + duy/dx = 0.02
    eps0 = np.array([0.01, -0.005, 0.003, 0.01, 0.004, 0.02])

    # expected stress from the Lame form, independently of
    # constitutive_3d: sigma = lam*tr(eps)*I + 2*mu*eps_tensor
    lam, mu = _lame(E_STEEL, NU_STEEL)
    eps_tensor = np.array([
        [eps0[0], eps0[5] / 2.0, eps0[4] / 2.0],
        [eps0[5] / 2.0, eps0[1], eps0[3] / 2.0],
        [eps0[4] / 2.0, eps0[3] / 2.0, eps0[2]],
    ])
    sig_tensor = lam * np.trace(eps_tensor) * np.eye(3) \
        + 2.0 * mu * eps_tensor
    sig_exact = np.array([sig_tensor[0, 0], sig_tensor[1, 1],
                          sig_tensor[2, 2], sig_tensor[1, 2],
                          sig_tensor[0, 2], sig_tensor[0, 1]])

    err_sig = np.abs(sig - sig_exact).max() / np.abs(sig_exact).max()
    err_eps = np.abs(eps - eps0).max() / np.abs(eps0).max()

    # interior nodes: not in any of the six face node sets
    on_boundary = np.zeros(mesh.n_nodes, dtype=bool)
    for name in FACES:
        on_boundary[mesh.boundaries[name]] = True
    interior = np.where(~on_boundary)[0]
    assert len(interior) > 0, "patch test mesh has no interior nodes"
    x, y, z = (mesh.nodes[interior, 0], mesh.nodes[interior, 1],
               mesh.nodes[interior, 2])
    err_u = max(
        np.abs(ux[interior] - ux_fn(x, y, z)).max(),
        np.abs(uy[interior] - uy_fn(x, y, z)).max(),
        np.abs(uz[interior] - uz_fn(x, y, z)).max(),
    )
    return err_u, err_sig, err_eps, sig_exact


def test_convergence():
    """Test 5: manufactured solution u = (sin(pi x), 0, 0), E = 1."""
    E, nu = 1.0, 0.3
    lam, mu = _lame(E, nu)
    # eps_xx = pi*cos(pi x), sigma_xx = (lam + 2*mu)*pi*cos(pi x),
    # sigma_yy = sigma_zz = lam*pi*cos(pi x), shear 0
    #   =>  -div sigma = ((lam + 2*mu)*pi^2*sin(pi x), 0, 0)
    bx_fn = lambda x, y, z: (lam + 2.0 * mu) * np.pi ** 2 \
        * np.sin(np.pi * x)
    body = (bx_fn, 0.0, 0.0)
    ux_fn = lambda x, y, z: np.sin(np.pi * x)
    bc = {name: (ux_fn, 0.0, 0.0) for name in FACES}
    errs = []
    for n in (5, 10, 20):
        mesh = box_mesh(n, n, n)
        ux, uy, uz = solve_elasticity_3d(mesh, E, nu, bc=bc, body=body)
        x = mesh.nodes[:, 0]
        errs.append(np.sqrt(np.mean((ux - np.sin(np.pi * x)) ** 2
                                    + uy ** 2 + uz ** 2)))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2)
             for i in range(len(errs) - 1)]
    return errs, rates


def test_coupled_chain():
    """Test 6: heat solve (uniform T = 50) -> constrained thermal stress."""
    mesh = box_mesh(4, 4, 4)
    dT = 50.0
    bc_T = {name: dT for name in FACES}
    T = solve_tet_poisson(mesh, k=1.0, f=0.0, bc=bc_T)
    err_T = np.abs(T - dT).max()
    assert err_T < 1e-10, f"uniform temperature field off: {err_T}"

    bc = {name: (0.0, 0.0, 0.0) for name in FACES}
    res = thermal_stress_3d(mesh, T, E_STEEL, NU_STEEL, ALPHA_STEEL,
                            T_ref=0.0, bc=bc)
    sig = res["sig_elem"]
    s_exact = -E_STEEL * ALPHA_STEEL * dT / (1.0 - 2.0 * NU_STEEL)  # -3.0e8
    err_normal = np.abs(sig[:, :3] - s_exact).max() / abs(s_exact)
    err_shear = np.abs(sig[:, 3:]).max() / abs(s_exact)
    return err_T, err_normal, err_shear, s_exact, res["u_max"], res["vm_max"]


def main():
    err_xx, err_other, err_vm, err_ux, err_uy, s_exact, ux_exact = \
        test_uniaxial_tension()
    print(f"[uniaxial] sigma_exact = {s_exact/1e6:.1f} MPa, "
          f"rel err sxx = {err_xx:.3e}, other = {err_other:.3e}, "
          f"vm = {err_vm:.3e}; ux(x=1) exact = {ux_exact:.3e}, "
          f"rel err ux = {err_ux:.3e}, uy(0,1,0) rel err = {err_uy:.3e}")
    assert err_xx < 1e-8 and err_other < 1e-8 and err_vm < 1e-8, \
        "uniaxial stress validation failed (P1 affine exactness)"
    assert err_ux < 1e-8 and err_uy < 1e-8, \
        "uniaxial displacement validation failed (P1 affine exactness)"

    err_n, err_s, err_vm, u_max, s_exact = test_constrained_heating()
    print(f"[constrained heating] sigma_exact = {s_exact/1e6:.1f} MPa, "
          f"rel err normal = {err_n:.3e}, shear = {err_s:.3e}, "
          f"vm/|sigma| = {err_vm:.3e}, max|u| = {u_max:.3e}")
    assert err_n < 1e-6 and err_s < 1e-6 and err_vm < 1e-6, \
        "constrained heating validation failed"
    assert u_max < 1e-10, "constrained heating displacement must vanish"

    err_u, err_sig = test_free_expansion()
    print(f"[free expansion] max rel disp err = {err_u:.3e}, "
          f"max|sigma|/(E*alpha*dT) = {err_sig:.3e}")
    assert err_u < 1e-6 and err_sig < 1e-6, \
        "free expansion validation failed"

    err_u, err_sig, err_eps, sig_exact = test_affine_patch()
    print(f"[patch test] interior disp err = {err_u:.3e}, "
          f"rel err sigma = {err_sig:.3e}, eps = {err_eps:.3e} "
          f"(sigma_exact Voigt = "
          f"{[f'{s:.6e}' for s in sig_exact]})")
    assert err_u < 1e-9, "patch test interior displacement failed"
    assert err_sig < 1e-8 and err_eps < 1e-8, \
        "patch test stress/strain validation failed"

    errs, rates = test_convergence()
    print(f"[convergence] errors = {[f'{e:.3e}' for e in errs]}, "
          f"rates = {[f'{r:.2f}' for r in rates]} (expect ~2.0)")
    assert all(1.7 < r < 2.3 for r in rates), \
        f"convergence rate off: {rates}"

    err_T, err_n, err_s, s_exact, u_max, vm_max = test_coupled_chain()
    print(f"[coupled chain] temp err = {err_T:.3e}, "
          f"sigma_exact = {s_exact/1e6:.1f} MPa, "
          f"rel err normal = {err_n:.3e}, shear = {err_s:.3e}, "
          f"u_max = {u_max:.3e}, vm_max = {vm_max:.3e}")
    assert err_n < 1e-6 and err_s < 1e-6, \
        "coupled chain thermal stress validation failed"
    assert vm_max / abs(s_exact) < 1e-6, \
        "coupled chain von Mises must vanish (hydrostatic state)"

    print("ALL 3D ELASTIC VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
