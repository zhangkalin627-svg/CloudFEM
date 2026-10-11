"""Shared implicit-Euler time stepper for transient heat conduction."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .poisson import assemble, _gradients


def mass_matrix(mesh, rho_cp=1.0):
    """Consistent P1 mass matrix scaled by rho_cp (float or callable)."""
    n = mesh.n_nodes
    rows, cols, data = [], [], []
    rc_fn = rho_cp if callable(rho_cp) else lambda x, y: rho_cp
    for el in mesh.elements:
        p = mesh.nodes[el]
        _, _, area = _gradients(p)
        xc, yc = p[:, 0].mean(), p[:, 1].mean()
        me = rc_fn(xc, yc) * area / 12.0 * (np.ones((3, 3)) + np.eye(3))
        for i in range(3):
            for j in range(3):
                rows.append(el[i]); cols.append(el[j]); data.append(me[i, j])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def dirichlet_nodes(mesh, bc):
    """Resolve bc dict -> {node_index: value}."""
    n = mesh.n_nodes
    fixed = {}
    if not bc:
        return fixed
    for name, value in bc.items():
        ids = np.arange(n) if name == "all" else mesh.boundaries[name]
        v_fn = value if callable(value) else (lambda x, y, v=value: v)
        for i in ids:
            x, y = mesh.nodes[int(i)]
            fixed[int(i)] = v_fn(x, y)
    return fixed


def apply_dirichlet_system(A_lil, rhs, fixed):
    """Apply Dirichlet BCs to a LIL system matrix and rhs vector."""
    for i, v in fixed.items():
        A_lil.rows[i] = [i]
        A_lil.data[i] = [1.0]
        rhs[i] = v
    A = A_lil.tocsr()
    if fixed:
        fidx = np.array(sorted(fixed), dtype=int)
        v = np.array([fixed[i] for i in fidx])
        rhs = rhs - A[:, fidx] @ v
        for i, vv in fixed.items():
            rhs[i] = vv
        A[:, fidx] = 0
        A = A.tocsr()
        for i in fixed:
            A[i, i] = 1.0
    return A, rhs


def implicit_euler_heat(mesh, k, rho_cp, rhs_q, bc, t_end, dt, T0):
    """Implicit Euler for rho*cp dT/dt - div(k grad T) = Q.

    rhs_q : precomputed consistent load vector of the source Q.
    Returns (times, T_history).
    """
    n = mesh.n_nodes
    K, _ = assemble(mesh, k=k)
    M = mass_matrix(mesh, rho_cp)
    T = np.full(n, T0, dtype=float) if np.isscalar(T0) else np.asarray(T0, float)
    A_sys = (M / dt + K).tolil()
    fixed = dirichlet_nodes(mesh, bc)

    times = [0.0]
    hist = [T.copy()]
    t = 0.0
    while t < t_end - 1e-12:
        t += dt
        rhs = M @ T / dt + rhs_q
        A, rhs = apply_dirichlet_system(A_sys.copy(), rhs, fixed)
        T = np.asarray(spla.spsolve(A, rhs)).ravel()
        times.append(t)
        hist.append(T.copy())
    return np.array(times), np.array(hist)


def source_load_vector(mesh, q_nodal):
    """Consistent load vector from a nodal source field."""
    n = mesh.n_nodes
    rhs = np.zeros(n)
    for el in mesh.elements:
        _, _, area = _gradients(mesh.nodes[el])
        qe = q_nodal[el].mean()
        for i in el:
            rhs[i] += qe * area / 3.0
    return rhs


def element_to_nodal(mesh, q_elem):
    """Average element values to nodes."""
    q_nodal = np.zeros(mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    np.add.at(q_nodal, mesh.elements.ravel(), np.repeat(q_elem, 3))
    np.add.at(count, mesh.elements.ravel(), 1)
    return q_nodal / np.maximum(count, 1)
