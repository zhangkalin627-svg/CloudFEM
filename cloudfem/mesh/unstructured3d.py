"""Unstructured tetrahedral mesh generation via Delaunay triangulation.

v1.1 scope: mesh a sphere, a (solid or hollow) cylinder and an
L-shaped block (two-box union, non-convex) with approximately uniform
target edge length h, breaking the v0.7 box-only restriction. The
output is a plain TetMesh (nodes + tetrahedra + named boundary node
sets), so every existing 3D solver (steady/transient heat, elasticity,
modal) consumes it unchanged.

Algorithm (the 3D analogue of mesh/unstructured.py, v0.3):
  1. sample the true surface at spacing ~h (Fibonacci lattice on the
     sphere, rings on the cylinder, face grids on the L block),
  2. fill the interior with a jittered Cartesian grid of spacing ~h,
  3. scipy.spatial.Delaunay triangulation,
  4. drop tetrahedra whose centroid falls outside the domain (this is
     what carves the notch out of the L block),
  5. quality-aware Laplacian smoothing of interior nodes (boundary
     nodes fixed), then retriangulate,
  6. sliver cleanup: while a tetrahedron with radius-ratio quality
     below q_sliver remains and removals are left, drop the
     least-connected interior vertex of the worst sliver and
     retriangulate. Boundary vertices are never removed; a removal is
     reverted if the sliver count does not strictly decrease or the
     total volume drifts by more than 0.5 %.

Quality metric: radius ratio q = 3 * r_in / R_circ in (0, 1], with
r_in = 3V / S (V volume, S total face area) and R_circ the
circumradius; q = 1 for a regular tetrahedron. This is the standard
3D analogue of the 2D metric q = 4*sqrt(3)*A/sum(l^2) used by
mesh_quality in unstructured.py, but the two scales are not directly
comparable: radius-ratio values for a mesh of a given quality run
systematically lower than the 2D edge-based values, so the 2D floor
(qmin > 0.2) does not carry over. The cleanup threshold here is
q_sliver = 0.02 and the v1.1 validations assert qmin > 0.025
(measured qmin >= 0.031 on the meshes they use).

Known limitations:
  * Curved surfaces are approximated by planar facets, so the mesh
    volume of a sphere/cylinder underestimates the true volume with an
    O(h^2) faceting error (measured in tests/test_unstructured3d.py).
    For the cylinder the error is dominated by the circumferential
    polygon factor (nth/2pi)*sin(2pi/nth), nth = ceil(2*pi*R/h).
  * The centroid test assumes the domain is exactly the set where
    `contains` holds; for the L block it is exact, but a tetrahedron
    straddling the re-entrant edge can be kept or dropped based on
    its centroid alone, leaving an O(h) stair-step error at the notch
    (never a leak: all vertices are in the closed domain).
  * No size grading: h is uniform, so features much smaller than h
    (e.g. a very thin hollow-cylinder wall) are under-resolved.
"""

import numpy as np
from scipy.spatial import Delaunay

from .tet_mesh import TetMesh

# radius-ratio quality below which a tetrahedron counts as a sliver
Q_SLIVER = 0.02


def _deduplicate3(points, tol):
    """Merge points closer than tol, keeping the first occurrence."""
    cell = tol
    grid = {}
    keep = []
    for i, p in enumerate(points):
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)),
               int(np.floor(p[2] / cell)))
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy,
                                       key[2] + dz), []):
                        if np.linalg.norm(points[j] - p) < tol:
                            dup = True
                            break
                    if dup:
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid.setdefault(key, []).append(i)
            keep.append(i)
    return points[np.array(keep, dtype=int)], np.array(keep, dtype=int)


def _interior_points3(contains, lo, hi, h):
    """Jittered grid (spacing 0.9h, jitter +-0.2h) inside the domain.

    `contains` is the *shrunk* interior test supplied by each geometry
    (points kept stay ~0.35h away from the surface: much closer, and
    the Delaunay triangulation fills the gap with paper-thin
    tetrahedra; at ~0.35h the near-wall elements keep healthy aspect
    ratios and the surface samples still set the boundary resolution).
    """
    xs = [np.arange(lo[d], hi[d] + 0.45 * h, 0.9 * h) for d in range(3)]
    xx, yy, zz = np.meshgrid(*xs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    rng = np.random.default_rng(42)
    pts = pts + rng.uniform(-0.2 * h, 0.2 * h, size=pts.shape)
    return pts[contains(pts)]


def tet_quality(nodes, elements):
    """Radius-ratio quality q = 3*r_in/R_circ of each tetrahedron.

    nodes : (N, 3), elements : (M, 4). Returns (M,) array in (0, 1].
    Degenerate tetrahedra (zero volume or no finite circumcentre)
    get q = 0.
    """
    p = nodes[np.asarray(elements)]

    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    S = (tri_area(p[:, 0], p[:, 1], p[:, 2])
         + tri_area(p[:, 0], p[:, 1], p[:, 3])
         + tri_area(p[:, 0], p[:, 2], p[:, 3])
         + tri_area(p[:, 1], p[:, 2], p[:, 3]))
    m = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0],
                  p[:, 3] - p[:, 0]], axis=2)
    V = np.abs(np.linalg.det(m)) / 6.0
    r_in = 3.0 * V / np.maximum(S, 1e-300)

    # circumcentre: solve 2 (p_i - p_0) . c = |p_i|^2 - |p_0|^2, i=1..3
    A = 2.0 * np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0],
                        p[:, 3] - p[:, 0]], axis=1)
    b = (np.sum(p[:, 1:] ** 2, axis=2)
         - np.sum(p[:, :1] ** 2, axis=2))
    with np.errstate(all="ignore"):
        try:
            c = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # a single exactly-degenerate (coplanar) tetrahedron makes
            # the batched solve raise; fall back to the pseudoinverse,
            # which agrees with solve on every nonsingular system.
            c = (np.linalg.pinv(A) @ b[..., None])[..., 0]
    R = np.linalg.norm(c - p[:, 0], axis=1)
    q = 3.0 * r_in / np.maximum(R, 1e-300)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(q, 0.0, 1.0)


def mesh_quality3d(mesh):
    """(min, median, mean) radius-ratio quality of a TetMesh."""
    q = tet_quality(mesh.nodes, mesh.elements)
    return float(q.min()), float(np.median(q)), float(q.mean())


def _volumes(nodes, elements):
    p = nodes[elements]
    m = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0],
                  p[:, 3] - p[:, 0]], axis=2)
    return np.abs(np.linalg.det(m)) / 6.0


def _triangulate(points, is_bnd, contains, vtol=0.0):
    """Delaunay + centroid filter + reindex.

    Tetrahedra with volume <= vtol are also dropped: exactly
    coplanar point quadruples (common where sampled surface grids are
    planar) can otherwise enter the triangulation as zero-volume
    elements, which no solver can assemble.
    Returns (nodes, elements, is_bnd) restricted to used points.
    """
    tri = Delaunay(points)
    tets = tri.simplices
    cent = points[tets].mean(axis=1)
    tets = tets[contains(cent)]
    if vtol > 0.0:
        tets = tets[_volumes(points, tets) > vtol]
    used = np.unique(tets)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return points[used], remap[tets], is_bnd[used]


def _smooth3(nodes, elements, fixed, contains, steps=8):
    """Quality-aware Laplacian smoothing of interior nodes (3D).

    A node moves to its neighbours' centroid only if the worst
    radius-ratio quality of its adjacent tetrahedra improves, exactly
    as in the 2D _smooth. Nodes whose worst adjacent quality already
    exceeds 0.35 are skipped, and a candidate position outside the
    domain (possible near the L block's re-entrant edge) is rejected.
    Boundary (fixed) nodes never move.
    """
    n = len(nodes)
    nbrs = [set() for _ in range(n)]
    adj = [[] for _ in range(n)]
    for ti, el in enumerate(elements):
        for a in el:
            nbrs[a] |= {int(x) for x in el if x != a}
            adj[a].append(ti)
    pts = nodes.copy()
    for _ in range(steps):
        q_all = tet_quality(pts, elements)
        moved = 0
        for i in range(n):
            if i in fixed or not nbrs[i]:
                continue
            tis = np.array(adj[i])
            old_q = q_all[tis].min()
            if old_q > 0.35:
                continue
            cand = pts[sorted(nbrs[i])].mean(axis=0)
            if contains is not None and not contains(cand[None, :])[0]:
                continue
            saved = pts[i].copy()
            pts[i] = cand
            if tet_quality(pts, elements[tis]).min() > old_q:
                moved += 1
            else:
                pts[i] = saved
        if moved == 0:
            break
    return pts


def _build_tet_mesh(points, is_bnd, contains, contains_strict, h,
                    smooth_steps=8, max_remove=30):
    """Shared pipeline: triangulate, smooth, retriangulate, desliver.

    contains : closed-domain test (centroid filter, tag predicates).
    contains_strict : same domain shrunk by ~0.05h (smoothing guard).
    Returns (nodes, elements, is_bnd).
    """
    vtol = 1e-12 * h ** 3
    nodes, elements, bnd = _triangulate(points, is_bnd, contains,
                                        vtol=vtol)
    fixed = set(int(i) for i in np.where(bnd)[0])
    if smooth_steps > 0 and len(fixed) < len(nodes):
        nodes = _smooth3(nodes, elements, fixed, contains_strict,
                         steps=smooth_steps)
        nodes, elements, bnd = _triangulate(nodes, bnd, contains,
                                            vtol=vtol)

    # sliver cleanup by vertex removal + retriangulation
    q = tet_quality(nodes, elements)
    n_bad = int((q < Q_SLIVER).sum())
    vol0 = _volumes(nodes, elements).sum()
    removals = 0
    while n_bad > 0 and removals < max_remove:
        worst = elements[int(np.argmin(q))]
        fixed_now = set(int(i) for i in np.where(bnd)[0])
        cand = [int(v) for v in worst if int(v) not in fixed_now]
        if not cand:
            break
        counts = {v: int((elements == v).sum()) for v in cand}
        drop = min(cand, key=lambda v: counts[v])
        keep = np.ones(len(nodes), dtype=bool)
        keep[drop] = False
        new_nodes, new_elements, new_bnd = _triangulate(
            nodes[keep], bnd[keep], contains, vtol=vtol)
        new_q = tet_quality(new_nodes, new_elements)
        new_bad = int((new_q < Q_SLIVER).sum())
        new_vol = _volumes(new_nodes, new_elements).sum()
        if new_bad >= n_bad or abs(new_vol - vol0) > 0.005 * vol0:
            break  # no strict improvement (or volume drift): revert
        nodes, elements, bnd, q, n_bad = (new_nodes, new_elements,
                                          new_bnd, new_q, new_bad)
        removals += 1
    return nodes, elements, bnd


def _tags(nodes, predicates, tol):
    """Named boundary node sets from geometric predicates."""
    out = {}
    for name, fn in predicates.items():
        out[name] = np.where(fn(nodes))[0].astype(int)
    return out


def sphere_mesh(radius, h, center=(0.0, 0.0, 0.0), smooth_steps=8):
    """Unstructured tetrahedral mesh of a solid sphere.

    Surface points: Fibonacci lattice with mean spacing ~h (near-
    uniform, no pole clustering). Boundary tag: 'boundary'.
    """
    R = float(radius)
    c = np.asarray(center, dtype=float)

    n_surf = max(64, int(np.ceil(4.0 * np.pi * R ** 2 / (0.866 * h ** 2))))
    i = np.arange(n_surf)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_surf)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    surf = R * np.column_stack([np.sin(phi) * np.cos(theta),
                                np.sin(phi) * np.sin(theta),
                                np.cos(phi)]) + c

    contains = lambda P: np.sum((P - c) ** 2, axis=1) <= R ** 2
    shrunk = lambda P: np.sum((P - c) ** 2, axis=1) <= (R - 0.35 * h) ** 2
    ipts = _interior_points3(shrunk, c - R, c + R, h)
    points = np.vstack([surf, ipts])
    points, kept = _deduplicate3(points, tol=0.15 * h)
    is_bnd = kept < len(surf)

    nodes, elements, bnd = _build_tet_mesh(points, is_bnd, contains,
                                            shrunk, h, smooth_steps)
    tags = {"boundary": np.where(bnd)[0].astype(int)}
    return TetMesh(nodes, elements, tags)


def _cap_points(r_in, r_out, z, h):
    """Concentric-ring points on a flat annulus/disk at height z."""
    pts = []
    if r_in <= 0.0:
        pts.append([0.0, 0.0, z])
        r_start = h
    else:
        r_start = r_in
    r = r_start
    while r < r_out - 1e-9:
        m = max(6, int(np.ceil(2.0 * np.pi * r / h)))
        th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        pts.append(np.column_stack([r * np.cos(th), r * np.sin(th),
                                    np.full(m, z)]))
        r += h
    return np.vstack([np.atleast_2d(p) for p in pts]) if pts else \
        np.zeros((0, 3))


def _lateral_points(r, z0, z1, h):
    """Ring-stacked points on a cylindrical lateral surface.

    All rings share one angular phase. Alternating the phase by half
    a step between rings was tried first (it is what the 2D mesher's
    jitter suggests), but it makes the points of three consecutive
    rings form near-coplanar "diamond" quadruples whose tetrahedra
    collapse (~30x too little volume, q ~ 0.006); aligned rings give
    Kuhn-like prism cells and qmin ~ 0.03 (measured, v1.1 tests).
    """
    nz = max(2, int(round((z1 - z0) / h)))
    nth = max(10, int(np.ceil(2.0 * np.pi * r / h)))
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    rings = []
    for k in range(nz + 1):
        rings.append(np.column_stack([r * np.cos(th), r * np.sin(th),
                                      np.full(nth, z0 + k * (z1 - z0) / nz)]))
    return np.vstack(rings)


def cylinder_mesh(radius, height, h, r_inner=0.0, smooth_steps=8):
    """Unstructured tetrahedral mesh of a cylinder (axis = z).

    Solid if r_inner = 0, hollow (annular cross-section) otherwise.
    z ranges over [0, height]. Boundary tags: 'lateral' (outer wall),
    'inner' (inner wall, hollow only), 'zmin', 'zmax' (end caps) and
    'boundary' (all surface nodes).
    """
    R, H, ri = float(radius), float(height), float(r_inner)

    surf = [_lateral_points(R, 0.0, H, h),
            _cap_points(ri, R, 0.0, h),
            _cap_points(ri, R, H, h)]
    if ri > 0.0:
        surf.append(_lateral_points(ri, 0.0, H, h))
    surf = np.vstack(surf)

    def contains(P):
        r2 = P[:, 0] ** 2 + P[:, 1] ** 2
        return (r2 <= R ** 2) & (r2 >= ri ** 2) \
            & (P[:, 2] >= 0.0) & (P[:, 2] <= H)

    def shrunk(P):
        r2 = P[:, 0] ** 2 + P[:, 1] ** 2
        return (r2 <= (R - 0.35 * h) ** 2) \
            & (r2 >= (ri + 0.35 * h) ** 2 if ri > 0 else r2 >= 0.0) \
            & (P[:, 2] >= 0.35 * h) & (P[:, 2] <= H - 0.35 * h)

    ipts = _interior_points3(shrunk, (-R, -R, 0.0), (R, R, H), h)
    points = np.vstack([surf, ipts])
    points, kept = _deduplicate3(points, tol=0.15 * h)
    is_bnd = kept < len(surf)

    nodes, elements, bnd = _build_tet_mesh(points, is_bnd, contains,
                                            shrunk, h, smooth_steps)
    tol = 0.02 * h
    preds = {
        "lateral": lambda P: np.abs(np.linalg.norm(P[:, :2], axis=1)
                                    - R) < tol,
        "zmin": lambda P: np.abs(P[:, 2]) < tol,
        "zmax": lambda P: np.abs(P[:, 2] - H) < tol,
    }
    if ri > 0.0:
        preds["inner"] = lambda P: np.abs(
            np.linalg.norm(P[:, :2], axis=1) - ri) < tol
    tags = _tags(nodes, preds, tol)
    tags["boundary"] = np.where(bnd)[0].astype(int)
    return TetMesh(nodes, elements, tags)


def l_block_mesh(lx=3.0, ly=3.0, notch=(1.0, 1.0), height=1.0, h=0.35,
                 smooth_steps=8):
    """Unstructured tetrahedral mesh of an L-shaped block.

    The domain is [0, lx] x [0, ly] x [0, height] minus the notch
    rectangle [notch[0], lx] x [notch[1], ly] (extruded in z); i.e.
    the union of box A = [0, lx] x [0, notch[1]] and box
    B = [0, notch[0]] x [0, ly]. Boundary tags: 'xmin', 'xmax',
    'ymin', 'ymax', 'zmin', 'zmax' (nodes on those planes; the xmax
    set covers only the strip y <= notch[1], etc.) and 'boundary'.
    """
    nx_, ny_ = float(notch[0]), float(notch[1])
    boxes = [  # (x0, x1, y0, y1)
        (0.0, lx, 0.0, ny_),
        (0.0, nx_, 0.0, ly),
    ]
    H = float(height)

    def in_box(P, b, strict):
        x0, x1, y0, y1 = b
        if strict:
            return (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) \
                & (P[:, 1] < y1) & (P[:, 2] > 0.0) & (P[:, 2] < H)
        return (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) \
            & (P[:, 1] <= y1) & (P[:, 2] >= 0.0) & (P[:, 2] <= H)

    contains = lambda P: in_box(P, boxes[0], False) | in_box(P, boxes[1],
                                                             False)

    def shrunk(P):
        m = 0.35 * h
        out = np.zeros(len(P), dtype=bool)
        for (x0, x1, y0, y1) in boxes:
            out |= (P[:, 0] >= x0 + m) & (P[:, 0] <= x1 - m) \
                & (P[:, 1] >= y0 + m) & (P[:, 1] <= y1 - m) \
                & (P[:, 2] >= m) & (P[:, 2] <= H - m)
        return out

    # sample the 6 faces of both boxes on ~h grids; a sample survives
    # only if it is not strictly inside the *other* box (interface
    # samples are interior to the union and must be dropped).
    face_pts = []
    for bi, (x0, x1, y0, y1) in enumerate(boxes):
        other = boxes[1 - bi]
        grids = []
        xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / h)) + 1))
        ys = np.linspace(y0, y1, max(2, int(round((y1 - y0) / h)) + 1))
        zs = np.linspace(0.0, H, max(2, int(round(H / h)) + 1))
        for z in (0.0, H):
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            grids.append(np.column_stack([xx.ravel(), yy.ravel(),
                                          np.full(xx.size, z)]))
        for x in (x0, x1):
            yy, zz = np.meshgrid(ys, zs, indexing="ij")
            grids.append(np.column_stack([np.full(yy.size, x),
                                          yy.ravel(), zz.ravel()]))
        for y in (y0, y1):
            xx, zz = np.meshgrid(xs, zs, indexing="ij")
            grids.append(np.column_stack([xx.ravel(), np.full(xx.size, y),
                                          zz.ravel()]))
        cand = np.vstack(grids)
        keep = ~in_box(cand, other, strict=True)
        face_pts.append(cand[keep])
    surf = np.vstack(face_pts)

    ipts = _interior_points3(shrunk, (0.0, 0.0, 0.0), (lx, ly, H), h)
    points = np.vstack([surf, ipts])
    points, kept = _deduplicate3(points, tol=0.15 * h)
    is_bnd = kept < len(surf)

    nodes, elements, bnd = _build_tet_mesh(points, is_bnd, contains,
                                            shrunk, h, smooth_steps)
    tol = 0.02 * h
    preds = {
        "xmin": lambda P: np.abs(P[:, 0]) < tol,
        "xmax": lambda P: np.abs(P[:, 0] - lx) < tol,
        "ymin": lambda P: np.abs(P[:, 1]) < tol,
        "ymax": lambda P: np.abs(P[:, 1] - ly) < tol,
        "zmin": lambda P: np.abs(P[:, 2]) < tol,
        "zmax": lambda P: np.abs(P[:, 2] - H) < tol,
    }
    tags = _tags(nodes, preds, tol)
    # 'boundary' comes from the tracked surface-sample flags: the
    # re-entrant faces (x = notch[0] for y > notch[1], y = notch[1]
    # for x > notch[0]) lie on no global plane, so plane predicates
    # alone would miss them.
    tags["boundary"] = np.where(bnd)[0].astype(int)
    return TetMesh(nodes, elements, tags)
