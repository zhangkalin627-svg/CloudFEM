"""Unstructured triangular mesh generation via Delaunay triangulation.

v0.3 scope: mesh a single polygonal domain (convex or mildly concave)
with approximately uniform target edge length h. Multi-material models
are handled by assigning material per element centroid with callable
material functions (see physics modules), so the mesher only needs to
capture the geometry.

Algorithm:
  1. resample the polygon boundary at spacing ~h,
  2. fill the interior with a jittered grid of spacing ~h,
  3. scipy.spatial.Delaunay triangulation,
  4. drop triangles whose centroid falls outside the polygon,
  5. a few Laplacian smoothing sweeps on interior nodes.
"""

import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path

from .tri_mesh import TriMesh


def _resample_boundary(vertices, h):
    """Walk the polygon, emitting points spaced ~h apart.

    Intermediate points on straight edges get a tiny (1e-7*h) inward
    perturbation to break exact collinearity, which would otherwise let
    Qhull emit zero-area sliver triangles between consecutive boundary
    points. The geometric perturbation is far below discretization error.
    """
    verts = np.asarray(vertices, dtype=float)
    path = Path(verts)
    centroid = verts.mean(axis=0)
    pts = []
    n = len(verts)
    rng = np.random.default_rng(7)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        seg_len = float(np.linalg.norm(b - a))
        nseg = max(1, int(round(seg_len / h)))
        for k in range(nseg):
            t = k / nseg
            p = a + t * (b - a)
            if 0 < k < nseg:
                # nudge off the straight edge
                d = centroid - p
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-12:
                    p = p + d / nrm * (1e-7 * h)
            pts.append(p)
    return np.array(pts)


def _interior_points(path, h, bbox):
    """Jittered grid inside the polygon."""
    xs = np.arange(bbox[0], bbox[1] + h * 0.5, h * 0.9)
    ys = np.arange(bbox[2], bbox[3] + h * 0.5, h * 0.9)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(42)
    pts = pts + rng.uniform(-0.2 * h, 0.2 * h, size=pts.shape)
    inside = path.contains_points(pts)
    return pts[inside]


def _deduplicate(points, tol):
    """Merge points closer than tol, keeping the first occurrence."""
    cell = tol
    grid = {}
    keep = []
    for i, p in enumerate(points):
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)))
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((key[0] + dx, key[1] + dy), []):
                    if np.linalg.norm(points[j] - p) < tol:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid.setdefault(key, []).append(i)
            keep.append(i)
    return points[np.array(keep, dtype=int)], np.array(keep, dtype=int)


def _inner_ring(verts, bpts, h, path):
    """Points at ~0.7h inward from each boundary point (boundary layer).

    Supports the boundary in Delaunay triangulation, preventing sliver
    triangles between consecutive boundary points. Inward ~ toward the
    polygon centroid (valid for star-shaped domains w.r.t. centroid).
    """
    centroid = verts.mean(axis=0)
    ring = []
    for p in bpts:
        d = centroid - p
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        q = p + d / n * (0.7 * h)
        if path.contains_point(q):
            ring.append(q)
    return np.array(ring) if ring else np.zeros((0, 2))
    """Merge points closer than tol, keeping the first occurrence."""
    # grid hashing for speed
    cell = tol
    grid = {}
    keep = []
    for i, p in enumerate(points):
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)))
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((key[0] + dx, key[1] + dy), []):
                    if np.linalg.norm(points[j] - p) < tol:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            grid.setdefault(key, []).append(i)
            keep.append(i)
    return points[np.array(keep, dtype=int)], np.array(keep, dtype=int)


def _tri_quality(p):
    """Quality of triangles p (M,3,2): 4*sqrt(3)*area/sum(l^2)."""
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    x, y = p[:, :, 0], p[:, :, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    return 4 * np.sqrt(3) * area / np.maximum(a ** 2 + b ** 2 + c ** 2, 1e-30)


def _smooth(nodes, elements, fixed, steps=5):
    """Quality-aware Laplacian smoothing of interior nodes.

    A node moves to its neighbours' centroid only if the worst quality
    of its adjacent triangles improves (prevents collapse/degeneracy).
    """
    n = len(nodes)
    nbrs = [set() for _ in range(n)]
    adj_tris = [[] for _ in range(n)]
    for ti, (a, b, c) in enumerate(elements):
        nbrs[a] |= {b, c}
        nbrs[b] |= {a, c}
        nbrs[c] |= {a, b}
        adj_tris[a].append(ti)
        adj_tris[b].append(ti)
        adj_tris[c].append(ti)
    pts = nodes.copy()
    for _ in range(steps):
        for i in range(n):
            if i in fixed or not nbrs[i]:
                continue
            cand = pts[list(nbrs[i])].mean(axis=0)
            tis = adj_tris[i]
            old_q = _tri_quality(pts[elements[tis]]).min()
            trial = pts.copy()
            trial[i] = cand
            new_q = _tri_quality(trial[elements[tis]]).min()
            if new_q > old_q:
                pts[i] = cand
    return pts


def polygon_mesh(vertices, h, smooth_steps=8):
    """Triangular mesh of a polygonal domain.

    Parameters
    ----------
    vertices : (N, 2) array, polygon vertices in order (CCW or CW).
    h : target edge length.
    smooth_steps : Laplacian smoothing iterations.

    Returns TriMesh with a single 'boundary' node tag.
    """
    verts = np.asarray(vertices, dtype=float)
    path = Path(verts)
    bbox = (verts[:, 0].min(), verts[:, 0].max(),
            verts[:, 1].min(), verts[:, 1].max())

    bpts = _resample_boundary(verts, h)
    ipts = _interior_points(path, h, bbox)
    ring = _inner_ring(verts, bpts, h, path)
    points = np.vstack([bpts, ring, ipts])
    n_bnd_orig = len(bpts)
    # deduplicate (a jittered interior point may coincide with a boundary
    # point); a surviving point is on the boundary iff its original index
    # was a boundary point.
    points, kept = _deduplicate(points, tol=0.15 * h)
    is_bnd = kept < n_bnd_orig

    tri = Delaunay(points)
    tris = tri.simplices

    # keep triangles with centroid inside the polygon
    cent = points[tris].mean(axis=1)
    keep = path.contains_points(cent)
    elements = tris[keep]

    # drop unused points and reindex
    used = np.unique(elements)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    nodes = points[used]
    elements = remap[elements]

    fixed = set(np.where(is_bnd[used])[0])  # boundary nodes stay put
    nodes = _smooth(nodes, elements, fixed, steps=smooth_steps)

    bnd_ids = np.array(sorted(fixed), dtype=int)
    return TriMesh(nodes, elements, {"boundary": bnd_ids})


def disk_mesh(radius, h, center=(0.0, 0.0), n_seg=None):
    """Triangular mesh of a disk.

    n_seg defaults to ceil(2*pi*radius/h) so the polygonal boundary
    approximates the circle with O(h^2) geometric error (required for
    FEM convergence) and boundary points are spaced ~h without
    exact collinearity.
    """
    if n_seg is None:
        n_seg = max(32, int(np.ceil(2 * np.pi * radius / h)))
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    verts = np.column_stack([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
    ])
    return polygon_mesh(verts, h)


def mesh_quality(mesh):
    """Min/mean triangle quality q = 4*sqrt(3)*area/sum(l^2) in (0, 1]."""
    p = mesh.nodes[mesh.elements]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    area = mesh.element_areas()
    q = 4 * np.sqrt(3) * area / (a ** 2 + b ** 2 + c ** 2)
    return float(q.min()), float(q.mean())
