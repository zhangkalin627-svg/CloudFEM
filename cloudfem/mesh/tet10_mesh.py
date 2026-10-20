"""Second-order (10-node) tetrahedral meshes.

v1.5 scope: Tet10 elements built by enriching a P1 tetrahedral mesh
with edge-midpoint nodes. Structured box domains come from the Kuhn
box subdivision of :mod:`cloudfem.mesh.tet_mesh`; a generic
Tet4 -> Tet10 converter handles any :class:`TetMesh` (including the
unstructured meshes of :mod:`cloudfem.mesh.unstructured3d`), although
only the box path is exercised by the v1.5 solver tests.

Local node ordering of one element::

    0..3   corner nodes, same order as the parent Tet4 element
    4..9   edge midpoints, for edges (see EDGES)
           (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)

Corner node indices of the parent mesh are preserved: node ``i`` of
the Tet4 mesh is node ``i`` of the Tet10 mesh, and midpoint nodes are
appended after them. This keeps boundary sets and nodal fields
directly comparable between the P1 and P2 discretisations.

Assumption: all edge midpoints sit exactly at the geometric edge
midpoints, so every element map is affine and the corner-based
Jacobian used by the Tet10 element routines is exact. (True for
meshes produced here; a Tet10 mesh with genuinely curved edges is
out of scope.)
"""

import numpy as np

from .tet_mesh import TetMesh, box_mesh

# Local edges of a tetrahedron, in midpoint-node order (slots 4..9).
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


class Tet10Mesh:
    """Ten-node tetrahedral mesh container.

    Attributes
    ----------
    nodes : (N, 3) float array, node coordinates (x, y, z). The first
        ``n_corners`` entries are the corner (vertex) nodes.
    elements : (M, 10) int array, node indices per element in the
        local ordering described in the module docstring.
    boundaries : dict[str, int array], named node sets (corners and
        midpoints) on the boundary.
    """

    def __init__(self, nodes, elements, boundaries=None, n_corners=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.boundaries = dict(boundaries) if boundaries else {}
        if n_corners is None:
            n_corners = len(np.unique(self.elements[:, :4]))
        self.n_corners = int(n_corners)
        self._volumes = None

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    def element_volumes(self):
        """Absolute tetrahedron volumes from the corner nodes, (M,)."""
        if self._volumes is None:
            p = self.nodes[self.elements[:, :4]]  # (M, 4, 3)
            m = np.stack([p[:, 1] - p[:, 0],
                          p[:, 2] - p[:, 0],
                          p[:, 3] - p[:, 0]], axis=2)  # (M, 3, 3)
            self._volumes = np.abs(np.linalg.det(m)) / 6.0
        return self._volumes


def tet4_to_tet10(mesh):
    """Enrich a P1 :class:`TetMesh` with edge-midpoint nodes.

    Corner indices are preserved; midpoint nodes are appended in the
    order of the sorted unique edge list. A midpoint joins a named
    boundary set iff both of its parent corners belong to that set
    (exact for the planar faces of box meshes; an approximation for
    general curved boundaries, where a boundary edge's midpoint is
    placed on the chord, not on the surface).
    """
    els = np.asarray(mesh.elements, dtype=int)
    n0 = mesh.n_nodes
    m = els.shape[0]

    # (M, 6, 2) corner-index pairs in EDGES order
    pairs = np.stack([els[:, list(e)] for e in EDGES], axis=1)
    flat = np.sort(pairs.reshape(-1, 2), axis=1)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    mid_nodes = 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]])
    mid_index = n0 + inv.reshape(m, 6)  # (M, 6) global midpoint ids

    elements = np.column_stack([els, mid_index])
    nodes = np.vstack([mesh.nodes, mid_nodes])

    boundaries = {}
    edge_parents = uniq  # (E, 2) corner ids per unique edge
    for name, ids in mesh.boundaries.items():
        idset = set(int(i) for i in ids)
        keep = np.array([a in idset and b in idset
                         for a, b in edge_parents], dtype=bool)
        extra = n0 + np.nonzero(keep)[0]
        boundaries[name] = np.sort(np.concatenate(
            [np.asarray(ids, dtype=int), extra])).astype(int)

    return Tet10Mesh(nodes, elements, boundaries, n_corners=n0)


def box_mesh_tet10(nx, ny, nz, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                   zlim=(0.0, 1.0)):
    """Uniform Tet10 mesh of a box (Kuhn 6-tet cells + edge midpoints).

    Boundary tags as in :func:`cloudfem.mesh.tet_mesh.box_mesh`;
    the sets include the midpoint nodes lying on each face plane
    (selected by coordinate, so they are exact).
    """
    mesh = tet4_to_tet10(box_mesh(nx, ny, nz, xlim=xlim, ylim=ylim,
                                  zlim=zlim))
    scale = max(1.0, float(np.ptp(mesh.nodes)))
    tol = 1e-9 * scale
    lims = {"xmin": (0, xlim[0]), "xmax": (0, xlim[1]),
            "ymin": (1, ylim[0]), "ymax": (1, ylim[1]),
            "zmin": (2, zlim[0]), "zmax": (2, zlim[1])}
    boundaries = {}
    for name, (axis, val) in lims.items():
        boundaries[name] = np.nonzero(
            np.abs(mesh.nodes[:, axis] - val) <= tol)[0].astype(int)
    mesh.boundaries = boundaries
    return mesh
