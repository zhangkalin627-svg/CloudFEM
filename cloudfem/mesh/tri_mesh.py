"""Triangular mesh generation and mesh data structures.

MVP scope: structured rectangular domains split into triangles,
plus uniform refinement. Unstructured (Delaunay) meshing of
arbitrary polygons is on the roadmap.
"""

import numpy as np


class TriMesh:
    """Simple triangular mesh container.

    Attributes
    ----------
    nodes : (N, 2) float array, node coordinates (x, y).
    elements : (M, 3) int array, node indices of each triangle.
    boundaries : dict[str, int array], named node sets on the boundary,
        e.g. {'left': [...], 'right': [...], 'bottom': [...], 'top': [...]}.
    """

    def __init__(self, nodes, elements, boundaries=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.boundaries = dict(boundaries) if boundaries else {}
        self._areas = None

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    def element_areas(self):
        """Signed-doubled area helper -> absolute triangle areas, shape (M,)."""
        if self._areas is None:
            p = self.nodes[self.elements]  # (M, 3, 2)
            x = p[:, :, 0]
            y = p[:, :, 1]
            self._areas = 0.5 * np.abs(
                (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
            )
        return self._areas

    def refine(self):
        """Uniform red refinement: each triangle -> 4 congruent triangles."""
        nodes = self.nodes.tolist()
        elements = self.elements
        edge_mid = {}  # (i, j) sorted -> new node index

        def midpoint(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in edge_mid:
                edge_mid[key] = len(nodes)
                nodes.append(
                    ((self.nodes[i, 0] + self.nodes[j, 0]) / 2.0,
                     (self.nodes[i, 1] + self.nodes[j, 1]) / 2.0)
                )
            return edge_mid[key]

        new_elements = []
        for a, b, c in elements:
            mab = midpoint(a, b)
            mbc = midpoint(b, c)
            mca = midpoint(c, a)
            new_elements += [
                (a, mab, mca),
                (b, mbc, mab),
                (c, mca, mbc),
                (mab, mbc, mca),
            ]

        new_nodes = np.array(nodes)
        new_elements = np.array(new_elements, dtype=int)

        # Re-tag boundary nodes: old boundary nodes keep their tags,
        # edge midpoints inherit the tag of the edge's boundary.
        new_boundaries = {}
        old_index_of = {}  # not needed; new nodes beyond n_old are mids
        n_old = self.n_nodes
        # map new node -> (i, j) edge for midpoints
        mid_edge = {v: k for k, v in edge_mid.items()}
        for name, ids in self.boundaries.items():
            ids = np.asarray(ids)
            keep = [int(i) for i in ids]  # old nodes keep indices
            idset = set(int(i) for i in ids)
            for v, (i, j) in mid_edge.items():
                if i in idset and j in idset:
                    keep.append(v)
            new_boundaries[name] = np.array(sorted(set(keep)), dtype=int)

        return TriMesh(new_nodes, new_elements, new_boundaries)


def rect_mesh(nx, ny, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
    """Uniform triangular mesh of a rectangle.

    Each rectangular cell is split into two triangles.
    Boundary tags: 'left', 'right', 'bottom', 'top'.
    """
    x = np.linspace(xlim[0], xlim[1], nx + 1)
    y = np.linspace(ylim[0], ylim[1], ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    def idx(i, j):
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = idx(i, j)
            n1 = idx(i + 1, j)
            n2 = idx(i, j + 1)
            n3 = idx(i + 1, j + 1)
            elements.append((n0, n1, n3))
            elements.append((n0, n3, n2))
    elements = np.array(elements, dtype=int)

    left = np.array([idx(0, j) for j in range(ny + 1)], dtype=int)
    right = np.array([idx(nx, j) for j in range(ny + 1)], dtype=int)
    bottom = np.array([idx(i, 0) for i in range(nx + 1)], dtype=int)
    top = np.array([idx(i, ny) for i in range(nx + 1)], dtype=int)

    return TriMesh(
        nodes, elements,
        {"left": left, "right": right, "bottom": bottom, "top": top},
    )
