"""Tetrahedral mesh generation and mesh data structures.

v0.7 scope: structured box domains subdivided into tetrahedra
(Kuhn subdivision: 6 tetrahedra per cubic cell, all sharing the cell's
main diagonal, which keeps the subdivision conformal across cells).
"""

import numpy as np


class TetMesh:
    """Simple tetrahedral mesh container.

    Attributes
    ----------
    nodes : (N, 3) float array, node coordinates (x, y, z).
    elements : (M, 4) int array, node indices of each tetrahedron.
    boundaries : dict[str, int array], named node sets on the boundary,
        e.g. {'xmin': [...], 'xmax': [...], ...}.
    """

    def __init__(self, nodes, elements, boundaries=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.boundaries = dict(boundaries) if boundaries else {}
        self._volumes = None

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    def element_volumes(self):
        """Absolute tetrahedron volumes, shape (M,)."""
        if self._volumes is None:
            p = self.nodes[self.elements]  # (M, 4, 3)
            m = np.stack([p[:, 1] - p[:, 0],
                          p[:, 2] - p[:, 0],
                          p[:, 3] - p[:, 0]], axis=2)  # (M, 3, 3)
            self._volumes = np.abs(np.linalg.det(m)) / 6.0
        return self._volumes


def box_mesh(nx, ny, nz, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0)):
    """Uniform tetrahedral mesh of a box.

    Each cubic cell is split into 6 tetrahedra sharing the cell's
    main diagonal (0 -> 6), so neighbouring cells agree on faces.

    Boundary tags: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
    """
    x = np.linspace(xlim[0], xlim[1], nx + 1)
    y = np.linspace(ylim[0], ylim[1], ny + 1)
    z = np.linspace(zlim[0], zlim[1], nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def idx(i, j, k):
        # C-order flat index of nodes[i, j, k] from meshgrid(indexing="ij")
        return (i * (ny + 1) + j) * (nz + 1) + k

    # local cube corners, then 6 tets around diagonal (0, 6)
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [idx(i + di, j + dj, k + dk)
                     for dk in (0, 1) for dj in (0, 1) for di in (0, 1)]
                # c order: (0,0,0)=c0,(1,0,0)=c1,(0,1,0)=c2,(1,1,0)=c3,
                #          (0,0,1)=c4,(1,0,1)=c5,(0,1,1)=c6,(1,1,1)=c7
                n0, n1, n2, n3, n4, n5, n6, n7 = c
                elements += [
                    (n0, n1, n3, n7),
                    (n0, n1, n7, n5),
                    (n0, n5, n7, n4),
                    (n0, n3, n2, n7),
                    (n0, n2, n6, n7),
                    (n0, n6, n4, n7),
                ]
    elements = np.array(elements, dtype=int)

    xmin = np.array([idx(0, j, k) for k in range(nz + 1)
                     for j in range(ny + 1)], dtype=int)
    xmax = np.array([idx(nx, j, k) for k in range(nz + 1)
                     for j in range(ny + 1)], dtype=int)
    ymin = np.array([idx(i, 0, k) for k in range(nz + 1)
                     for i in range(nx + 1)], dtype=int)
    ymax = np.array([idx(i, ny, k) for k in range(nz + 1)
                     for i in range(nx + 1)], dtype=int)
    zmin = np.array([idx(i, j, 0) for j in range(ny + 1)
                     for i in range(nx + 1)], dtype=int)
    zmax = np.array([idx(i, j, nz) for j in range(ny + 1)
                     for i in range(nx + 1)], dtype=int)

    return TetMesh(
        nodes, elements,
        {"xmin": xmin, "xmax": xmax, "ymin": ymin,
         "ymax": ymax, "zmin": zmin, "zmax": zmax},
    )
