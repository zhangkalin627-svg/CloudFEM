"""Minimal Web UI for CloudFEM steady-state heat conduction (v0.6).

Dependency-pure: stdlib http.server only — no Flask/Django, staying true
to CloudFEM's "numpy + scipy + matplotlib only" philosophy.

Workflow in the browser:
  1. click on the canvas to place polygon vertices (any closed 2D shape),
  2. "Mesh"   -> Delaunay triangular mesh of the polygon,
  3. "Solve"  -> steady heat, T=Thot on the left edge, T=Tcold on the right
                 edge, adiabatic elsewhere; temperature contours are drawn.

Model coordinates are the canvas pixels (x right, y up; the page flips
the y axis when drawing). Canvas: 600 x 450 model units.

Run:  python3 -m cloudfem.webui [--port 8123]
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from ..mesh.unstructured import polygon_mesh, mesh_quality
from ..physics.heat import steady_heat

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 1 << 20  # 1 MB request cap


# --------------------------------------------------------------------------
# geometry / solve helpers (also used by the test-suite directly)
# --------------------------------------------------------------------------

def mesh_polygon(vertices, h):
    """Build a triangular mesh of the polygon and tag left/right boundaries.

    vertices : sequence of (x, y), in order (CW or CCW).
    h        : target edge length.

    Boundary nodes on the leftmost / rightmost edges are tagged
    'left' / 'right' (within h/4 of the extreme x).  Near-zero-area
    Qhull slivers (q < 1e-6) on long straight edges are dropped first;
    the removal is guarded to < 1e-6 of the domain area.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.shape[0] < 3:
        raise ValueError("polygon needs at least 3 vertices")
    if not np.isfinite(verts).all():
        raise ValueError("vertices must be finite numbers")
    if float(h) <= 0:
        raise ValueError("h must be positive")
    # reject absurd inputs before burning Qhull
    bbox = np.ptp(verts, axis=0)
    if (bbox < 1e-9).any():
        raise ValueError("polygon is degenerate (zero extent)")
    if bbox.max() / float(h) > 500:
        raise ValueError("h too small for the domain size "
                         "(would create > ~2.5e5 elements)")

    mesh = polygon_mesh(verts, float(h))
    mesh = _drop_slivers(mesh, q_floor=1e-6)
    xmin, xmax = mesh.nodes[:, 0].min(), mesh.nodes[:, 0].max()
    tol = float(h) * 0.25
    bnd = mesh.boundaries.get("boundary", np.array([], dtype=int))
    left = bnd[mesh.nodes[bnd, 0] <= xmin + tol]
    right = bnd[mesh.nodes[bnd, 0] >= xmax - tol]
    mesh.boundaries["left"] = left
    mesh.boundaries["right"] = right
    return mesh


def _tri_qualities(nodes, elements):
    """Vectorised triangle quality q = 4*sqrt(3)*area/sum(l^2)."""
    p = nodes[elements]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    x, y = p[:, :, 0], p[:, :, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    return 4 * np.sqrt(3) * area / (a ** 2 + b ** 2 + c ** 2), area


def _drop_slivers(mesh, q_floor=1e-6):
    """Remove near-zero-area sliver triangles from the mesh.

    Qhull occasionally emits needle triangles (q ~ 1e-9) between the
    nudged boundary points on long straight edges.  Their stiffness
    entries scale as 1/area and poison the matrix conditioning, while
    their geometric area is ~1e-9 of the domain.  Dropping them is a
    net win; the removal is guarded so it can never eat a real part
    of the domain.
    """
    from ..mesh.tri_mesh import TriMesh

    q, area = _tri_qualities(mesh.nodes, mesh.elements)
    keep = q >= q_floor
    if keep.all():
        return mesh
    removed_frac = area[~keep].sum() / area.sum()
    if removed_frac > 1e-6:
        raise RuntimeError(
            f"sliver filter would remove {removed_frac:.2e} of the domain "
            f"({(~keep).sum()} triangles) - refusing")
    elements = mesh.elements[keep]
    used = np.unique(elements)
    remap = np.full(mesh.n_nodes, -1, dtype=int)
    remap[used] = np.arange(len(used))
    nodes = mesh.nodes[used]
    elements = remap[elements]
    new_mesh = TriMesh(nodes, elements)
    # rebuild boundary tags from the surviving single-sided edges
    bedges = new_mesh.boundary_edges()
    bnd = np.unique(bedges.ravel()) if len(bedges) else np.array([], dtype=int)
    new_mesh.boundaries["boundary"] = bnd
    return new_mesh


def solve_request(data):
    """Solve steady heat on the submitted polygon.

    data: dict with keys
        vertices : [[x, y], ...]      (required)
        h        : target edge length (default 20)
        t_hot    : Dirichlet value on left edge (default 100)
        t_cold   : Dirichlet value on right edge (default 0)
        k        : conductivity (default 1.0)

    Returns a JSON-serialisable dict with nodes, elements, nodal
    temperatures and a few stats.
    """
    vertices = data.get("vertices")
    if not vertices or len(vertices) < 3:
        raise ValueError("need at least 3 vertices")
    h = float(data.get("h", 20.0))
    t_hot = float(data.get("t_hot", 100.0))
    t_cold = float(data.get("t_cold", 0.0))
    k = float(data.get("k", 1.0))
    if k <= 0:
        raise ValueError("k must be positive")

    mesh = mesh_polygon(vertices, h)
    T = steady_heat(mesh, k=k,
                    bc={"left": t_hot, "right": t_cold})

    return {
        "nodes": mesh.nodes.tolist(),
        "elements": mesh.elements.tolist(),
        "temps": T.tolist(),
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "t_min": float(T.min()),
        "t_max": float(T.max()),
        "min_quality": float(mesh_quality(mesh)[0]),
    }


def mesh_quality_report(data):
    """Mesh-only variant used by the UI preview step."""
    vertices = data.get("vertices")
    if not vertices or len(vertices) < 3:
        raise ValueError("need at least 3 vertices")
    h = float(data.get("h", 20.0))
    mesh = mesh_polygon(vertices, h)
    qmin, qmean = mesh_quality(mesh)
    return {
        "nodes": mesh.nodes.tolist(),
        "elements": mesh.elements.tolist(),
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "min_quality": float(qmin),
        "mean_quality": float(qmean),
    }


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "CloudFEM-WebUI/0.6"

    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            raise ValueError("bad request body length")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), ctype="text/html")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/mesh":
                result = mesh_quality_report(data)
            elif path == "/api/solve":
                result = solve_request(data)
            else:
                self._send(404, json.dumps({"error": "unknown endpoint"}))
                return
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}))
            return
        except Exception as e:  # noqa: BLE001 - report solver failures cleanly
            self._send(500, json.dumps({"error": f"solve failed: {e}"}))
            return
        self._send(200, json.dumps(result))

    def log_message(self, fmt, *args):  # quieter than the stdlib default
        pass


def run(port=8123):
    """Start the Web UI server (blocking)."""
    server = HTTPServer(("127.0.0.1", port), _Handler)
    print(f"CloudFEM v0.6 Web UI: http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import sys

    _port = 8123
    if "--port" in sys.argv:
        _port = int(sys.argv[sys.argv.index("--port") + 1])
    run(_port)
