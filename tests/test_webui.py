"""Tests for CloudFEM v0.6 Web UI.

Checks, not just "it ran":
  1. solve on a rectangle vs the analytic linear temperature profile,
  2. mesh quality floor on an L-shaped (concave) polygon,
  3. discrete maximum principle on the L-shape (0 <= T <= 100),
  4. HTTP round-trip: GET / serves the page, POST /api/mesh & /api/solve
     return sane shapes, bad input -> 400,
  5. degenerate inputs are rejected with ValueError.
"""

import json
import threading
import urllib.request
import urllib.error

import numpy as np

from cloudfem.webui import mesh_polygon, solve_request, mesh_quality_report
from cloudfem.webui.server import _Handler
from cloudfem.mesh.unstructured import mesh_quality
from http.server import HTTPServer


RECT = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
LSHAPE = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0],
          [50.0, 50.0], [50.0, 100.0], [0.0, 100.0]]


def test_solve_rectangle_vs_analytic():
    """Steady heat on a rectangle must reproduce T(x) = 100*(1 - x/100).

    Left edge (x=0) is T_hot=100, right edge (x=100) is T_cold=0,
    top/bottom insulated -> 1-D linear profile.
    """
    res = solve_request({"vertices": RECT, "h": 8.0,
                         "t_hot": 100.0, "t_cold": 0.0})
    nodes = np.array(res["nodes"])
    T = np.array(res["temps"])
    exact = 100.0 * (1.0 - nodes[:, 0] / 100.0)
    num = float(np.sqrt(np.mean((T - exact) ** 2)))
    den = float(np.sqrt(np.mean(exact ** 2)))
    assert num / den < 0.02, f"relative L2 error {num / den:.4f} >= 2%"


def test_mesh_quality_l_shape():
    """Concave polygon mesh must stay above the quality floor."""
    mesh = mesh_polygon(LSHAPE, 8.0)
    qmin, qmean = mesh_quality(mesh)
    assert qmin > 0.1, f"min quality {qmin:.3f} too low"
    assert qmean > 0.5, f"mean quality {qmean:.3f} too low"
    assert len(mesh.boundaries["left"]) > 0
    assert len(mesh.boundaries["right"]) > 0


def test_discrete_maximum_principle():
    """With Dirichlet bounds [0, 100] and no source, T must stay inside."""
    res = solve_request({"vertices": LSHAPE, "h": 10.0,
                         "t_hot": 100.0, "t_cold": 0.0})
    T = np.array(res["temps"])
    assert T.min() >= -1e-6, f"T below 0: {T.min()}"
    assert T.max() <= 100.0 + 1e-6, f"T above 100: {T.max()}"
    assert res["n_elements"] > 0


def _free_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port


def _post(port, path, obj):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def test_http_roundtrip():
    srv, port = _free_server()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=10) as r:
            html = r.read().decode("utf-8")
        assert r.status == 200
        assert "CloudFEM v0.6" in html and "canvas" in html

        code, m = _post(port, "/api/mesh",
                        {"vertices": RECT, "h": 10.0})
        assert code == 200
        assert m["n_elements"] > 0 and m["min_quality"] > 0.1

        code, s = _post(port, "/api/solve",
                        {"vertices": RECT, "h": 10.0,
                         "t_hot": 100.0, "t_cold": 0.0})
        assert code == 200
        T = np.array(s["temps"])
        assert len(T) == s["n_nodes"]
        assert abs(s["t_min"] - 0.0) < 1e-9
        assert abs(s["t_max"] - 100.0) < 1e-9

        # bad input -> 400, not a traceback
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/solve",
            data=json.dumps({"vertices": [[0, 0]]}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()


def test_reject_degenerate():
    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    assert raises(lambda: solve_request({"vertices": [[0, 0], [1, 1]], "h": 10.0}))
    assert raises(lambda: solve_request({"vertices": RECT, "h": 0.0}))
    assert raises(lambda: mesh_quality_report({"vertices": [], "h": 10.0}))
    assert raises(lambda: solve_request({"vertices": RECT, "h": 10.0, "k": -1.0}))


if __name__ == "__main__":
    test_solve_rectangle_vs_analytic()
    print("[webui] rectangle vs analytic linear profile: PASSED")
    test_mesh_quality_l_shape()
    print("[webui] L-shape mesh quality floor: PASSED")
    test_discrete_maximum_principle()
    print("[webui] discrete maximum principle: PASSED")
    test_http_roundtrip()
    print("[webui] HTTP round-trip (page + mesh + solve + 400 handling): PASSED")
    test_reject_degenerate()
    print("[webui] degenerate input rejection: PASSED")
    print("WEB UI VALIDATION PASSED")
