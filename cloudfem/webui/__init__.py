"""v0.6 Web UI: draw a polygon in the browser, mesh it, solve steady heat.

Entry point: ``python3 -m cloudfem.webui`` (default port 8123).
"""

from .server import run, mesh_polygon, solve_request, mesh_quality_report

__all__ = ["run", "mesh_polygon", "solve_request", "mesh_quality_report"]
