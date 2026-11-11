"""Load the official CPU-only client by file path and bound its connection wait."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def official_client_class(submodule):
    source = Path(submodule) / "grasp_gen/serving/zmq_client.py"
    if not source.is_file():
        raise RuntimeError("GraspGen submodule missing: bash scripts/graspgen_demos.sh init")
    spec = importlib.util.spec_from_file_location("ros2project_official_graspgen_client", source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError("Install requirements-graspgen-client.txt in the project environment") from exc
    return module.GraspGenClient


def infer(points_centered, cfg, *, metadata_callback=None):
    """Use public get_metadata/infer; disabling upstream retry makes both calls bounded.

    No reconnection/retry after timeout: a new invocation gets a fresh REQ socket.
    metadata_callback validates the checkpoint BEFORE sending the point cloud.
    """
    cls = official_client_class(cfg["submodule"])
    server = cfg["server"]
    try:
        with cls(host=server["host"], port=server["port"], timeout_ms=server["timeout_ms"],
                 wait_for_server=False) as client:
            metadata = client.get_metadata()
            if metadata_callback:
                metadata_callback(metadata)
            poses, scores = client.infer(points_centered, **cfg["inference"])
            return poses, scores, metadata
    except Exception as exc:
        # Import lazily so PCA does not require the client dependencies.
        import zmq
        if isinstance(exc, zmq.error.Again):
            raise TimeoutError(f"GraspGen timeout at {server['host']}:{server['port']} "
                               f"({server['timeout_ms']} ms per request); check GPU server / tunnel") from exc
        raise
