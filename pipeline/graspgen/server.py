"""GPU-only launcher: validate assets, then run the unchanged official ZMQ server.

Adds provenance to its metadata, without copying or replacing model algorithms.
"""
import argparse
import sys

from pipeline.graspgen.config import ROOT
from pipeline.graspgen.prepare import check_models, check_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper-config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--code", default=str(ROOT / "third_party/GraspGen"))
    args = parser.parse_args(argv)
    revision = check_source(args.code)
    provenance = check_models(args.gripper_config, hashes=True)
    provenance.update(revision=revision, planner="diffusion")
    sys.path.insert(0, args.code)
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run this server in the GraspGen GPU environment")
    from grasp_gen.serving.zmq_server import GraspGenZMQServer
    server = GraspGenZMQServer(args.gripper_config, host=args.host, port=args.port)
    server._metadata["ros2project"] = provenance
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
