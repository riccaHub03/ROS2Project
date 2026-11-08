"""Copy a saved deprojection cloud/report to a portable, self-contained directory."""
import argparse
import shutil
from pathlib import Path

from pipeline.common.report import StageReport
from pipeline.graspgen.config import DEFAULT_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deprojection", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    dep = StageReport.load(args.deprojection)
    if not dep.stage.endswith("_deprojection"):
        parser.error("Expected a deprojection report")
    cloud = Path(dep.artifacts["pointcloud_ply"])
    if not cloud.is_absolute():
        cloud = args.deprojection.resolve().parent / cloud
    args.destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cloud, args.destination / "cloud.ply")
    shutil.copy2(DEFAULT_CONFIG, args.destination / "graspgen_reference.yaml")
    shutil.copy2(args.deprojection, args.destination / "source_report.json")
    dep.artifacts = {"pointcloud_ply": "cloud.ply"}
    dep.inputs = {"original_deprojection": str(args.deprojection.resolve())}
    dep.save(args.destination / "deprojection.json")
    print(args.destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
