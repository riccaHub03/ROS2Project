"""Prepare the shared iiwa 7 and gripper assets at their pinned revisions."""
import subprocess
from pathlib import Path

from pipeline.sim.robotiq_description import ROOT, REVISION, prepared_description


def main():
    from pipeline.robot_assets import prepare_iiwa7
    prepare_iiwa7()
    repo = ROOT / "data/external/robotiq-ros"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/robotiq/robotiq-ros.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Existing Robotiq checkout is {revision}, expected {REVISION}; source left untouched")
    subprocess.run(["python3", str(ROOT / "scripts/prepare_robotiq_description.py")], check=True)
    metadata, urdf = prepared_description()
    print(f"Portable Robotiq ready: {urdf}; revision {metadata['revision']}")
    print("Keep your SAM3 weights and surgical STL files in data/external and data/inputs as documented.")


if __name__ == "__main__":
    main()
