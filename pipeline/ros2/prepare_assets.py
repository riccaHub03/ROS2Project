"""Prepare portable gripper assets; keep the original source/checkpoint pins."""
import subprocess
from pathlib import Path

from pipeline.sim.robotiq_description import ROOT, REVISION, prepared_description


def main():
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
