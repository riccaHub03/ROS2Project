"""Prepare the pinned official iiwa 7 model without a ROS installation.

Changes to generated artifacts: omit ROS/Gazebo tags, use official collision
STLs for visuals, add zero inertia to the virtual flange, make paths relative.
Upstream kinematics, physical link inertias and joint limits are retained.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import yaml

from pipeline.robot_model import ROOT, RobotModel, file_sha256, robot_settings


def prepare_iiwa7():
    import xacro
    settings = robot_settings({"robot": {"model": "iiwa7_r800"}})
    pin = settings["source"]["revision"]
    workspace = yaml.safe_load((ROOT / "ros2/workspace.repos").read_text())
    if workspace["repositories"]["lbr_fri_ros2_stack"]["version"] != pin:
        raise ValueError("Robot description pin differs from ros2/workspace.repos; review the source revision")
    repo = ROOT / "data/external/iiwa7-source"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--depth", "1", "--no-checkout",
                        settings["source"]["repository"], str(repo)], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--cone", "lbr_description/urdf/iiwa7",
                        "lbr_description/meshes/iiwa7/collision"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", pin], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", pin], check=True, env=env)
    revision = subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo),
                                        "rev-parse", "HEAD"], text=True).strip()
    if revision != pin:
        raise ValueError(f"Existing iiwa source is {revision}, expected {pin}; checkout left unchanged")
    if subprocess.check_output(["git", "-c", f"safe.directory={repo}", "-C", str(repo), "status", "--porcelain"], text=True):
        raise ValueError("Official iiwa source contains local changes; source left unchanged")
    description = repo / "lbr_description"
    source = description / "urdf/iiwa7/iiwa7_description.xacro"
    out = Path(settings["urdf"]).parent
    out.mkdir(parents=True, exist_ok=True)
    # Edit a temporary copy; the official source and its license remain intact.
    xml = ET.fromstring(source.read_text().replace("$(find lbr_description)", str(description)))
    macro = xml.find("{http://www.ros.org/wiki/xacro}macro")
    for child in list(macro):
        if child.tag in {"{http://www.ros.org/wiki/xacro}" + tag for tag in
                         ("include", "lbr_gazebo", "lbr_system_interface")}:
            macro.remove(child)
    with tempfile.TemporaryDirectory(prefix="iiwa7-xacro-") as tmp:
        stage = Path(tmp)
        ET.register_namespace("xacro", "http://www.ros.org/wiki/xacro")
        ET.ElementTree(xml).write(stage / "description.xacro", encoding="unicode")
        (stage / "robot.xacro").write_text(
            '<robot name="iiwa7_r800" xmlns:xacro="http://www.ros.org/wiki/xacro">'
            f'<xacro:include filename="{stage / "description.xacro"}"/>'
            '<xacro:iiwa7 robot_name="lbr"/></robot>')
        robot = ET.fromstring(xacro.process_file(str(stage / "robot.xacro")).toxml())
    for link in robot.findall("link"):
        collision = link.find("collision")
        if collision is not None:
            visual = link.find("visual")
            if visual is not None:
                link.remove(visual)
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            material = ET.SubElement(visual, "material", name="iiwa_silver")
            ET.SubElement(material, "color", rgba="0.75 0.78 0.82 1")
            link.append(visual)
        elif link.get("name") == settings["flange_link"]:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0")
            ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")
    for mesh in robot.iter("mesh"):
        relative = mesh.get("filename").removeprefix("package://lbr_description/")
        source_mesh = description / relative
        destination = out / "meshes" / source_mesh.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source_mesh, destination)
        mesh.set("filename", "meshes/" + destination.name)
    robot.insert(0, ET.Comment(" Derived from pinned LBR Stack; see description.json and LICENSE. "
                               "ROS/Gazebo omitted; collision STLs used for visuals; virtual flange has zero mass. "))
    ET.indent(robot)
    urdf = Path(settings["urdf"])
    ET.ElementTree(robot).write(urdf, encoding="unicode", xml_declaration=True)
    (out / "LICENSE").write_bytes(subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "show", f"{pin}:LICENSE"], env=env))
    inputs = out / "source"
    inputs.mkdir(exist_ok=True)
    for file in (source, source.with_name("joint_limits.yaml")):
        shutil.copy2(file, inputs / file.name)
    assets = {str(path.relative_to(out)): file_sha256(path) for path in
              [urdf, out / "LICENSE", *sorted((out / "meshes").glob("*.stl")), *sorted(inputs.iterdir())]}
    metadata = dict(schema_version=1, model="iiwa7_r800", urdf=urdf.name, source=settings["source"],
                    assets_sha256=assets, changes=["ROS/Gazebo omitted", "collision STL visuals", "zero-mass virtual flange"],
                    physical_grasp_validated=False)
    Path(settings["metadata"]).write_text(json.dumps(metadata, indent=2) + "\n")
    model = RobotModel.from_config({"robot": {"model": "iiwa7_r800"}})
    print(f"Prepared {model.model_id}: {urdf}; flange {model.flange_link}; source {revision}")
    return model


if __name__ == "__main__":
    prepare_iiwa7()
