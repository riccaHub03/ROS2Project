"""Expand pinned official Robotiq Xacro for standalone PyBullet use.

Asset preparation only: no physics client, inference or motion is started.
Original sources and licenses remain in data/external/robotiq-ros.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import xacro

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5b7f9189cb26197a4f6850937a1ee4ce667a3577"


def transform(origin):
    from trimesh.transformations import euler_matrix
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    t = euler_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def prepare():
    repo = ROOT / "data/external/robotiq-ros"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Robotiq source revision changed; review the model before preparing assets")
    description = repo / "grippers/robotiq_description"
    out = ROOT / "data/external/robotiq-prepared"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robotiq-xacro-") as tmp:
        stage = Path(tmp)
        # Resolve the package includes without installing a ROS environment.
        for source in (description / "urdf").glob("*.xacro"):
            (stage / source.name).write_text(source.read_text().replace(
                "$(find robotiq_description)/urdf/", str(stage) + "/"))
        wrapper = stage / "standalone.xacro"
        wrapper.write_text(f'''<robot name="robotiq_2f_140" xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:include filename="{stage / 'robotiq_2f_140_macro.urdf.xacro'}"/>
<link name="graspgen_base"/>
<xacro:robotiq_gripper name="robotiq" prefix="" parent="graspgen_base" include_ros2_control="false">
  <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:robotiq_gripper>
<link name="graspgen_tcp"/>
<joint name="graspgen_tcp_joint" type="fixed"><parent link="graspgen_base"/>
<child link="graspgen_tcp"/><origin xyz="0 0 0.195" rpy="0 0 {-math.pi / 2}"/></joint>
</robot>''')
        xml = ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    for link in xml.findall("link"):
        # PyBullet does not reliably render DAE. Reuse the official collision
        # mesh for visual geometry as well, with the same link origin.
        collision = link.find("collision")
        visual = link.find("visual")
        if collision is not None and visual is not None:
            link.remove(visual)
            visual = ET.fromstring(ET.tostring(collision))
            visual.tag = "visual"
            material = ET.SubElement(visual, "material", name="robotiq_grey")
            ET.SubElement(material, "color", rgba="0.2 0.24 0.28 1")
            link.append(visual)
    for mesh in xml.iter("mesh"):
        name = mesh.get("filename")
        name = name.replace("package://robotiq_description/", str(description) + "/")
        if not Path(name).is_file():
            raise FileNotFoundError(name)
        mesh.set("filename", name)
    urdf = out / "robotiq_2f_140.urdf"
    ET.indent(xml)
    ET.ElementTree(xml).write(urdf, encoding="unicode", xml_declaration=True)
    joints = []
    for joint in xml.findall("joint"):
        mimic = joint.find("mimic")
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append({"name": joint.get("name"), "type": joint.get("type"),
                       "parent": joint.find("parent").get("link"), "child": joint.find("child").get("link"),
                       "origin": transform(origin if origin is not None else ET.Element("origin")).tolist(),
                       "axis": [float(x) for x in axis.get("xyz").split()] if axis is not None else [0., 0., 1.],
                       "multiplier": float(mimic.get("multiplier", "1")) if mimic is not None else 1.,
                       "mimic": mimic.get("joint") if mimic is not None else None})
    # Static URDF FK establishes the pad opening curve, including mimic signs.
    from trimesh.transformations import rotation_matrix
    angles = np.linspace(0., .77, 301)
    widths = []
    for angle in angles:
        poses = {"graspgen_base": np.eye(4)}
        pending = list(joints)
        while pending:
            ready = [j for j in pending if j["parent"] in poses]
            if not ready:
                raise ValueError("URDF joint graph is not a tree")
            for j in ready:
                q = angle * j["multiplier"] if j["type"] in ("revolute", "continuous") else 0.
                poses[j["child"]] = poses[j["parent"]] @ j["origin"] @ rotation_matrix(q, j["axis"])
                pending.remove(j)
        a, b = (poses[name][:3, 3] for name in ("left_inner_finger_pad", "right_inner_finger_pad"))
        # Signed separation along GraspGen X; stop before the pads cross.
        widths.append(float(a[0] - b[0]))
    valid = next((i for i, width in enumerate(widths) if width <= 0), len(widths))
    angles, widths = angles[:valid], widths[:valid]
    if not all(a > b for a, b in zip(widths[:-1], widths[1:])):
        raise ValueError("Official pad opening is not monotonic over the configured joint range")
    metadata = {"source_url": "https://github.com/robotiq/robotiq-ros", "revision": revision,
                "urdf": str(urdf), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                "geometry": "official collision meshes used for collision and visualization",
                "root_frame": "graspgen_base", "tcp_frame": "graspgen_tcp",
                "T_grasp_tcp": [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, .195], [0, 0, 0, 1]],
                "joints": joints, "opening_angles_rad": angles.tolist(), "opening_widths_m": widths,
                "physical_contact_validated": False}
    (out / "description.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {urdf}; official joints and collision meshes; no simulation started")


if __name__ == "__main__":
    prepare()
