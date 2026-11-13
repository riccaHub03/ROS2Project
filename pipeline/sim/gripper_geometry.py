"""Shared TCP conventions. The Robotiq profile loads official articulated meshes.

The dimensions below remain useful for the conservative mesh cross-section
estimate. They no longer generate Robotiq collision/display boxes. Finger
control is kinematic; physical force/contact grasp stability is not validated.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperGeometry:
    name: str
    tcp_offset: tuple
    opening: float
    palm_half: tuple
    palm_z: float
    finger_half: tuple
    finger_z: float
    finger_rest_y: float = 0.005


DEMO = GripperGeometry("visual_parallel_jaws", (0., 0., .10), .080,
                       (.032, .025, .0125), .025, (.013, .004, .025), .075)
ROBOTIQ_PROXY = GripperGeometry("robotiq_2f_140_official_meshes", (0., 0., .195), .136,
                               (.043, .050, .035), .040, (.013, .004, .025), .170)


def for_config(config):
    name = config.get("simulation_gripper", DEMO.name)
    if name == ROBOTIQ_PROXY.name:
        return ROBOTIQ_PROXY
    if name != DEMO.name:
        raise ValueError(f"Unknown simulation gripper: {name}")
    return DEMO
