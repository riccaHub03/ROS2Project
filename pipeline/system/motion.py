"""Simulation speed references; retain configured dynamics and hardware velocity ceilings."""
import math


def motion_speed(value):
    factor = float(value)
    if not math.isfinite(factor) or not 0.25 <= factor <= 2:
        raise ValueError("motion speed must be finite and between 0.25 and 2")
    return factor


def configure_motion(control, factor=1.):
    factor = motion_speed(factor)
    ik = control.setdefault("differential_ik", {})
    original = ik.get("joint_max_velocity_rad_s", [.35] * 7)
    ik.update(enabled=True, position_tolerance_m=.0005, orientation_tolerance_rad=.0025,
              joint_max_velocity_rad_s=[min(float(v), .35 * factor) for v in original])
    velocities = control.setdefault("velocities", {})
    for key, reference in (("cartesian_linear_speed_m_s", .04), ("cartesian_angular_speed_rad_s", .20)):
        velocities[key] = min(reference * factor, velocities.get(key, reference * factor))
    control["simulation_speed_factor"] = factor
