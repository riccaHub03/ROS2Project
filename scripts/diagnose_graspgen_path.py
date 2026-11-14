"""Manual CPU validation of a saved GraspGen set from HOME. No execution.

Reuses the production planner and actual saved mesh/capture/cloud. It creates
only kinematic collision models, never steps physics or starts a GUI/server.
Passing this check does not validate servo tracking, object transport or contact.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pybullet as p

from pipeline.common import paths
from pipeline.common.report import StageReport, write_json
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.robot_model import RobotModel
from pipeline.sim.graspgen_motion import GraspGenMotionPreview
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.run_dir.resolve(), args.out_dir.resolve()
    if out.exists():
        raise ValueError('Choose a NEW output directory; existing runs are preserved')
    required = ['sim_capture.json', 'sim_deprojection.json', 'sim_deprojection_cloud.ply',
                'grasp_graspgen.json', 'plan.json']
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    out.mkdir(parents=True)
    for name in required:
        shutil.copy2(source/name, out/name)
    os.environ[paths.OUTPUT_ENV] = str(out)
    config = load_environment_config(args.config)
    model = RobotModel.from_config(config)
    config['simulation_gripper'] = ROBOTIQ_PROXY.name
    control = config.setdefault('motion_control', {})
    control.setdefault('differential_ik', {}).update(
        enabled=True, joint_max_velocity_rad_s=[.35]*7,
        position_tolerance_m=.0005, orientation_tolerance_rad=.0025)
    capture = StageReport.load(out/'sim_capture.json').data
    report = StageReport.load(out/'grasp_graspgen.json').data
    plan = StageReport.load(out/'plan.json').data
    pose = capture['scene_object_pose_exact']
    base = RobotBaseFrame.from_dict(capture['robot_base_pose_world'])
    mesh_file = Path(capture['mesh_file'])
    metadata = json.loads(mesh_file.with_suffix('.json').read_text())
    if metadata['source_mesh'] != capture['source_stl_file']:
        raise ValueError('Cached mesh metadata and source capture disagree')
    summary = {'mode': 'saved_data_kinematic_path_check_from_HOME',
               'source_run': str(source), 'physics_steps': 0, 'inference_requests': 0,
               'trajectory_executed': False, 'physical_grasp_validated': False,
               'hardware_motion_allowed': False, 'start_q_rad': list(model.home),
               'robot_description': model.report(ROBOTIQ_PROXY)}
    planner = None
    cid = p.connect(p.DIRECT)
    try:
        robot, joints, flange = model.load(cid, base.position, base.quaternion)
        for joint, value in zip(joints, model.home):
            p.resetJointState(robot, joint, value, physicsClientId=cid)
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(mesh_file), physicsClientId=cid)
        tool = p.createMultiBody(0, shape, -1, pose['position_xyz_m'], pose['orientation_xyzw'],
                                 physicsClientId=cid)
        source_tray = config['environment']['trays'][plan['source_tray']]
        tool_bottom = source_tray['position_xyz_m'][2] + source_tray['dimensions_xyz_m'][2]/2 + .001
        scene = SimpleNamespace(
            config=config, base_frame=base, gripper_geometry=ROBOTIQ_PROXY,
            procedural_tool_fallback=False, mesh_file=str(mesh_file), mesh_info=metadata,
            cid=cid, robot=robot, joints=joints, ee=flange, robot_model=model,
            tool=tool, source=plan['source_tray'], destination=plan['destination_tray'],
            trays=config['environment']['trays'], tool_bottom=tool_bottom,
            gui=False, status=lambda message: print(message, flush=True),
            show_grasp=lambda report: None, show_path=lambda waypoints: None)
        planner = GraspGenMotionPreview(scene, report)
        summary.update(status='PATH_CHECK_PASSED_NOT_EXECUTED',
                       selected_server_index=planner.validation['selected_server_index'],
                       geometric_refinement=planner.validation['geometric_refinement'],
                       phases=planner.waypoints,
                       planned_duration_s=sum(phase['duration'] for phase in planner.phases))
        StageReport('grasp', data=report).save(out/'grasp_simulation_selection.json')
        code = 0
    except (ValueError, RuntimeError) as exc:
        summary.update(status='PATH_CHECK_FAILED', error=str(exc))
        print(str(exc), flush=True)
        code = 1
    finally:
        if planner is not None:
            planner.close()
        p.disconnect(cid)
    write_json(out/'saved_path_check.json', summary)
    print(summary['status'], flush=True)
    print(out/'saved_path_check.json', flush=True)
    return code


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    raise SystemExit(main())
