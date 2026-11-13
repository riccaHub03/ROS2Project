"""Replay saved poses in a CPU-only collision model. No server, GUI or inference.

Must be invoked manually unless the user explicitly authorizes diagnostic runs.
Does not step physics, regenerate data, or command the live simulation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.robot_cell import RobotCell
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
from pipeline.sim.grasp_refinement import refine_grasp, TCP
from pipeline.frame_transformer import quaternion_from_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=120.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output file to preserve previous diagnostics')
    capture = json.loads((args.run_dir/'sim_capture.json').read_text())['data']
    report = json.loads((args.run_dir/'grasp_graspgen.json').read_text())['data']
    config = yaml.safe_load(args.config.read_text())
    config['simulation_gripper'] = ROBOTIQ_PROXY.name
    base = RobotBaseFrame.from_dict(capture['robot_base_pose_world'])
    pose = capture['scene_object_pose_exact']
    position, quaternion = base.pose_world_to_base(pose['position_xyz_m'], pose['orientation_xyzw'])
    result = {'run_dir': str(args.run_dir.resolve()), 'physics_steps': 0, 'inference_requests': 0,
              'physical_grasp_validated': False, 'attempts': []}
    start = time.monotonic()
    def heartbeat():
        if time.monotonic()-start > args.budget_s:
            raise TimeoutError('Diagnostic budget reached; remaining candidates not evaluated')
    with RobotCell(config, base, position_tolerance_m=.001, angle_tolerance_rad=.005) as cell:
        cell.configure_tool_mesh(capture['mesh_file'])
        cell.set_stationary_tool([.1, .1, .1], position, quaternion)  # mesh overrides box extents
        for candidate in sorted(report['candidates'], key=lambda c: -c['score']):
            begin = time.monotonic()
            item = {'server_index': candidate['server_index'], 'source_score': candidate['score']}
            try:
                transform, width, refinement = refine_grasp(
                    cell, np.asarray(candidate['T_base_gripper']), tuple(config['environment']['trays']), heartbeat)
                item.update(geometry_passed=True, refinement=refinement,
                            refined_T_base_tcp=(transform@TCP).tolist())
                # One diagnostic IK attempt; this does NOT authorize motion.
                tcp = transform@TCP
                sol = cell.solve_ik(tcp[:3, 3], quaternion_from_matrix(tcp[:3, :3]),
                                    [0, .35, 0, -1.3, 0, 1., 0])
                item['single_seed_ik'] = {'ok': sol.ok, 'position_error_m': sol.position_error_m,
                                          'joint_margin_rad': sol.joint_margin_rad}
            except (ValueError, TimeoutError) as exc:
                item.update(geometry_passed=False, error=str(exc))
                if isinstance(exc, TimeoutError):
                    result['search_incomplete'] = True
            item['elapsed_s'] = round(time.monotonic()-begin, 3)
            result['attempts'].append(item)
            print(f"#{item['server_index']} geometry={item['geometry_passed']} {item.get('error', '')}", flush=True)
            result['elapsed_s'] = round(time.monotonic()-start, 3)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            if item['geometry_passed'] or result.get('search_incomplete'):
                break
    print('Summary:', dict(Counter(x['geometry_passed'] for x in result['attempts'])))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
