"""Test di validazione per il Controllore KUKA W-S-A-D e Puntatore."""

import unittest
import numpy as np

from interactive_controller import InteractiveRobotController


class TestInteractiveRobotController(unittest.TestCase):
    def setUp(self):
        self.controller = InteractiveRobotController(
            tool_name="Scalpel",
            source_tray="tray_2",
            gui=False,  # Headless test
            jog_speed=0.22,
            start_with_mouse_pointer=False,
        )

    def tearDown(self):
        self.controller.close()

    def test_initialization(self):
        self.assertGreaterEqual(self.controller.robot_id, 0)
        self.assertGreaterEqual(self.controller.gripper_id, 0)
        self.assertGreaterEqual(self.controller.table_id, 0)
        self.assertEqual(len(self.controller.joints), 7)
        self.assertFalse(self.controller.mouse_pointer_mode)

    def test_tcp_pose_and_ik_tracking(self):
        initial_tcp_pos, _ = self.controller.get_tcp_pose()
        self.assertEqual(len(initial_tcp_pos), 3)

        # Sposta target verso il vassoio
        target_pos = [0.60, -0.42, 0.20]
        self.controller.solve_and_apply_ik(target_pos, self.controller.target_tcp_orn)

        for _ in range(30):
            self.controller.step(dt=1.0/60.0)

        actual_tcp_pos, _ = self.controller.get_tcp_pose()
        error = np.linalg.norm(np.asarray(actual_tcp_pos) - np.asarray(target_pos))
        self.assertLess(error, 0.05)

    def test_cycle_camera_views(self):
        views = [
            InteractiveRobotController.VIEW_CAMERA,
            InteractiveRobotController.VIEW_ORBIT,
            InteractiveRobotController.VIEW_WRIST,
            InteractiveRobotController.VIEW_SURGEON,
        ]
        for v in views:
            self.controller.set_camera_view(v)
            self.assertEqual(self.controller.current_view, v)

    def test_toggle_grasp_and_reset_home(self):
        tool_pos = self.controller.tool_initial_pos
        self.controller.solve_and_apply_ik(tool_pos, self.controller.target_tcp_orn)
        for _ in range(20):
            self.controller.step(dt=1.0/60.0)

        self.controller.toggle_grasp()
        self.controller.reset_home()
        self.assertIsNone(self.controller.attachment_constraint)

    def test_mouse_pointer_toggle(self):
        self.assertFalse(self.controller.mouse_pointer_mode)
        # Attiva modalità puntatore
        self.controller.mouse_pointer_mode = True
        self.controller.mouse_world_target = np.array([0.55, -0.38, 0.08])
        for _ in range(15):
            self.controller.step(dt=1.0/60.0)
        actual_pos, _ = self.controller.get_tcp_pose()
        self.assertAlmostEqual(actual_pos[0], 0.55, delta=0.08)
        self.assertAlmostEqual(actual_pos[1], -0.38, delta=0.08)


if __name__ == "__main__":
    unittest.main()
