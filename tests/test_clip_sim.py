"""The clip adapter must preserve candidates and never clear hardware motion."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from pipeline.graspgen.config import load_config
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
from scripts.test_clip_sim import ClipMotion


class ClipMotionValidationTests(unittest.TestCase):
    def setUp(self):
        cfg = load_config(None)
        metadata = {'gripper_name': cfg['model_gripper'], 'model_name': 'diffusion-discriminator',
                    'gripper_config': cfg['checkpoint_config'],
                    'ros2project': {'revision': cfg['expected_revision'], 'planner': 'diffusion',
                                    'config_sha256': 'config', 'generator_sha256': 'gen',
                                    'discriminator_sha256': 'dis'}}
        pose = np.eye(4).tolist()
        candidate = {'server_index': 7, 'score': .8, 'T_base_gripper': pose, 'T_base_tcp': pose}
        self.original = {'configuration': cfg, 'server_metadata': metadata, 'candidates': [candidate]}
        self.report = {'source_pipeline': 'real', 'motion': {'allowed': False},
                       'candidates': copy.deepcopy(self.original['candidates'])}
        self.adapter = object.__new__(ClipMotion)
        self.adapter.scene = SimpleNamespace(gripper_geometry=ROBOTIQ_PROXY)
        self.adapter._original = self.original
        self.adapter._mapping = np.eye(4)
        self.adapter._manifest = {'source_cloud_sha256': 'abc', 'assumptions': ['proxy']}
        self.adapter.validation = {}

    def test_matching_source_keeps_hardware_blocked(self):
        self.adapter._check_source(self.report)
        self.assertFalse(self.adapter.validation['hardware_motion_allowed'])

    def test_rejects_inconsistent_candidate_transform(self):
        self.report['candidates'][0]['T_base_gripper'][0][3] = .1
        with self.assertRaisesRegex(ValueError, 'transforms do not match'):
            self.adapter._check_source(self.report)

    def test_rejects_report_with_unblocked_hardware(self):
        self.report['motion']['allowed'] = True
        with self.assertRaisesRegex(ValueError, 'must stay blocked'):
            self.adapter._check_source(self.report)


if __name__ == '__main__':
    unittest.main()
