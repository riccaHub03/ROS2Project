"""Recorded RGB-D input for the real pipeline; no robot or network required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from pipeline.common.report import StageError
from pipeline.config_loader import load_environment_config
from pipeline.real import capture, pipeline


class RecordedClipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clip = self.root / 'clip'
        self.clip.mkdir()
        self.raw = np.arange(24, dtype=np.uint16).reshape(4, 6) + 100
        Image.fromarray(np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)).save(self.clip / 'color_000003.png')
        Image.fromarray(self.raw).save(self.clip / 'depth_000003.png')
        self.meta = {'intrinsics_color': {'fu': 20., 'fv': 21., 'ppu': 3., 'ppv': 2., 'width': 6, 'height': 4},
                     'depth_aligned_to_color': True}
        (self.clip / 'camera.yaml').write_text(yaml.safe_dump(self.meta))
        self.options = {'directory': str(self.clip), 'index': 3, 'depth_scale': .001, 'crop': [1, 1, 4, 2]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_capture_uses_crop_scaled_depth_and_never_enables_motion(self):
        config = load_environment_config(None)
        config['camera_overhead']['extrinsics']['verified'] = True
        config_path = self.root / 'verified.yaml'
        config_path.write_text(yaml.safe_dump(config))
        report = capture.run(source='clip', clip=self.options, config_path=str(config_path), out_dir=str(self.root / 'out'))
        self.assertEqual(report.data['source_type'], 'recorded_clip_rgbd')
        self.assertFalse(report.data['motion_safe'])
        self.assertFalse(report.data['extrinsics_verified'])
        self.assertEqual(report.data['camera']['resolution_wh'], [4, 2])
        self.assertEqual(report.data['camera']['cx_cy'], [2., 1.])
        np.testing.assert_allclose(np.load(report.artifact('depth_npy')), self.raw[1:3, 1:5] * .001, rtol=1e-6)
        self.assertEqual(json.loads(report.artifact('clip_input_json').read_text())['index'], 3)

    def test_clip_rejects_unknown_scale_unaligned_or_missing_pair(self):
        for opts in ({**self.options, 'depth_scale': None}, {**self.options, 'depth_scale': 0},
                     {**self.options, 'index': 4}, {**self.options, 'crop': [5, 0, 2, 2]}):
            with self.subTest(opts=opts), self.assertRaises(StageError):
                capture.run(source='clip', clip=opts, out_dir=str(self.root / 'bad'))
        self.meta['depth_aligned_to_color'] = False
        (self.clip / 'camera.yaml').write_text(yaml.safe_dump(self.meta))
        with self.assertRaises(StageError):
            capture.run(source='clip', clip=self.options, out_dir=str(self.root / 'bad'))

    def test_explicit_missing_image_does_not_fall_back(self):
        with self.assertRaisesRegex(StageError, 'image file not found'):
            capture.find_image(str(self.root / 'missing.png'))

    def test_require_llm_fails_before_capture(self):
        out = self.root / 'llm_missing'
        with patch('pipeline.plan.query_ollama', return_value=(None, None)):
            result = pipeline.main(['--source', 'clip', '--clip-dir', str(self.clip), '--frame-index', '3',
                                    '--depth-scale', '.001', '--require-llm', '--out-dir', str(out)])
        self.assertEqual(result, 1)
        self.assertFalse((out / 'real_capture.json').exists())

    def test_wrong_target_fails_before_capture(self):
        out = self.root / 'wrong_target'
        result = {'action': 'pick_and_place', 'target_instrument': 'scalpel', 'target_keyword': 'scalpel',
                  'source_tray': 'tray_2', 'destination_tray': 'tray_1'}
        with patch('pipeline.plan.query_ollama', return_value=(result, 'http://localhost')):
            code = pipeline.main(['--source', 'clip', '--clip-dir', str(self.clip), '--frame-index', '3',
                                  '--depth-scale', '.001', '--require-llm', '--expected-keyword', 'scissors',
                                  '--out-dir', str(out)])
        self.assertEqual(code, 1)
        self.assertFalse((out / 'real_capture.json').exists())

    def test_nonempty_clip_output_rejected(self):
        out = self.root / 'existing'
        out.mkdir()
        (out / 'previous.txt').write_text('preserve')
        code = pipeline.main(['--source', 'clip', '--out-dir', str(out)])
        self.assertEqual(code, 1)
        self.assertEqual((out / 'previous.txt').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
