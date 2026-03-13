import unittest
from unittest.mock import MagicMock
import sys
import subprocess
import os

# Mock external deps for app import
mock_gradio = MagicMock()
mock_matplotlib = MagicMock()
mock_librosa = MagicMock()
mock_pydub = MagicMock()
mock_numpy = MagicMock()
mock_scipy = MagicMock()

sys.modules['gradio'] = mock_gradio
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib.pyplot
sys.modules['librosa'] = mock_librosa
sys.modules['librosa.display'] = mock_librosa.display
sys.modules['pydub'] = mock_pydub
sys.modules['numpy'] = mock_numpy
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.signal'] = mock_scipy.signal

import app


class TestInferenceCutoffUnits(unittest.TestCase):
    def setUp(self):
        self.orig_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_defaults_to_khz_cutoff(self):
        ok = MagicMock(stdout='', stderr='')
        app.subprocess.run = MagicMock(return_value=ok)
        app.os.path.exists = MagicMock(return_value=True)
        app.is_likely_corrupted_audio = MagicMock(return_value=False)

        app.run_a2sb_inference('/tmp/in.wav', '/tmp/out.wav', 50, 14000)

        args, _ = app.subprocess.run.call_args
        cmd = args[0]
        self.assertIn('-c', cmd)
        self.assertEqual(cmd[cmd.index('-c') + 1], '14')

    def test_allows_hz_override_via_env(self):
        os.environ['A2SB_CUTOFF_UNIT'] = 'hz'
        ok = MagicMock(stdout='', stderr='')
        app.subprocess.run = MagicMock(return_value=ok)
        app.os.path.exists = MagicMock(return_value=True)
        app.is_likely_corrupted_audio = MagicMock(return_value=False)

        app.run_a2sb_inference('/tmp/in.wav', '/tmp/out.wav', 50, 14000)

        args, _ = app.subprocess.run.call_args
        cmd = args[0]
        self.assertEqual(cmd[cmd.index('-c') + 1], '14000')

    def test_falls_back_to_hz_when_khz_output_is_invalid(self):
        ok = MagicMock(stdout='', stderr='')
        app.subprocess.run = MagicMock(side_effect=[ok, ok])
        app.os.path.exists = MagicMock(return_value=True)
        app.is_likely_corrupted_audio = MagicMock(side_effect=[True, False])

        app.run_a2sb_inference('/tmp/in.wav', '/tmp/out.wav', 50, 14000)

        first_cmd = app.subprocess.run.call_args_list[0][0][0]
        second_cmd = app.subprocess.run.call_args_list[1][0][0]
        self.assertEqual(first_cmd[first_cmd.index('-c') + 1], '14')
        self.assertEqual(second_cmd[second_cmd.index('-c') + 1], '14000')

    def test_retries_with_hz_when_khz_command_fails(self):
        failure = subprocess.CalledProcessError(
            returncode=2,
            cmd=['python3', 'A2SB_upsample_api.py'],
            stderr='invalid cutoff',
        )
        ok = MagicMock(stdout='', stderr='')
        app.subprocess.run = MagicMock(side_effect=[failure, ok])
        app.os.path.exists = MagicMock(return_value=True)
        app.is_likely_corrupted_audio = MagicMock(return_value=False)

        app.run_a2sb_inference('/tmp/in.wav', '/tmp/out.wav', 50, 14000)

        first_cmd = app.subprocess.run.call_args_list[0][0][0]
        second_cmd = app.subprocess.run.call_args_list[1][0][0]
        self.assertEqual(first_cmd[first_cmd.index('-c') + 1], '14')
        self.assertEqual(second_cmd[second_cmd.index('-c') + 1], '14000')


if __name__ == '__main__':
    unittest.main()
