import os
import unittest
from unittest.mock import Mock, patch

from studio.scoreboard.runtime import create_session


class RuntimeTests(unittest.TestCase):
    @patch.dict(os.environ, {'SCORESIGHT_DEVICE': 'cuda'})
    def test_cuda_failure_cannot_silently_run_on_cpu(self):
        ort = Mock()
        ort.get_available_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort.InferenceSession.return_value.get_providers.return_value = ['CPUExecutionProvider']
        with patch.dict('sys.modules', {'onnxruntime': ort}):
            with self.assertRaisesRegex(RuntimeError, 'refusing silent CPU fallback'):
                create_session('model.onnx', None)

    @patch.dict(os.environ, {'SCORESIGHT_DEVICE': 'cuda'})
    def test_cuda_requires_gpu_runtime(self):
        ort = Mock()
        ort.get_available_providers.return_value = ['CPUExecutionProvider']
        with patch.dict('sys.modules', {'onnxruntime': ort}):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                create_session('model.onnx', None)
        ort.InferenceSession.assert_not_called()
