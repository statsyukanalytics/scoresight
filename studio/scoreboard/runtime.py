"""Explicit ONNX device selection shared by OCR and scoreboard detection."""
import os


def create_session(path, options):
    import onnxruntime as ort

    device = os.environ.get('SCORESIGHT_DEVICE', 'cpu').lower()
    if device not in ('cpu', 'cuda'):
        raise ValueError('SCORESIGHT_DEVICE must be cpu or cuda')
    provider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
    if device == 'cuda':
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        if provider not in ort.get_available_providers():
            raise RuntimeError('CUDA requested but onnxruntime-gpu is unavailable')
    session = ort.InferenceSession(path, options, providers=[provider])
    if provider not in session.get_providers():
        raise RuntimeError(f'{provider} failed to load; refusing silent CPU fallback')
    session.disable_fallback()
    return session


def batch_frames():
    """Bound CPU work and speculative crops separately from GPU throughput."""
    default = 8 if os.environ.get('SCORESIGHT_DEVICE', 'cpu').lower() == 'cpu' else 16
    value = int(os.environ.get('SCORESIGHT_BATCH_FRAMES', str(default)))
    if not 1 <= value <= 32:
        raise ValueError('SCORESIGHT_BATCH_FRAMES must be between 1 and 32')
    return value
