"""Auto crop and Auto layout: the scoreboard and its fields, found by an AI
model on the picture.

The model (D-FINE-S, Apache-2.0) finds six kinds of boxes on a picture: the
home score, the away score, the clock, the period, other fields and the
scoreboard itself. It is ScoreSight's own, published on Hugging Face
(MODEL_REPO, with its evaluation), and it isn't bundled: it is downloaded the
first time it is used, at a pinned revision, checked against its pinned
SHA-256, into models_dir(). The pictures never leave the computer.

This module doesn't touch the UI: the main window (auto_layout_ui) runs it on a worker
thread and turns its results into the crop and the fields' boxes.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import certifi
import cv2
import numpy as np

import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def get_data_dir():
    return str(Path(os.environ.get("SCORESIGHT_LAYOUT_RESOURCES", Path(__file__).parent)))

MODEL_REPO = "royshilkrot/scoreboard-layout"
MODEL_REVISION = "e8c58bd89bd3b7a8ce43be9024dc254cb7b889f8"
MODEL_FILE = "fields_board.onnx"
MODEL_SHA256 = "511919c6bee1d898ecba9a80d5ab98fe3f9f238021a799cfcfdcb48ccd75515a"
MODEL_SIZE = 41_663_092
DEFAULT_ENDPOINT = "https://huggingface.co"

# the model's classes, by its label index
CLASSES = ("home", "away", "clock", "period", "other", "board")
# the predefined field each role fills
ROLE_FIELDS = {
    "home": "Home Score",
    "away": "Away Score",
    "clock": "Time",
    "period": "Period",
}
INPUT_SIZE = 640
# a box with a lower score is ignored
MIN_SCORE = 0.3
# Auto crop keeps this much of the scoreboard's size around it, on each side
CROP_MARGIN = 0.1
# a field's box is grown by this share of its height on each side: the model's
# boxes are tight around the digits
FIELD_PADDING = 0.1
# two fields' boxes overlapping more than this are the same field
SAME_BOX_IOU = 0.5

THREADS = 2
ATTEMPTS = 3
TIMEOUT_S = 60.0
CHUNK = 1 << 20


@dataclass(frozen=True)
class Detection:
    kind: str  # one of CLASSES
    rect: tuple[float, float, float, float]  # x, y, width, height in picture pixels
    score: float


class DownloadCancelled(Exception):
    pass


def models_dir() -> str:
    return os.path.join(get_data_dir(), "models")


def model_path() -> str:
    """Where the model is kept: one folder per revision, so a new pinned
    revision downloads its own file next to the old one."""
    return os.path.join(
        models_dir(), "scoreboard-layout", MODEL_REVISION[:12], MODEL_FILE
    )


def model_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip() or DEFAULT_ENDPOINT
    return f"{endpoint.rstrip('/')}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"


def model_available() -> bool:
    """Whether the model was downloaded (its hash was checked then)."""
    path = model_path()
    return os.path.isfile(path) and os.path.getsize(path) == MODEL_SIZE


def download_model(
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """Download the model unless it is there; returns its path. progress(done,
    total) is called as the bytes arrive (on the calling thread). The file is
    written as .part and moved into place only once its SHA-256 matches, so an
    interrupted download never leaves a bad model behind. Network errors are
    retried; raises OSError (or DownloadCancelled) if it can't be done."""
    path = model_path()
    if model_available():
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    url = model_url()
    error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(2.0 * attempt)
        try:
            _download(url, part, progress, cancelled)
            digest = _sha256(part)
            if digest != MODEL_SHA256:
                raise OSError(
                    f"the downloaded model's SHA-256 is {digest}, not {MODEL_SHA256}"
                )
            os.replace(part, path)
            logger.info("Downloaded the layout model to %s", path)
            return path
        except DownloadCancelled:
            _remove(part)
            raise
        except (OSError, urllib.error.URLError) as e:
            error = e
            logger.warning(
                "Downloading the layout model failed (attempt %d of %d): %s",
                attempt + 1,
                ATTEMPTS,
                e,
            )
            _remove(part)
    raise OSError(f"could not download the layout model from {url}: {error}")


def _tls_context() -> ssl.SSLContext:
    """Trusts certifi's certificates, like requests (the update check): the
    Python of the macOS app finds none of its own, so urllib's default
    failed there with CERTIFICATE_VERIFY_FAILED (#140)."""
    return ssl.create_default_context(cafile=certifi.where())


def _download(url, part, progress, cancelled):
    request = urllib.request.Request(url, headers={"User-Agent": "ScoreSight"})
    with urllib.request.urlopen(
        request, timeout=TIMEOUT_S, context=_tls_context()
    ) as response:
        total = int(response.headers.get("Content-Length") or MODEL_SIZE)
        done = 0
        with open(part, "wb") as f:
            while True:
                if cancelled is not None and cancelled():
                    raise DownloadCancelled()
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def to_model_input(frame: np.ndarray) -> np.ndarray:
    """A frame (BGR, BGRA or gray, as the capture reads them) as the model's
    input: 1 x 3 x 640 x 640, RGB, 0-1, the whole frame resized."""
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


class LayoutDetector:
    """The model, loaded once (onnxruntime is imported on first use)."""

    def __init__(self, path: str):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = THREADS
        self.session = onnxruntime.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )

    def detect(
        self, frame: np.ndarray, min_score: float = MIN_SCORE
    ) -> list[Detection]:
        """The boxes found on frame with at least min_score, best first."""
        height, width = frame.shape[:2]
        labels, boxes, scores = self.session.run(
            None,
            {
                "images": to_model_input(frame),
                "orig_target_sizes": np.array([[width, height]], np.int64),
            },
        )
        return decode(labels[0], boxes[0], scores[0], width, height, min_score)


def decode(
    labels, boxes, scores, width, height, min_score=MIN_SCORE
) -> list[Detection]:
    """The model's outputs as Detections inside the picture, best first."""
    found = []
    for label, (x1, y1, x2, y2), score in zip(labels, boxes, scores):
        if score < min_score or not 0 <= int(label) < len(CLASSES):
            continue
        x1, x2 = max(0.0, float(x1)), min(float(width), float(x2))
        y1, y2 = max(0.0, float(y1)), min(float(height), float(y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        found.append(
            Detection(CLASSES[int(label)], (x1, y1, x2 - x1, y2 - y1), float(score))
        )
    found.sort(key=lambda d: -d.score)
    return found


_detector: LayoutDetector | None = None
_detector_lock = threading.Lock()


def detector() -> LayoutDetector:
    """The loaded model (downloaded first if needed; see download_model)."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = LayoutDetector(download_model())
        return _detector


def iou(a, b) -> float:
    """Intersection over union of two (x, y, width, height) boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def board_crop(
    detections: list[Detection], width: int, height: int, margin: float = CROP_MARGIN
) -> tuple[int, int, int, int] | None:
    """Auto crop: the best scoreboard box grown by margin of its size on each
    side, within the picture, as (x, y, width, height) pixels; None without a
    scoreboard."""
    boards = [d for d in detections if d.kind == "board"]
    if not boards:
        return None
    x, y, w, h = max(boards, key=lambda d: d.score).rect
    left = max(0, int(np.floor(x - margin * w)))
    top = max(0, int(np.floor(y - margin * h)))
    right = min(width, int(np.ceil(x + w + margin * w)))
    bottom = min(height, int(np.ceil(y + h + margin * h)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right - left, bottom - top


def field_boxes(
    detections: list[Detection], width: int, height: int
) -> dict[str, tuple[int, int, int, int]]:
    """Auto layout: the predefined field of each role found, with its box as
    (x, y, width, height) pixels, padded and within the picture. Each role
    takes its best box; a box already taken by a role with a better score
    (the same digits named home and away) is left to it."""
    taken: list[tuple[float, float, float, float]] = []
    fields = {}
    candidates = [d for d in detections if d.kind in ROLE_FIELDS]
    # the roles whose best box is surest choose first
    for detection in sorted(candidates, key=lambda d: -d.score):
        name = ROLE_FIELDS[detection.kind]
        if name in fields or any(iou(detection.rect, t) > SAME_BOX_IOU for t in taken):
            continue
        taken.append(detection.rect)
        x, y, w, h = detection.rect
        pad = FIELD_PADDING * h
        left, top = max(0, round(x - pad)), max(0, round(y - pad))
        right = min(width, round(x + w + pad))
        bottom = min(height, round(y + h + pad))
        fields[name] = (left, top, right - left, bottom - top)
    return fields


if __name__ == '__main__':
    print(download_model())
