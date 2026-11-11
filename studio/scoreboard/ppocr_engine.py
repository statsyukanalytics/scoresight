"""The PP-OCR engine: PaddleOCR's PP-OCRv6 text recognition models on
onnxruntime, next to Tesseract (tesseract.py).

A box already is a line of text, so only the recognition model runs (no text
detection). The models are PaddlePaddle's ONNX exports, downloaded by
scripts/fetch_models.py into ocr_models/ppocr/ (bundled by scoresight.spec).
The choices (stored as ocr_model, see ocr_models.py):

  ppocr:auto   "PP-OCR (automatic)", the default for new setups: accurate
               while it keeps up with Detections / s, fast when it can't
               (AutoChooser)
  ppocr:small  PP-OCRv6_small_rec, "PP-OCR (accurate)": 5.2M parameters, 21 MB
  ppocr:tiny   PP-OCRv6_tiny_rec, "PP-OCR (fast)": 1.1M parameters, 4.5 MB

The preprocessing and decoding are the OCR evaluation's (ocr_eval/engines/
ppocr.py), which match RapidOCR's:
- the box's colour crop (BGR, as the capture thread has it after rotation,
  crop, stabilization and four-corner correction, never binarized: the
  evaluation measured that ScoreSight's binarization costs these models 28-44
  points and gray about 1) is resized to the model's height (48 px) keeping its
  aspect ratio, scaled to [-1, 1] and padded with zeros on the right to at
  least the model's minimum width (320 px);
- greedy CTC decoding (PaddleOCR's CTCLabelDecode: the classes are the blank,
  characters.txt and a space), limited to the box's characters (its field
  type's whitelist, defaults.CHAR_WHITELISTS): at every time step the blank
  competes only with the allowed characters, and characters that are the same
  under NFKC (e.g. full-width digits) count as one. The confidence (0-1) is the
  mean probability of the characters kept; the evaluation found it well
  calibrated (at 0.9, about 1% of the crops were published wrong).

All of a frame's boxes are read together (Recognizer.recognize): one
session.run per distinct padded width, so a crop's reading never depends on the
other boxes in the frame. Every box up to 6.67:1 (320/48) has the same padded
width, so a frame usually takes a single run.

onnxruntime is imported, and the model loaded, on a background thread when the
model is first selected (start_loading): the import takes about 66 ms and
creating the small model's session 60 ms, which would otherwise hold up the
window or the capture thread. Until it is ready, TextDetector returns no
reading (Empty) for the boxes; if it fails, TextDetector falls back to
Tesseract and says so. Recognizers are shared (get_recognizer): onnxruntime's
sessions can run on several threads at once.

The cost: on a fast desktop CPU's performance core the small model takes about
17 ms per box on one thread, on one of its efficiency cores about 43 ms (10
boxes read 5 times a second: 87% and 213% of a core); tiny takes a fifth of
that. Laptop cores are more like the efficiency cores, hence PP-OCR
(automatic). The sessions use INTRA_OP_THREADS (2) onnxruntime threads that
sleep instead of spinning while idle: two threads make a read about 1.8 times
faster for about 5% more CPU time (spinning costs CPU and gains nothing here).
Set SCORESIGHT_PPOCR_THREADS to change it (0: onnxruntime's default, every
core).
"""

import importlib.util
import json
import math
import os
import threading
import time
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def resource_path(*parts):
    return str(Path(os.environ.get("SCORESIGHT_RESOURCES", Path(__file__).parent)) / Path(*parts))

PREFIX = "ppocr:"
AUTO = "ppocr:auto"
SMALL = "ppocr:small"
TINY = "ppocr:tiny"

ONNX_FILE = "inference.onnx"
CHARACTERS_FILE = "characters.txt"
CONFIG_FILE = "config.json"

THREADS_ENV = "SCORESIGHT_PPOCR_THREADS"
INTRA_OP_THREADS = 2
# A frame with more boxes than this is read in several runs.
MAX_BATCH = 32
# PaddleOCR's recognition input when config.json doesn't say otherwise
DEFAULT_IMAGE_HEIGHT = 48
DEFAULT_MIN_WIDTH = 320

FETCH_COMMAND = "python scripts/fetch_models.py"


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    folder: str  # in ocr_models/ppocr/
    description: str


MODELS = {
    SMALL: ModelInfo(SMALL, "PP-OCRv6_small_rec", "PP-OCRv6 small"),
    TINY: ModelInfo(TINY, "PP-OCRv6_tiny_rec", "PP-OCRv6 tiny"),
}

# Recognizer states
NOT_LOADED = "not loaded"
LOADING = "loading"
READY = "ready"
FAILED = "failed"


def is_ppocr(model) -> bool:
    """Whether a stored ocr_model value is a PP-OCR choice (a model or AUTO)."""
    return isinstance(model, str) and (model in MODELS or model == AUTO)


def models_of(choice: str) -> tuple[str, ...]:
    """The models a PP-OCR choice uses: AUTO uses both."""
    return (SMALL, TINY) if choice == AUTO else (choice,)


def model_files(model_id: str) -> tuple[str, str, str]:
    """The (inference.onnx, characters.txt, config.json) paths of a model."""
    folder = MODELS[model_id].folder
    return (
        resource_path("ocr_models", "ppocr", folder, "inference.onnx"),
        resource_path("ocr_models", "ppocr", folder, "characters.txt"),
        resource_path("ocr_models", "ppocr", folder, "config.json"),
    )


def models_present(model_id: str) -> bool:
    return all(os.path.isfile(path) for path in model_files(model_id))


_runtime_installed: bool | None = None


def runtime_installed() -> bool:
    """Whether onnxruntime can be found (without importing it)."""
    global _runtime_installed
    if _runtime_installed is None:
        try:
            _runtime_installed = importlib.util.find_spec("onnxruntime") is not None
        except (ImportError, ValueError):
            _runtime_installed = False
    return _runtime_installed


MISSING_MODELS = "models"
MISSING_RUNTIME = "runtime"


def missing(model_id: str) -> str | None:
    """What a PP-OCR model lacks: MISSING_MODELS (its files: from source, run
    scripts/fetch_models.py), MISSING_RUNTIME (onnxruntime) or None. Doesn't
    import onnxruntime."""
    if not is_ppocr(model_id) or not all(
        models_present(model) for model in models_of(model_id)
    ):
        return MISSING_MODELS
    if not runtime_installed():
        return MISSING_RUNTIME
    return None


def unavailable_reason(model_id: str) -> str | None:
    """Why a PP-OCR model can't be used, or None if its files and onnxruntime
    are there (for the log). Doesn't import onnxruntime."""
    if not is_ppocr(model_id):
        return f"unknown PP-OCR model {model_id!r}"
    lacking = missing(model_id)
    if lacking == MISSING_MODELS:
        return f"the model files are missing (run {FETCH_COMMAND})"
    if lacking == MISSING_RUNTIME:
        return "onnxruntime is not installed"
    return None


def available(model_id: str) -> bool:
    return unavailable_reason(model_id) is None


# --- preprocessing and decoding (numpy only, so they can be tested without models)


def padded_width(height: int, width: int, image_height: int, min_width: int) -> int:
    """The input width PaddleOCR gives a crop of this size on its own: the
    resized width, at least min_width."""
    ratio = width / max(1, height)
    return max(min_width, int(image_height * ratio))


def resize_normalize(
    image: np.ndarray, image_height: int, target_width: int
) -> np.ndarray:
    """PaddleOCR's resize_norm_img: (3, image_height, target_width) float32,
    the crop resized to image_height keeping its aspect ratio (at most
    target_width wide), scaled to [-1, 1] and padded with zeros on the right."""
    height, width = image.shape[:2]
    ratio = width / max(1, height)
    resized_width = max(1, min(target_width, math.ceil(image_height * ratio)))
    resized = cv2.resize(image, (resized_width, image_height))
    resized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    padded = np.zeros((3, image_height, target_width), dtype=np.float32)
    padded[:, :, :resized_width] = resized
    return padded


def as_probabilities(output: np.ndarray) -> np.ndarray:
    """The exported models end with a softmax; apply one if a model doesn't."""
    sums = output.sum(axis=-1)
    if output.min() >= 0.0 and np.allclose(sums, 1.0, atol=1e-3):
        return output
    shifted = output - output.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def ctc_greedy(
    probs: np.ndarray, labels: list[str]
) -> tuple[str, float | None, np.ndarray]:
    """Greedy CTC decoding of one (time, classes) probability matrix whose
    class 0 is the blank: the best class per step, repeats merged, blanks
    dropped. Returns the text, the mean probability of the kept steps (None if
    nothing is kept) and the kept class indices."""
    best = probs.argmax(axis=1)
    keep = best != 0
    keep[1:] &= best[1:] != best[:-1]
    kept = best[keep]
    text = "".join(labels[i] for i in kept)
    if kept.size == 0:
        return text, None, kept
    confidence = float(probs[np.flatnonzero(keep), kept].mean())
    return text, confidence, kept


@dataclass
class ClassGroups:
    """The classes a box's whitelist allows, grouped by what they read as."""

    labels: list[str]  # "" (blank) then the output string of each group
    columns: np.ndarray  # class indices, group after group
    starts: np.ndarray  # where each group starts in columns

    def reduce(self, probs: np.ndarray) -> np.ndarray:
        """(time, classes) -> (time, 1 + groups): the blank, then the summed
        probability of each group's classes."""
        grouped = np.add.reduceat(probs[:, self.columns], self.starts, axis=1)
        return np.concatenate([probs[:, :1], grouped], axis=1)


def class_groups(characters: list[str], allowed: str) -> ClassGroups:
    """Group the model's classes (characters[0] is the blank) whose NFKC form
    is made only of allowed characters by that form. Everything else can't be
    output."""
    groups: dict[str, list[int]] = {}
    for index, char in enumerate(characters):
        if index == 0:
            continue
        form = unicodedata.normalize("NFKC", char)
        if form and all(c in allowed for c in form):
            groups.setdefault(form, []).append(index)
    labels = [""]
    columns: list[int] = []
    starts: list[int] = []
    for form, indices in groups.items():
        labels.append(form)
        starts.append(len(columns))
        columns.extend(indices)
    if not columns:
        raise ValueError(f"the model has none of the characters {allowed!r}")
    return ClassGroups(labels, np.array(columns), np.array(starts))


def read_characters(path: str) -> list[str]:
    """The classes of a model from its characters.txt (one character per line,
    UTF-8): the blank, the characters and a space (CTCLabelDecode with
    use_space_char)."""
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not all(lines):
        raise ValueError(f"{path}: not a character list")
    return ["", *lines, " "]


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: not a model configuration")
    return config


def to_model_input(crop: np.ndarray, rgb: bool) -> np.ndarray:
    """A 3-channel uint8 image in the model's channel order from a crop of the
    frame (BGR, BGRA or gray)."""
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    if rgb:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


def intra_op_threads() -> int:
    value = os.environ.get(THREADS_ENV, "").strip()
    if not value:
        return INTRA_OP_THREADS
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning("Ignoring %s=%r: not a whole number", THREADS_ENV, value)
        return INTRA_OP_THREADS


# --- the recognizer -----------------------------------------------------------


class Recognizer:
    """One PP-OCR model: loaded once (lazily, on a background thread), then
    used from any thread."""

    def __init__(
        self, model_id: str, threads: int | None = None, spinning: bool = False
    ):
        self.model_id = model_id
        self.info = MODELS[model_id]
        self.threads = intra_op_threads() if threads is None else threads
        # onnxruntime's idle pool threads busy-wait for work (its default) or
        # sleep (here): with two threads it barely changes a read's speed, and
        # it doesn't burn CPU between frames
        self.spinning = spinning
        self.state = NOT_LOADED
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.runtime_version: str | None = None
        # set once the load finished, whether it worked or not
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._session = None
        self._input_name = None
        self._characters: list[str] = []
        self._image_height = DEFAULT_IMAGE_HEIGHT
        self._min_width = DEFAULT_MIN_WIDTH
        self._rgb = False
        self._groups: dict[str, ClassGroups] = {}

    def __repr__(self):
        return f"<Recognizer {self.model_id} {self.state}>"

    # The interface TextDetector uses for a PP-OCR choice, shared with
    # AutoChooser: the recognizer to read with now, and how long it took.

    def active(self) -> "Recognizer":
        return self

    def record(self, model_id: str, boxes: int, seconds: float, interval_ms):
        pass

    @property
    def ready(self) -> bool:
        return self.state == READY

    def start_loading(self) -> None:
        """Load the model on a background thread, unless it is loaded,
        loading or failed to load (a failure is final for the session: its
        cause, a missing or broken onnxruntime or model, doesn't go away)."""
        with self._lock:
            if self.state != NOT_LOADED:
                return
            self.state = LOADING
        threading.Thread(
            target=self._load_quietly,
            name=f"ppocr-load-{self.info.folder}",
            daemon=True,
        ).start()

    def load(self) -> bool:
        """Load the model on this thread (if it isn't loaded or loading) and
        return whether it is ready."""
        with self._lock:
            start = self.state == NOT_LOADED
            if start:
                self.state = LOADING
        if start:
            self._load_quietly()
        return self.wait()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a load in progress (up to timeout seconds); return whether
        the model is ready. False at once if no load was started."""
        if self.state == NOT_LOADED:
            return False
        self._done.wait(timeout)
        return self.state == READY

    def _load_quietly(self):
        started = time.perf_counter()
        try:
            self._load()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            logger.exception("PP-OCR: can't load %s", self.info.folder)
            with self._lock:
                self.error = message
                self.state = FAILED
        else:
            self.load_seconds = time.perf_counter() - started
            with self._lock:
                self.state = READY
            logger.info(
                "PP-OCR: loaded %s (onnxruntime %s, %s intra-op threads) in %.0f ms",
                self.info.folder,
                self.runtime_version,
                self.threads or "default",
                self.load_seconds * 1000,
            )
        finally:
            self._done.set()

    def _load(self):
        reason = unavailable_reason(self.model_id)
        if reason is not None:
            raise RuntimeError(reason)
        onnx_path, characters_path, config_path = model_files(self.model_id)
        config = read_config(config_path)
        characters = read_characters(characters_path)
        image_height = int(config.get("image_height", DEFAULT_IMAGE_HEIGHT))
        min_width = int(config.get("min_width", DEFAULT_MIN_WIDTH))
        rgb = str(config.get("channel_order", "BGR")).upper() == "RGB"

        import onnxruntime

        options = onnxruntime.SessionOptions()
        # the exports have initializers onnxruntime warns about
        options.log_severity_level = 3
        if self.threads:
            options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        if not self.spinning:
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = onnxruntime.InferenceSession(
            onnx_path, options, providers=["CPUExecutionProvider"]
        )
        classes = session.get_outputs()[0].shape[-1]
        if isinstance(classes, int) and classes != len(characters):
            raise RuntimeError(
                f"{self.info.folder}: the model has {classes} classes but "
                f"{CHARACTERS_FILE} gives {len(characters)}"
            )
        with self._lock:
            self._session = session
            self._input_name = session.get_inputs()[0].name
            self._characters = characters
            self._image_height = image_height
            self._min_width = min_width
            self._rgb = rgb
            self._groups = {}
        self.runtime_version = getattr(onnxruntime, "__version__", "?")

    def close(self):
        with self._lock:
            self._session = None
            self.state = NOT_LOADED
            self._done.clear()

    def _groups_for(self, characters: list[str], allowed: str) -> ClassGroups:
        with self._lock:
            groups = self._groups.get(allowed)
        if groups is None:
            groups = class_groups(characters, allowed)
            with self._lock:
                self._groups[allowed] = groups
        return groups

    def decode(
        self, probs: np.ndarray, characters: list[str], allowed: str | None
    ) -> tuple[str, float | None]:
        if not allowed:
            text, confidence, _ = ctc_greedy(probs, characters)
            return text, confidence
        groups = self._groups_for(characters, allowed)
        text, confidence, _ = ctc_greedy(groups.reduce(probs), groups.labels)
        return text, confidence

    def recognize(
        self, crops: list[np.ndarray], whitelists: list[str | None] | None = None
    ) -> list[tuple[str, float | None]]:
        """Read every crop (BGR, BGRA or gray) and return (text, confidence
        0-1 or None when nothing was read) for each, in order. whitelists[i]
        limits crop i's characters (None: any). Crops of the same padded width
        go through the model in one run. Raises RuntimeError if the model
        isn't loaded."""
        with self._lock:
            session = self._session
            input_name = self._input_name
            characters = self._characters
            image_height = self._image_height
            min_width = self._min_width
            rgb = self._rgb
        if session is None:
            raise RuntimeError(f"PP-OCR model {self.model_id} is not loaded")
        if whitelists is None:
            whitelists = [None] * len(crops)
        results: list[tuple[str, float | None]] = [("", None)] * len(crops)
        by_width: dict[int, list[int]] = {}
        images = {}
        for index, crop in enumerate(crops):
            if crop is None or crop.ndim < 2 or crop.shape[0] < 1 or crop.shape[1] < 1:
                continue
            image = to_model_input(crop, rgb)
            images[index] = image
            width = padded_width(
                image.shape[0], image.shape[1], image_height, min_width
            )
            by_width.setdefault(width, []).append(index)
        for width in sorted(by_width):
            indices = by_width[width]
            for start in range(0, len(indices), MAX_BATCH):
                chunk = indices[start : start + MAX_BATCH]
                batch = np.stack(
                    [resize_normalize(images[i], image_height, width) for i in chunk]
                )
                [output] = session.run(None, {input_name: batch})
                probs = as_probabilities(np.asarray(output, dtype=np.float32))
                if probs.shape[-1] != len(characters):
                    raise RuntimeError(
                        f"{probs.shape[-1]} classes for {len(characters)} characters"
                    )
                for row, index in enumerate(chunk):
                    results[index] = self.decode(
                        probs[row], characters, whitelists[index]
                    )
        return results


_recognizers: dict[str, Recognizer] = {}
_recognizers_lock = threading.Lock()


def get_recognizer(model_id: str) -> Recognizer:
    """The process's Recognizer for a model (created on first use, not loaded:
    call start_loading or load)."""
    with _recognizers_lock:
        recognizer = _recognizers.get(model_id)
        if recognizer is None:
            recognizer = _recognizers[model_id] = Recognizer(model_id)
        return recognizer


# --- PP-OCR (automatic) -------------------------------------------------------

# Accurate (small) is too slow when a frame's batch takes more than this share
# of the time between detections (1000 / Detections per second) ...
AUTO_SLOW_SHARE = 0.7
# ... for this long in a row: automatic then switches to fast (tiny).
AUTO_SLOW_SECONDS = 3.0
# Back to accurate when its measured cost per box, times the boxes read now,
# would take less than this share of the time between detections ...
AUTO_FIT_SHARE = 0.5
# ... for this long in a row (and at least that long after the switch).
AUTO_FIT_SECONDS = 10.0
# The measured times are smoothed (exponential moving average with this weight).
AUTO_SMOOTHING = 0.3

# why AutoChooser reads with the model it reads with
AUTO_STARTING = "starting"  # accurate, not measured yet
AUTO_KEEPS_UP = "keeps up"  # accurate keeps up
AUTO_TOO_SLOW = "too slow"  # fast: accurate couldn't keep up
AUTO_UNAVAILABLE = "unavailable"  # fast: accurate didn't load


class AutoChooser:
    """PP-OCR (automatic): reads with accurate (small) while a frame's boxes
    take it less than AUTO_SLOW_SHARE of the time between detections, and
    with fast (tiny) once it has needed more for AUTO_SLOW_SECONDS. It goes
    back to accurate when, from accurate's measured cost per box, the boxes
    would take it less than AUTO_FIT_SHARE of that time for AUTO_FIT_SECONDS
    (e.g. after Detections / s was lowered or boxes were removed). Each
    switch is logged, and status() says which model it uses and why, for the
    window. Both models load in the background."""

    model_id = AUTO

    def __init__(self, clock=time.monotonic):
        self.accurate = get_recognizer(SMALL)
        self.fast = get_recognizer(TINY)
        self._clock = clock
        self._lock = threading.Lock()
        self.current = SMALL
        self.reason = AUTO_STARTING
        # accurate's smoothed time per frame and per box (ms), as last
        # measured (per frame: or estimated for the boxes read now), fast's,
        # and the time between detections they were measured against
        self.accurate_ms: float | None = None
        self.accurate_ms_per_box: float | None = None
        self.fast_ms: float | None = None
        self.interval_ms: float | None = None
        self._slow_since: float | None = None
        self._fit_since: float | None = None
        self._switched_at: float | None = None

    def __repr__(self):
        return f"<AutoChooser {self.current} ({self.reason})>"

    def start_loading(self):
        self.accurate.start_loading()
        self.fast.start_loading()

    def _usable(self) -> Recognizer:
        """The recognizer to read with: the current one, or the other one if
        the current one failed to load and the other didn't (for good: a
        failure is final)."""
        if (
            self.current == SMALL
            and self.accurate.state == FAILED
            and self.fast.state != FAILED
        ):
            with self._lock:
                self.current = TINY
                self.reason = AUTO_UNAVAILABLE
        elif (
            self.current == TINY
            and self.fast.state == FAILED
            and self.accurate.state != FAILED
        ):
            # better slow than Tesseract
            with self._lock:
                self.current = SMALL
                self.reason = AUTO_KEEPS_UP
        return self.accurate if self.current == SMALL else self.fast

    def active(self) -> Recognizer:
        return self._usable()

    @property
    def state(self) -> str:
        return self._usable().state

    @property
    def error(self) -> str | None:
        return self._usable().error or self.accurate.error

    def wait(self, timeout: float | None = None) -> bool:
        self.accurate.wait(timeout)
        self.fast.wait(timeout)
        return self._usable().state == READY

    def status(self) -> tuple[str, str, float | None]:
        """(the model it reads with, why: an AUTO_* reason, accurate's time
        per frame in ms: measured, or estimated for the boxes read now)."""
        self._usable()
        with self._lock:
            return self.current, self.reason, self.accurate_ms

    def record(self, model_id: str, boxes: int, seconds: float, interval_ms):
        """A frame's `boxes` boxes took model_id `seconds` to read, with
        `interval_ms` between detections (None: unknown; nothing changes)."""
        if boxes <= 0 or not interval_ms or interval_ms <= 0:
            return
        now = self._clock()
        ms = seconds * 1000.0
        switch = None
        with self._lock:
            self.interval_ms = float(interval_ms)
            if model_id == SMALL:
                self.accurate_ms = _smooth(self.accurate_ms, ms)
                self.accurate_ms_per_box = _smooth(self.accurate_ms_per_box, ms / boxes)
            elif model_id == TINY:
                self.fast_ms = _smooth(self.fast_ms, ms)
                if self.accurate_ms_per_box is not None:
                    # what accurate would take for the boxes read now
                    self.accurate_ms = self.accurate_ms_per_box * boxes
            if self.current == SMALL and model_id == SMALL:
                if self.accurate_ms > AUTO_SLOW_SHARE * interval_ms:
                    if self._slow_since is None:
                        self._slow_since = now
                    if (
                        now - self._slow_since >= AUTO_SLOW_SECONDS
                        and self.fast.state != FAILED
                    ):
                        switch = TINY
                else:
                    self._slow_since = None
                    self.reason = AUTO_KEEPS_UP
            elif (
                self.current == TINY
                and model_id == TINY
                and self.reason == AUTO_TOO_SLOW
                and self.accurate_ms is not None
            ):
                if self.accurate_ms < AUTO_FIT_SHARE * interval_ms:
                    if self._fit_since is None:
                        self._fit_since = now
                    if (
                        now - self._fit_since >= AUTO_FIT_SECONDS
                        and now - self._switched_at >= AUTO_FIT_SECONDS
                    ):
                        switch = SMALL
                else:
                    self._fit_since = None
            if switch is not None:
                self.current = switch
                self.reason = AUTO_TOO_SLOW if switch == TINY else AUTO_KEEPS_UP
                self._switched_at = now
                self._slow_since = None
                self._fit_since = None
            accurate_ms = self.accurate_ms
        if switch == TINY:
            logger.info(
                "PP-OCR automatic: reading with fast: accurate needs %.0f ms per "
                "reading, more than %.0f%% of the %.0f ms between readings",
                accurate_ms,
                AUTO_SLOW_SHARE * 100,
                interval_ms,
            )
            self.fast.start_loading()
        elif switch == SMALL:
            logger.info(
                "PP-OCR automatic: reading with accurate again: it would need "
                "%.0f ms per reading, of the %.0f ms between readings",
                accurate_ms,
                interval_ms,
            )


def _smooth(previous: float | None, value: float) -> float:
    if previous is None:
        return value
    return previous + AUTO_SMOOTHING * (value - previous)


def create_choice(model: str):
    """What reads a PP-OCR choice: the model's Recognizer, or an AutoChooser
    for AUTO."""
    if model == AUTO:
        return AutoChooser()
    return get_recognizer(model)
