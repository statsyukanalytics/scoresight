"""Download the PP-OCRv6 text recognition models of ScoreSight's PP-OCR engine.

    python scripts/fetch_models.py            # fetch what is missing, verify all
    python scripts/fetch_models.py --check    # verify only: exit 1 if a file is
                                              # missing or doesn't match
    python scripts/fetch_models.py --force    # download everything again

The weights aren't in git. This puts them in ocr_models/ppocr/<model>/
(gitignored), where the app (src/ppocr_engine.py) and the PyInstaller build
(scoresight.spec) find them:

    inference.onnx   the model, from its Hugging Face repository at a pinned
                     revision, checked against its pinned SHA-256
    characters.txt   the model's character list (PaddleOCR's CTCLabelDecode
                     character_dict), one per line, UTF-8; written from the
                     model's inference.yml (also pinned) so that the app doesn't
                     have to parse YAML (203 ms for small's 151 KB in pure
                     Python), and checked against its own pinned SHA-256
    config.json      the input the model expects (height, minimum width, channel
                     order) and where it came from

ocr_models/ppocr/LICENSE (Apache-2.0, PaddleOCR's) and NOTICE are committed.

Idempotent: files that are present and match are kept, so a second run (or a
restored CI cache) downloads nothing. Downloads are retried with a growing
delay, written to a .part file and moved into place only once their hash
matches, so an interrupted run never leaves a bad model behind. Any failure
exits with 1: the CI builds and tests fail closed. Set HF_ENDPOINT to use a
Hugging Face mirror. Needs PyYAML (requirements-dev.txt, requirements-build.txt)
only when it has to write characters.txt.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "ocr_models" / "ppocr"
DEFAULT_ENDPOINT = "https://huggingface.co"

ONNX_FILE = "inference.onnx"
CHARACTERS_FILE = "characters.txt"
CONFIG_FILE = "config.json"
CONFIG_FORMAT = 1

ATTEMPTS = 5
FIRST_RETRY_DELAY_S = 2.0
TIMEOUT_S = 60.0
CHUNK = 1 << 20


@dataclass(frozen=True)
class Model:
    folder: str  # under ocr_models/ppocr/
    repo: str  # Hugging Face repository
    revision: str  # commit
    onnx_sha256: str
    onnx_size: int
    yml_sha256: str
    characters: int  # entries in the character list
    characters_sha256: str  # of characters.txt as written here


# The revisions the OCR evaluation measured (ocr_eval/engines/ppocr.py, report
# ocr_eval/results/2026-10-11/REPORT.md).
MODELS = (
    Model(
        folder="PP-OCRv6_small_rec",
        repo="PaddlePaddle/PP-OCRv6_small_rec_onnx",
        revision="b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        onnx_sha256="5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
        onnx_size=21159378,
        yml_sha256="ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
        characters=18708,
        characters_sha256="b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    ),
    Model(
        folder="PP-OCRv6_tiny_rec",
        repo="PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        revision="2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
        onnx_sha256="9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
        onnx_size=4462639,
        yml_sha256="66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1",
        characters=6904,
        characters_sha256="c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd",
    ),
)


class FetchError(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_url(model: Model, name: str) -> str:
    endpoint = (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    return f"{endpoint}/{model.repo}/resolve/{model.revision}/{name}"


def download(url: str, destination: Path, sha256: str, attempts: int = ATTEMPTS):
    """Download url to destination (through destination.part), retrying, and
    check its SHA-256. Raises FetchError when every attempt failed."""
    partial = destination.with_name(destination.name + ".part")
    delay = FIRST_RETRY_DELAY_S
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "scoresight-fetch-models"}
            )
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                with open(partial, "wb") as f:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        digest.update(block)
                        f.write(block)
            if digest.hexdigest() != sha256:
                raise FetchError(
                    f"SHA-256 mismatch: got {digest.hexdigest()}, expected {sha256}"
                )
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError, FetchError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                break
            print(
                f"  attempt {attempt}/{attempts} failed ({error}); retrying in "
                f"{delay:.0f} s",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise FetchError(f"can't download {url}: {last_error}")


def read_character_dict(yml_path: Path) -> tuple[list[str], dict]:
    """(character_dict, input config) from a PaddleOCR inference.yml."""
    try:
        import yaml
    except ImportError as error:
        raise FetchError(
            "PyYAML is needed to read inference.yml: pip install -r "
            "requirements-dev.txt (or requirements-build.txt)"
        ) from error
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(yml_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=loader)
    post = config.get("PostProcess") or {}
    if post.get("name") != "CTCLabelDecode":
        raise FetchError(f"not a CTC model: PostProcess {post.get('name')!r}")
    characters = post.get("character_dict")
    if not isinstance(characters, list) or not characters:
        raise FetchError("inference.yml has no character_dict")
    for char in characters:
        if not isinstance(char, str) or not char or "\n" in char or "\r" in char:
            raise FetchError(f"can't store the character {char!r} on a line")
    shape, mode = [3, 48, 320], "BGR"
    for op in (config.get("PreProcess") or {}).get("transform_ops") or []:
        if not isinstance(op, dict):
            continue
        if "RecResizeImg" in op:
            shape = list(op["RecResizeImg"]["image_shape"])
        if "DecodeImage" in op:
            mode = (op["DecodeImage"] or {}).get("img_mode", mode)
    if len(shape) != 3 or shape[0] != 3:
        raise FetchError(f"unexpected input shape {shape}")
    model_name = (config.get("Global") or {}).get("model_name")
    return characters, {
        "model_name": model_name,
        "image_height": int(shape[1]),
        "min_width": int(shape[2]),
        "channel_order": str(mode).upper(),
    }


def characters_text(characters: list[str]) -> bytes:
    return "".join(f"{char}\n" for char in characters).encode("utf-8")


def expected_config(model: Model, input_config: dict) -> dict:
    return {
        "format": CONFIG_FORMAT,
        **input_config,
        "characters": model.characters,
        # PaddleOCR's CTCLabelDecode with use_space_char: the classes are the
        # blank, characters.txt and a space
        "classes": model.characters + 2,
        "source": f"https://huggingface.co/{model.repo}",
        "revision": model.revision,
        "onnx_sha256": model.onnx_sha256,
        "license": "Apache-2.0",
    }


def write_atomically(path: Path, data: bytes):
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, path)


def config_ok(model: Model, path: Path) -> bool:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(config, dict)
        and config.get("format") == CONFIG_FORMAT
        and config.get("revision") == model.revision
        and config.get("classes") == model.characters + 2
        and isinstance(config.get("image_height"), int)
        and isinstance(config.get("min_width"), int)
    )


def problems(model: Model, directory: Path) -> list[str]:
    """What is missing or wrong in a model's folder ([] if it is complete)."""
    found = []
    onnx = directory / ONNX_FILE
    if not onnx.is_file():
        found.append(f"{ONNX_FILE} is missing")
    elif onnx.stat().st_size != model.onnx_size or sha256_of(onnx) != model.onnx_sha256:
        found.append(f"{ONNX_FILE} doesn't match its SHA-256")
    characters = directory / CHARACTERS_FILE
    if not characters.is_file():
        found.append(f"{CHARACTERS_FILE} is missing")
    elif sha256_of(characters) != model.characters_sha256:
        found.append(f"{CHARACTERS_FILE} doesn't match its SHA-256")
    if not config_ok(model, directory / CONFIG_FILE):
        found.append(f"{CONFIG_FILE} is missing or out of date")
    return found


def fetch(model: Model, root: Path = MODELS_DIR, force: bool = False) -> bool:
    """Make the model's folder complete. Returns True if it downloaded
    anything; raises FetchError on failure."""
    directory = root / model.folder
    directory.mkdir(parents=True, exist_ok=True)
    if not force and not problems(model, directory):
        return False
    onnx = directory / ONNX_FILE
    if (
        force
        or not onnx.is_file()
        or onnx.stat().st_size != model.onnx_size
        or sha256_of(onnx) != model.onnx_sha256
    ):
        print(f"  downloading {model.repo}/{ONNX_FILE} ...", flush=True)
        download(file_url(model, ONNX_FILE), onnx, model.onnx_sha256)
    with tempfile.TemporaryDirectory(prefix="scoresight-fetch-") as tmp:
        yml = Path(tmp) / "inference.yml"
        print(f"  downloading {model.repo}/inference.yml ...", flush=True)
        download(file_url(model, "inference.yml"), yml, model.yml_sha256)
        characters, input_config = read_character_dict(yml)
    if len(characters) != model.characters:
        raise FetchError(
            f"{model.folder}: {len(characters)} characters, expected {model.characters}"
        )
    text = characters_text(characters)
    if hashlib.sha256(text).hexdigest() != model.characters_sha256:
        raise FetchError(f"{model.folder}: the character list doesn't match")
    write_atomically(directory / CHARACTERS_FILE, text)
    config = expected_config(model, input_config)
    write_atomically(
        directory / CONFIG_FILE,
        (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    remaining = problems(model, directory)
    if remaining:
        raise FetchError(f"{model.folder}: {'; '.join(remaining)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true", help="only verify the files, download nothing"
    )
    group.add_argument("--force", action="store_true", help="download everything again")
    parser.add_argument(
        "--dest",
        type=Path,
        default=MODELS_DIR,
        help="the ocr_models/ppocr folder (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    failed = False
    for model in MODELS:
        directory = args.dest / model.folder
        if args.check:
            found = problems(model, directory)
            if found:
                failed = True
                print(f"{model.folder}: {'; '.join(found)}")
            else:
                print(f"{model.folder}: ok")
            continue
        print(f"{model.folder} ({model.repo}@{model.revision[:8]})", flush=True)
        try:
            downloaded = fetch(model, args.dest, force=args.force)
        except (FetchError, OSError) as error:
            failed = True
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            continue
        print("  downloaded and verified" if downloaded else "  present and verified")
    if failed:
        print(
            "Some models are missing or don't match"
            + (": run python scripts/fetch_models.py" if args.check else ""),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
