"""Resolve a stable replay ID on the worker that reads the media."""
import re
import json

from yt_dlp import YoutubeDL


def _resolve(video_id, progressive):
    if not re.fullmatch(r"[0-9]{1,20}", video_id):
        raise ValueError("Invalid archive video ID")
    selector = "b[protocol=https]" if progressive else "b[height=720][protocol^=m3u8]/b[protocol=https]"
    with YoutubeDL({"quiet": True, "noplaylist": True, "format": selector, "socket_timeout": 20}) as ydl:
        return ydl.extract_info("https://ok.ru/video/" + video_id, download=False)


def resolve_replay(video_id, progressive=False):
    info = _resolve(video_id, progressive)
    return info["url"], info.get("http_headers", {}), float(info["duration"])


def ocr_source(video_id, stack):
    from .replay_buffer import buffered_playlist

    info = _resolve(video_id, False)
    url = info["url"]
    headers = info.get("http_headers", {})
    if info.get("protocol", "").startswith("m3u8"):
        local = stack.enter_context(buffered_playlist(url, headers))
        if local:
            print(json.dumps({"stage": "replay", "mode": "buffered_hls"}), flush=True)
            return local, []
    print(json.dumps({"stage": "replay", "mode": "direct"}), flush=True)
    return url, ffmpeg_headers(headers)


def ffmpeg_headers(headers):
    return ["-headers", "".join(f"{key}: {value}\r\n" for key, value in headers.items())]
