"""Resolve a stable replay ID on the worker that reads the media."""
import re

from yt_dlp import YoutubeDL


def resolve_replay(video_id, progressive=False):
    if not re.fullmatch(r"[0-9]{1,20}", video_id):
        raise ValueError("Invalid archive video ID")
    selector = "b[protocol=https]" if progressive else "b[height=720][protocol^=m3u8]/b[protocol=https]"
    with YoutubeDL({"quiet": True, "noplaylist": True, "format": selector, "socket_timeout": 20}) as ydl:
        info = ydl.extract_info("https://ok.ru/video/" + video_id, download=False)
    return info["url"], info.get("http_headers", {}), float(info["duration"])


def ffmpeg_headers(headers):
    return ["-headers", "".join(f"{key}: {value}\r\n" for key, value in headers.items())]
