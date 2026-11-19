"""Read ahead a bounded window of archive HLS segments for one OCR job."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


WINDOW = 8
SEGMENT_LIMIT = 16 << 20


def _playlist(content, base):
    lines = content.splitlines()
    if not lines or lines[0] != "#EXTM3U" or "#EXT-X-ENDLIST" not in lines:
        return None
    segments = []
    for i, line in enumerate(lines):
        if line.startswith("#"):
            # Let FFmpeg handle keys, init fragments, byte ranges and other
            # playlist features which require more than a whole-segment fetch.
            if "URI=" in line or line.startswith(("#EXT-X-BYTERANGE", "#EXT-X-STREAM-INF")):
                return None
        elif line:
            url = urljoin(base, line)
            if urlsplit(url).scheme not in ("http", "https"):
                return None
            lines[i] = f"/{len(segments)}.ts"
            segments.append(url)
    if segments:
        return ("\n".join(lines) + "\n").encode(), segments


@contextmanager
def buffered_playlist(url, headers):
    def open_url(target):
        return urlopen(Request(target, headers=headers), timeout=20)

    try:
        with open_url(url) as response:
            content = response.read((2 << 20) + 1)
            parsed = _playlist(content.decode("utf-8-sig"), response.geturl()) if len(content) <= 2 << 20 else None
    except (OSError, HTTPException, UnicodeError):
        parsed = None
    if not parsed:
        yield None
        return

    playlist, segments = parsed
    lock = Lock()
    pending = {}
    requested = set()
    failures = set()

    def fetch(index):
        for attempt in range(3):
            try:
                with open_url(segments[index]) as upstream:
                    size = upstream.headers.get("Content-Length")
                    if size and int(size) > SEGMENT_LIMIT:
                        return None
                    data = upstream.read(SEGMENT_LIMIT + 1)
                    if len(data) > SEGMENT_LIMIT:
                        return None  # Stream unusually large segments directly.
                    if not data or (size and len(data) != int(size)):
                        raise OSError("Incomplete archive segment")
                    return data
            except (OSError, HTTPException):
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        def segment(index):
            with lock:
                # Retain the preceding segment for immediate reopens. Seeking
                # elsewhere cancels queued work outside the new window.
                for old in list(pending):
                    if not index - 1 <= old < index + WINDOW:
                        old_future = pending.pop(old)
                        if old_future not in requested:
                            old_future.cancel()
                for pos in range(index, min(index + WINDOW, len(segments))):
                    if pos not in pending:
                        pending[pos] = pool.submit(fetch, pos)
                future = pending[index]
                requested.add(future)
            try:
                return future.result()
            finally:
                with lock:
                    requested.discard(future)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/game.m3u8":
                    data = playlist
                else:
                    key = self.path.removeprefix("/").removesuffix(".ts")
                    if not key.isascii() or not key.isdigit() or len(key) > 8 or int(key) >= len(segments):
                        self.send_error(404)
                        return
                    index = int(key)
                    try:
                        data = segment(index)
                        if data is None:
                            with open_url(segments[index]) as upstream:
                                self.send_response(200)
                                if size := upstream.headers.get("Content-Length"):
                                    self.send_header("Content-Length", size)
                                self.end_headers()
                                while chunk := upstream.read(65536):
                                    try:
                                        self.wfile.write(chunk)
                                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                                        return
                            return
                    except (OSError, HTTPException):
                        failures.add(index)
                        self.send_error(502)
                        return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # FFprobe can finish before consuming its last request.

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}/game.m3u8"
            finally:
                server.shutdown()
                worker.join()
                with lock:
                    for future in pending.values():
                        future.cancel()
    if failures:
        raise OSError(f"Archive segments could not be read: {sorted(failures)}")
