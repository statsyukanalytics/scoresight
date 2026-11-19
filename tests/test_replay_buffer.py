import io
import threading
import unittest
from collections import Counter
from urllib.request import urlopen
from unittest.mock import patch

from studio.scoreboard.replay_buffer import buffered_playlist


class Reply(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(len(data))}

    def geturl(self):
        return self.url


class ReplayBufferTests(unittest.TestCase):
    def test_fetches_ahead_and_reuses_segments_across_probe_and_decode(self):
        url = "https://example.test/archive/game.m3u8"
        manifest = b"#EXTM3U\n#EXTINF:6,\na.ts\n#EXTINF:6,\nb.ts\n#EXT-X-ENDLIST\n"
        fetched_second = threading.Event()
        requests = Counter()

        def archive(request, timeout):
            target = request.full_url
            self.assertEqual(request.get_header("Referer"), "archive")
            requests[target] += 1
            if target == url:
                return Reply(manifest, target)
            if target.endswith("/b.ts"):
                fetched_second.set()
            else:
                self.assertTrue(fetched_second.wait(3))
            return Reply(target[-4:].encode(), target)

        with patch("studio.scoreboard.replay_buffer.urlopen", side_effect=archive):
            with buffered_playlist(url, {"Referer": "archive"}) as source:
                with urlopen(source) as response:
                    self.assertIn(b"#EXTINF:6,\n/0.ts", response.read())
                for i, expected in [(0, b"a.ts"), (0, b"a.ts"), (1, b"b.ts")]:
                    with urlopen(source.replace("game.m3u8", f"{i}.ts")) as response:
                        self.assertEqual(response.read(), expected)
        self.assertEqual(sorted(requests.values()), [1, 1, 1])

    def test_complex_and_live_playlists_use_native_reader(self):
        for playlist in (b"#EXTM3U\na.ts", b"#EXTM3U\n#EXT-X-BYTERANGE:10@0\na.ts\n#EXT-X-ENDLIST"):
            with patch("studio.scoreboard.replay_buffer.urlopen", return_value=Reply(playlist, "https://example.test/game")):
                with buffered_playlist("https://example.test/game", {}) as source:
                    self.assertIsNone(source)
