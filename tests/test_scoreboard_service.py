import json
import unittest
from unittest.mock import patch
from google.api_core.exceptions import NotFound

from studio.scoreboard.service import app


class SourceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch('studio.scoreboard.service.storage.Client')
    def test_new_and_legacy_sources_use_exact_generation(self, storage):
        bucket = storage.return_value.bucket.return_value
        blob = bucket.blob.return_value
        blob.metadata = {'retriedby': 'nhl-clip', 'nhlgameid': '2026020292'}
        blob.name = 'game.mp4'
        blob.metageneration = 1
        blob.exists.return_value = True
        for prefix in ('scoreboard/sources/5100/', 'videos/5100/'):
            with self.subTest(prefix=prefix):
                bucket.reset_mock()
                blob.metadata = {'retriedby': 'nhl-clip', 'nhlgameid': '2026020292'}
                response = self.client.post('/', json={
                    'bucket': 'statsyuk-game-videos',
                    'name': prefix + 'game.mp4', 'generation': '123',
                })
                self.assertEqual(response.status_code, 204)
                self.assertEqual(bucket.blob.call_args_list[0].args, (prefix + 'game.mp4',))
                self.assertEqual(bucket.blob.call_args_list[0].kwargs, {'generation': 123})
                blob.patch.assert_called_with(if_metageneration_match=1)

    @patch('studio.scoreboard.service.storage.Client')
    def test_results_and_other_buckets_do_not_trigger_ocr(self, storage):
        for bucket, name in (
            ('statsyuk-game-videos', 'scoreboard/v2-auto/game/nhl-timeline.json'),
            ('other-bucket', 'scoreboard/sources/game.mp4'),
            ('statsyuk-game-videos', 'staging/game.mp4'),
        ):
            response = self.client.post('/', json={'bucket': bucket, 'name': name})
            self.assertEqual(response.status_code, 204)
        storage.assert_not_called()

    @patch('studio.scoreboard.service.storage.Client')
    def test_unmarked_video_is_ignored(self, storage):
        blob = storage.return_value.bucket.return_value.blob.return_value
        blob.metadata = {}
        response = self.client.post('/', json={
            'bucket': 'statsyuk-game-videos', 'name': 'videos/ordinary.mp4', 'generation': '123',
        })
        self.assertEqual(response.status_code, 204)
        blob.download_to_filename.assert_not_called()

    @patch('studio.scoreboard.service.storage.Client')
    def test_local_result_for_exact_source_is_preserved(self, storage):
        blob = storage.return_value.bucket.return_value.blob.return_value
        blob.metadata = {'retriedby': 'nhl-clip', 'nhlgameid': '2026020287',
                         'nhltimeline': 'gs://statsyuk-game-videos/scoreboard/local/result.json'}
        blob.exists.return_value = True
        blob.download_as_text.return_value = json.dumps({
            'schema_version': 2, 'nhl_game_id': 2026020287,
            'source': {'uri': 'gs://statsyuk-game-videos/videos/game.mp4', 'generation': '123'},
        })
        response = self.client.post('/', json={
            'bucket': 'statsyuk-game-videos', 'name': 'videos/game.mp4', 'generation': '123',
        })
        self.assertEqual(response.status_code, 204)
        blob.patch.assert_not_called()
        blob.download_to_filename.assert_not_called()

    @patch('studio.scoreboard.service.storage.Client')
    def test_deleted_source_delivery_is_acknowledged(self, storage):
        blob = storage.return_value.bucket.return_value.blob.return_value
        blob.reload.side_effect = NotFound('source deleted')
        response = self.client.post('/', json={'bucket':'statsyuk-game-videos',
            'name':'scoreboard/sources/test.mp4', 'generation':'123'})
        self.assertEqual(response.status_code,204)
        blob.download_to_filename.assert_not_called()


if __name__ == '__main__':
    unittest.main()
