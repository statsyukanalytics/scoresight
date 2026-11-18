import os
import unittest
from unittest.mock import patch

from studio.scoreboard.t4_batch import submit


class T4SubmissionTests(unittest.TestCase):
    @patch.dict(os.environ, {'OCR_WORKER_ACCOUNT': 'worker@example.test',
                             'OCR_WORKER_IMAGE': 'test-image', 'OCR_PROVISIONING_MODEL': 'STANDARD'})
    @patch('studio.scoreboard.t4_batch.AuthorizedSession')
    @patch('studio.scoreboard.t4_batch.google.auth.default', return_value=(object(), 'test-project'))
    def test_redelivery_reuses_job_and_replacement_gets_new_job(self, auth, session):
        post = session.return_value.__enter__.return_value.post
        post.return_value.status_code = 409
        event = {'name': 'scoreboard/sources/test.mp4', 'generation': '1', 'bucket': 'videos'}
        submit(event)
        submit(dict(event, extra='delivery metadata'))
        submit(dict(event, generation='2'))
        first, duplicate, replacement = [call.kwargs for call in post.call_args_list]
        self.assertEqual(first['params'], duplicate['params'])
        self.assertNotEqual(first['params'], replacement['params'])
        policy = first['json']['allocationPolicy']['instances'][0]['policy']
        self.assertEqual(policy['accelerators'], [{'type': 'nvidia-tesla-t4', 'count': 1}])
        self.assertEqual(policy['provisioningModel'], 'STANDARD')
        post.return_value.raise_for_status.assert_not_called()
