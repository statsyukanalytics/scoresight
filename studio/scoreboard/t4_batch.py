"""Submit and execute one ScoreSight GPU task per immutable video upload."""
import base64
import hashlib
import json
import os

import google.auth
from google.auth.transport.requests import AuthorizedSession


def submit(event):
    credentials, project = google.auth.default()
    region = os.environ.get('BATCH_REGION', 'us-central1')
    image = os.environ['OCR_WORKER_IMAGE']
    identity = os.environ['OCR_WORKER_ACCOUNT']
    source = {key: str(event[key]) for key in ('bucket', 'name', 'generation')}
    encoded = json.dumps(source, sort_keys=True)
    job_id = 'scoresight-gpu-' + hashlib.sha256(encoded.encode()).hexdigest()[:32]
    task = {
        'maxRunDuration': '5400s', 'maxRetryCount': 1,
        'lifecyclePolicies': [{'action': 'RETRY_TASK', 'actionCondition': {'exitCodes': [50001]}}],
        'runnables': [{'container': {'imageUri': image, 'entrypoint': 'python',
                                     'commands': ['-m', 'studio.scoreboard.t4_batch']}}],
        'environment': {'variables': {'OCR_SOURCE': encoded, 'SCORESIGHT_DEVICE': 'cuda',
                                      'OCR_DATABASE_SECRET': f'projects/{project}/secrets/nhl-scoreboard-database-url/versions/1'}},
    }
    policy = {'machineType': 'n1-standard-4', 'provisioningModel': os.environ.get('OCR_PROVISIONING_MODEL', 'STANDARD'),
              'accelerators': [{'type': 'nvidia-tesla-t4', 'count': 1}],
              'bootDisk': {'sizeGb': 50, 'type': 'pd-balanced'}}
    body = {'taskGroups': [{'taskCount': 1, 'taskSpec': task}],
            'allocationPolicy': {'instances': [{'policy': policy, 'installGpuDrivers': True}],
                                 'network': {'networkInterfaces': [{'noExternalIpAddress': True,
                                     'network': f'projects/{project}/global/networks/default',
                                     'subnetwork': f'projects/{project}/regions/{region}/subnetworks/default'}]},
                                 'serviceAccount': {'email': identity}},
            'logsPolicy': {'destination': 'CLOUD_LOGGING'}}
    url = f'https://batch.googleapis.com/v1/projects/{project}/locations/{region}/jobs'
    with AuthorizedSession(credentials) as session:
        response = session.post(url, params={'jobId': job_id}, json=body, timeout=30)
        if response.status_code != 409:
            response.raise_for_status()
    print(f'ScoreSight Batch job {job_id}', flush=True)


if __name__ == '__main__':
    from .service import process_event
    credentials, _ = google.auth.default()
    with AuthorizedSession(credentials) as session:
        secret = os.environ['OCR_DATABASE_SECRET']
        response = session.get(f'https://secretmanager.googleapis.com/v1/{secret}:access', timeout=20)
        response.raise_for_status()
        os.environ['NHL_DATABASE_URL'] = base64.b64decode(response.json()['payload']['data']).decode()
    message, status = process_event(json.loads(os.environ['OCR_SOURCE']), dispatch=False)
    if status != 204:
        raise RuntimeError(str(message))
