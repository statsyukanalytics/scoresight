"""Authenticated Cloud Run/Eventarc receiver for finalized NHL clip uploads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from flask import Flask, request
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

app = Flask(__name__)


@app.post('/')
def finalized():
    event = request.get_json()
    bucket_name = os.environ.get('VIDEO_BUCKET', 'statsyuk-game-videos')
    if not isinstance(event, dict) or event.get('bucket') != bucket_name:
        return '', 204
    name = event.get('name', '')
    # New sources stay outside the tracking validator's videos/ prefix.
    # Legacy sources remain supported for existing OCR jobs.
    if not name.startswith(('scoreboard/sources/', 'videos/')):
        return '', 204
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(name, generation=int(event['generation']))
    blob.reload()
    metadata = blob.metadata or {}
    if metadata.get('retriedby') != 'nhl-clip':
        return '', 204
    # A completed local run can publish the same source generation while an older
    # cloud delivery is retrying. Do not replace that completed result.
    published = metadata.get('nhltimeline', '')
    if published.startswith(f'gs://{bucket_name}/scoreboard/'):
        result_blob = bucket.blob(published.removeprefix(f'gs://{bucket_name}/'))
        if result_blob.exists():
            result = json.loads(result_blob.download_as_text())
            source = result.get('source', {})
            if (result.get('schema_version') == 2
                    and source.get('uri') == f'gs://{bucket_name}/{name}'
                    and str(source.get('generation')) == str(event['generation'])
                    and str(result.get('nhl_game_id')) == metadata.get('nhlgameid')):
                return '', 204
    profile = metadata.get('scoreboardprofile') or 'auto'
    profiles = Path(__file__).parent / 'profiles'
    if profile != 'auto' and profile not in {p.stem for p in profiles.glob('*.json')}:
        return {'error': 'Unknown scoreboard profile'}, 400
    game_id = int(metadata['nhlgameid'])
    # Source generation identifies the exact input, even if a name is reused.
    prefix = f'scoreboard/v2-auto-r2/{game_id}/{event["generation"]}/{blob.name.rsplit("/", 1)[-1]}'
    complete = bucket.blob(prefix + '/nhl-timeline.json')
    if complete.exists():
        blob.metadata = {**metadata, 'nhltimeline': f'gs://{bucket_name}/{complete.name}'}
        blob.patch(if_metageneration_match=blob.metageneration)
        return '', 204
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps({'stage': 'download', 'nhl_game_id': game_id, 'source': name}), flush=True)
        folder = Path(tmp)
        video = folder / 'video'
        blob.download_to_filename(video)
        output = folder / 'output'
        command = [sys.executable, '-m', 'studio.scoreboard', '--video', str(video),
                   '--game', str(game_id), '--output', str(output)]
        if profile != 'auto':
            command += ['--config', str(profiles / (profile + '.json'))]
        subprocess.run(command, check=True, timeout=3300)
        timeline = json.loads((output / 'nhl-timeline.json').read_text(encoding='utf8'))
        timeline['source'] = {'uri': f'gs://{bucket_name}/{name}', 'generation': event['generation']}
        # Publish the timeline last: retries only skip fully published results.
        for filename in ('observations.csv', 'summary.json', 'layout-observations.json'):
            try:
                bucket.blob(prefix + '/' + filename).upload_from_filename(output / filename, if_generation_match=0)
            except PreconditionFailed:
                pass  # A previous delivery already published this immutable result.
        try:
            complete.upload_from_string(json.dumps(timeline, default=str), content_type='application/json', if_generation_match=0)
        except PreconditionFailed:
            pass
        blob.metadata = {**metadata, 'nhltimeline': f'gs://{bucket_name}/{complete.name}'}
        blob.patch(if_metageneration_match=blob.metageneration)
    return '', 204
