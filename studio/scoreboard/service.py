"""Authenticated Cloud Run/Eventarc receiver for finalized NHL clip uploads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import time
from contextlib import ExitStack
from functools import lru_cache
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, Response, request
from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed, NotFound

app = Flask(__name__)
PLAYBACK_CHUNK_BYTES = 64 * 1024
# Cover the MP4 sample index and initial frames in one browser request.
PLAYBACK_RANGE_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=64)
def playback_source(video_id, time_bucket):
    from .linked_replay import resolve_replay
    return resolve_replay(video_id, progressive=True)


@app.get('/replay/<video_id>')
def replay_bytes(video_id):
    if not re.fullmatch(r'[0-9]{1,20}', video_id):
        return '', 404
    match = re.fullmatch(r'bytes=(\d{1,16})-(\d{0,16})', request.headers.get('Range', 'bytes=0-'))
    if not match:
        return '', 416
    start = int(match[1])
    range_limit = start + PLAYBACK_RANGE_BYTES - 1
    end = min(int(match[2]), range_limit) if match[2] else range_limit
    if end < start:
        return '', 416
    try:
        url, headers, _ = playback_source(video_id, int(time.time()) // 300)
        with ExitStack() as connection:
            upstream = connection.enter_context(urlopen(
                Request(url, headers={**headers, 'Range': f'bytes={start}-{end}'}),
                timeout=20,
            ))
            bounds = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', upstream.headers.get('Content-Range', ''))
            if upstream.status != 206 or not bounds:
                return 'Archive does not support this range', 502
            first_byte, last_byte, total = map(int, bounds.groups())
            if first_byte != start or not start <= last_byte <= end or last_byte >= total:
                return 'Archive does not support this range', 502
            length = last_byte - first_byte + 1
            first_chunk = upstream.read1(min(length, PLAYBACK_CHUNK_BYTES))
            if not first_chunk:
                return 'Incomplete archive response', 502

            def chunks():
                yield first_chunk
                remaining = length - len(first_chunk)
                while remaining:
                    chunk = upstream.read1(min(remaining, PLAYBACK_CHUNK_BYTES))
                    if not chunk:
                        raise OSError('Archive response ended before the advertised byte range')
                    remaining -= len(chunk)
                    yield chunk

            response = Response(chunks(), status=206, mimetype='video/mp4', headers={
                'Content-Range': bounds[0], 'Content-Length': str(length),
                'Accept-Ranges': 'bytes', 'Cache-Control': 'private, no-store',
            })
            response.call_on_close(connection.pop_all().close)
            return response
    except HTTPError as exc:
        exc.close()
        playback_source.cache_clear()
        return '', 416 if exc.code == 416 else 502
    except (URLError, TimeoutError):
        return 'Archive unavailable', 502


@app.post('/')
def finalized():
    return process_event(request.get_json(), dispatch=True)


def process_event(event, *, dispatch):
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
    try:
        blob.reload()
    except NotFound:
        return '', 204  # A deleted/cancelled exact-generation source cannot be processed.
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
    prefix = f'scoreboard/v2-integrated/{game_id}/{event["generation"]}/{blob.name.rsplit("/", 1)[-1]}'
    complete = bucket.blob(prefix + '/nhl-timeline.json')
    if complete.exists():
        blob.metadata = {**metadata, 'nhltimeline': f'gs://{bucket_name}/{complete.name}'}
        blob.patch(if_metageneration_match=blob.metageneration)
        return '', 204
    if dispatch:
        from .t4_batch import submit
        submit(event)
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
        env = os.environ.copy()
        if metadata.get('replayprovider') == 'okru':
            env['SCORESIGHT_LINKED_REPLAY'] = metadata['replayid']
        subprocess.run(command, check=True, env=env)
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
