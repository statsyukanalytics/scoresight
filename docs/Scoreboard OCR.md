# Scoreboard OCR and NHL timing

The pipeline reads the broadcast clock and period with the existing ScoreSight PP-OCR recognizer, keeps decoded frame timestamps, and joins them to CapAngry's `expanded_play_by_play` and `player_shifts` tables. It keeps every fetched record, including events that do not appear in the uploaded clip.

## Local run

Install `requirements-scoreboard.txt` and put FFmpeg and FFprobe on PATH. Download the pinned models with `python -m studio.scoreboard.fetch_models`. Set `NHL_DATABASE_URL` to the NHL database connection string through your environment. Queries run in a read-only transaction.

```powershell
python -m studio.scoreboard --video game.mp4 --config studio/scoreboard/profiles/tsn-2026-720p.json --game 2026020292 --output clips/game/scoreboard
```

Use a new output directory for each run. The profile must match both the broadcast layout and video resolution. Existing profiles are copied from `nhl-tracking-data/configs/scoreboards`. Different layouts need measured clock and period rectangles. No automatic broadcast-layout detector is claimed.

Outputs include raw OCR, `observations.csv`, a runtime summary, and `nhl-timeline.json`. Each event and shift retains its original database record and gains candidate video intervals. Missing or low-confidence clocks break continuity. No interpolation crosses an unreadable sample or a large clock jump. Stopped clocks and repeated footage can match multiple intervals. These are clock candidates requiring review, not verified live-play timestamps or frame-exact event locations. Shootouts are left unmatched.

## Clip uploads

The companion app changes are in `statsyukanalytics/statsyukreactv2` (PR #225): `NhlClipForm.tsx`, the clips API route, and `packages/domain/src/clips/nhl-clip.ts`. The form asks for a scoreboard layout and includes `scoreboardprofile` in the uploaded object's metadata, alongside the existing NHL game ID.

`studio.scoreboard.service` receives a Google Cloud Storage finalized event. It processes only objects in `videos/` with the NHL clip marker and a supported scoreboard profile. It reads the exact object generation, runs OCR and alignment, and saves results under `scoreboard/<nhl-game>/<generation>/<source-name>/`. It sets `nhltimeline` on the source object's metadata to the result's GCS URI. A completed result makes delivery retries skip processing. It does not start another tracking run or change NHL database records.

The worker image is defined in `scoreboard.Dockerfile`. Deploy it to `grounded-jetty-485516-r2` as an authenticated Cloud Run service, with concurrency 1, maximum instances 1, a 3600-second request timeout, and adequate memory for the source video (temporary storage uses instance memory). Use an Eventarc `google.cloud.storage.object.v1.finalized` trigger for `statsyuk-game-videos`. Grant its trigger identity Cloud Run invoker; the worker identity needs source-object read/update, result-object create/read, and access to the secret containing `NHL_DATABASE_URL`. Do not enable unauthenticated invocation. Deploy the companion app changes as well.

The worker is deployed as `scoreboard-ocr` in `grounded-jetty-485516-r2/us-central1`; Eventarc trigger `scoreboard-ocr-upload` receives finalized uploads from the video bucket. The service is private, uses the dedicated `scoreboard-ocr` service account, and reads secret version `nhl-scoreboard-database-url:1`. Its bucket role allows object get/create/update only under `videos/` and `scoreboard/`. The trigger subscription has a 600-second acknowledgement deadline. The companion app integration is [PR #225](https://github.com/statsyukanalytics/statsyukreactv2/pull/225), on the isolated `codex/scoreboard-ocr` branch. Production activation of its selector is awaiting the repository-required approving review. The worker image is `us-central1-docker.pkg.dev/grounded-jetty-485516-r2/statsyuk/scoreboard-ocr:20261111-retry` (Cloud Build `d2a40afc-81cf-4f28-b8d3-4b653a2f0e26`). Results are created with generation preconditions so partial retries do not require object overwrite/delete permission. The current app does not yet display the resulting timeline; the source object's `nhltimeline` metadata points to it.

## Provenance and checks

The recognizer and model-fetch script come from ScoreSight commit `90b9d92be9afd45eeee916219973dab621d4f5ab`; its license is preserved in `SCORESIGHT-LICENSE`, and model license/notice are under `ocr_models/ppocr/`. The recognizer uses standard logging and configurable resource paths instead of desktop-app logging. The extraction script is ported from `nhl-tracking-data/scripts/ocr/extract_clock.py` and uses ScoreSight's CPU runtime rather than requiring CUDA.

`python -m unittest discover -s tests -p test_scoreboard.py -v` checks missing readings, clock stoppages, repeated footage, regular/playoff overtime, shift boundaries, and low-confidence parsing. A three-second video fixture produced six valid OCR observations with frame timestamps. A read-only database smoke check loaded 318 events and 740 shifts for game 2026020292. Cloud verification on 2026-11-11 also passed: a synthetic 720p upload automatically produced six valid observations, five clock intervals, 318 events and 740 shifts. The fixture matched two start-of-game events and twelve shifts; these verify the join mechanism, not real broadcast accuracy. The generated fixture and result objects were removed after verification. Domain and web TypeScript checks passed.

## Repository ownership

This fork owns the Statsyuk OCR worker, profiles, extraction, NHL alignment, tests, and deployment configuration. They were migrated from clip-studio commits `3b6bb69` and `7f12756` without changing runtime behavior. The `studio.scoreboard` package name is retained for compatibility with the deployed worker. Clip-studio contains no OCR runtime. The web app only supplies upload metadata; the bucket event invokes this worker independently.

The headless recognizer is pinned to the upstream revision documented above; upstream desktop code remains available separately. Updating the upstream desktop code does not automatically update the pinned headless copy. Model weights are fetched during image build and are not committed. Credentials remain in Secret Manager.

Build from this repository root with `docker build -f scoreboard.Dockerfile -t scoreboard-ocr .`. The existing deployment continues to run its verified image; this source relocation does not require a redeployment.
