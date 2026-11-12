# Candidate connection release: a2bb457

Deployed to `scoreboard-ocr-00007-9n7` in `grounded-jetty-485516-r2` (us-central1), CPU only.
Cloud Build: `52f54d25-fa66-477b-9a28-5a52ceed454b`.
Image digest: `sha256:37a9cc8c98258ffc0abdd8c64b9da855bcde6a21148ea0dff72073e5124fe13e`.

## Minnesota–Montreal test game

NHL game `2026020287`; source generation `1794438081965608`, MD5 `gsd3zOQ0F8v3u3Gn9p7feA==`.
The original full CPU run was reused, replacing source-video windows 4770–4990 and 5995–6265 seconds with fresh automatic CPU extraction. Both recoveries retain original normalized frame PTS and source identity. Their runtimes were 148.2 and 170.1 seconds. This is not a fresh full-game benchmark.

| Records | Previous connections | Current connections | Strict intervals | Review only | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Events | 255/328 | 323/328 (98.5%) | 259 | 64 | 5 |
| Shift records | 565/647 | 643/647 (99.4%) | 575 | 68 | 4 |

There are 63 ambiguous event records and 585 ambiguous shift records; connection coverage does not measure timestamp accuracy. The five unresolved events are third-period IDs 816, 813, 819, 366, and 814.

The broadcast displays `2ND` after a completed second-period clock reset, then switches to `3RD` at the same countdown. Observations from video 5370.040667–5911.581667 are quarantined from strict alignment and exposed as possible third-period connections. The source label is preserved; this interpretation still needs review.

Published artifacts and source pointer:
`gs://statsyuk-game-videos/scoreboard/enriched/2026020287/1794438081965608/candidates-a2bb457/nhl-timeline.json`

The same prefix contains composite observations, comparison metrics, and each recovery's raw readings, layouts, and summary. Publication verified the downloaded result, source generation and MD5, and guarded the metadata change by generation and metageneration (5 → 6).

## Validation

- 55 scoreboard tests and 10 automatic-field tests passed.
- 9 explorer tests and the web TypeScript check passed.
- The deployed Eventarc worker processed a 15-second power-play fixture on CPU: 26/30 accepted samples, 10.87 seconds OCR wall time, period 3, remaining game clock 10:46–10:58. No accepted reading used the penalty timer.
- Dashboard changes: https://github.com/statsyukanalytics/statsyukreactv2/pull/232 (required review before merge). Review candidates never trigger automatic seeks or on-ice membership.

The full game's cloud extraction throughput has not been benchmarked by this release. The existing worker timeout still applies.
