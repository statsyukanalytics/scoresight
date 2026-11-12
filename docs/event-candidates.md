# Event connection evidence

`scoreboard-evidence-v2` keeps clock intervals separate from review candidates.
An event or shift with no strict interval can now expose `review_candidates`:

- `exact_reading`: an accepted observation, including an isolated reading.
- `nearby_clock`: an observed anchor with its game-clock error, at most ten seconds away. This is not an estimated event timestamp.
- `bracketed_clock`: the full video window between adjacent accepted readings. Maximum span: 120 video seconds and 60 game-clock seconds. Period changes, clock reversals, and implausible countdowns do not form brackets.
- `shift_clock_overlap`: accepted readings inside a half-open shift range.

`clock_status`, `interval_ids`, and `occurrences` retain strict interval evidence.
`status` includes review candidates; separated candidates remain ambiguous.
`connection_coverage` reports strict, review-only, and unresolved records separately.
Review candidates never add on-ice membership or supported live-play segments.

A period label that resets after an observed period end, persists through a sustained countdown, and joins the next period without another clock reset is flagged in `period_conflicts`. Those readings are quarantined from strict alignment. The next-period interpretation is offered for review with the original observed period retained. This may represent a stale broadcast label or edited footage; it is not a verified correction.

Automatic field discovery rejects distant clock/period pairs and prevents a competing timer from replacing working game-clock geometry. Conflicting readings remain unresolved.

For targeted CPU recovery, `studio.scoreboard.extract --start-seconds N --seconds D` samples a source window. Input seeking uses `-copyts -start_at_zero`; output observations retain the normalized original source PTS. Recovery artifacts must record the source identity, sampled windows, extraction hashes, and base run; do not silently merge readings from another video or replace the provenance of the original full run.
