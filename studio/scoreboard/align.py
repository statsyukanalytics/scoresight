"""Conservative, bidirectional NHL clock alignment. No candidate is verified live."""
import math
from .candidates import add_candidates, conflict_at, period_conflicts


def seconds(value):
    minutes, value = str(value).split(':')
    return int(minutes) * 60 + float(value)


def _reading(row, lengths):
    try:
        if str(row.get('period_raw', '')).strip().upper() == 'SO':
            return None
        if str(row['valid']).lower() not in ('true', '1'):
            return None
        period = int(row['period'])
        length = lengths.get(period)
        t = float(row['video_seconds'])
        elapsed = length - float(row['clock_seconds']) if length is not None else -1
        if length is None or not math.isfinite(t) or not math.isfinite(elapsed) or t < 0 or not 0 <= elapsed <= length:
            return None
        return t, elapsed, period
    except (KeyError, TypeError, ValueError):
        return None


def clock_intervals(observations, period_lengths, max_gap=1.5):
    intervals = []
    previous = None
    for row in observations:
        current = _reading(row, period_lengths)
        if previous and current:
            t0, c0, p0 = previous
            t1, c1, p1 = current
            if p0 == p1 and 0 < t1 - t0 <= max_gap and 0 <= c1 - c0 <= t1 - t0 + 1:
                intervals.append({'period': p1, 'video': [t0, t1], 'elapsed': [c0, c1],
                                  'state': 'clock_held' if c0 == c1 else 'clock_advancing'})
        previous = current
    return intervals


def _runs(intervals):
    """Contiguous candidates form one occurrence; separated footage remains separate."""
    runs = []
    for interval in intervals:
        a, b = interval['video']
        if runs and abs(runs[-1][1] - a) < 1e-6:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return runs


def _match(record, indexes, intervals):
    selected = [intervals[i] for i in indexes]
    runs = _runs(selected)
    return {'record': record, 'interval_ids': indexes,
            'video_intervals': [x['video'] for x in selected], 'occurrences': runs,
            'status': 'unmatched' if not runs else 'ambiguous' if len(runs) > 1 else 'candidate'}


def align(observations, events, shifts, playoff=False, duration=None):
    lengths = {p: 1200 if p <= 3 or playoff else 300 for p in range(1, 21) if playoff or p <= 4}
    # Stable ordering; duplicated PTS cannot create a negative or zero-length segment.
    observations = sorted(observations, key=lambda r: float(r['video_seconds']))
    readings = [r for row in observations if (r := _reading(row, lengths)) is not None]
    conflicts = period_conflicts(readings, lengths)
    strict = [dict(r, valid=False) if conflict_at(float(r['video_seconds']), conflicts) else r
              for r in observations]
    intervals = clock_intervals(strict, lengths)
    for i, interval in enumerate(intervals):
        interval.update(id=i, event_indexes=[], shift_indexes=[], marker_indexes=[])
    matched_events = []
    for event_index, event in enumerate(events):
        try:
            clock = seconds(event['time_in_period'])
        except (KeyError, TypeError, ValueError):
            clock = None
        indexes = []
        if clock is not None and event.get('period_type') != 'SO':
            for i, interval in enumerate(intervals):
                if interval['period'] == event['period_number'] and interval['elapsed'][0] <= clock <= interval['elapsed'][1]:
                    indexes.append(i)
                    interval['event_indexes'].append(event_index)
        match = _match(event, indexes, intervals)
        # Ordering constrains the meaning of a held clock, not its exact video frame.
        match['timing_role'] = ('restart' if event.get('type_desc_key') in ('faceoff', 'period-start')
                                else 'stop' if event.get('type_desc_key') in ('stoppage', 'period-end', 'game-end', 'goal')
                                else 'event')
        matched_events.append(match)
    matched_shifts = []
    for shift_index, shift in enumerate(shifts):
        indexes = []
        start, end = shift.get('start_time'), shift.get('end_time')
        # NHL shift charts also contain zero-duration goal markers (505), not on-ice shifts (517).
        marker = start is not None and end == start and shift.get('type_code') == 505
        if marker:
            for i, interval in enumerate(intervals):
                if interval['period'] == shift['period'] and interval['elapsed'][0] <= start <= interval['elapsed'][1]:
                    indexes.append(i)
                    interval['marker_indexes'].append(shift_index)
        if start is not None and end is not None and end > start:
            for i, interval in enumerate(intervals):
                a, b = interval['elapsed']
                # Half-open shifts; a running segment touching only its start is not on-ice evidence.
                overlaps = start <= a < end if a == b else start < b and a < end
                if interval['period'] == shift['period'] and overlaps:
                    indexes.append(i)
                    interval['shift_indexes'].append(shift_index)
        match = _match(shift, indexes, intervals)
        match['record_kind'] = 'point_marker' if marker else 'shift'
        matched_shifts.append(match)
    # Explicit unaligned video spans. Period transitions are candidates for intermission,
    # not proof: the broadcast could cut directly between periods.
    gaps = []
    cursor = 0.0
    last_period = None
    last_elapsed = None
    for interval in intervals:
        start, end = interval['video']
        if start > cursor:
            reason = 'missing_or_discontinuous_clock'
            if last_period is not None and last_period != interval['period']:
                reason = 'period_transition'
            elif last_elapsed is not None and interval['elapsed'][0] < last_elapsed:
                reason = 'clock_reversal_possible_replay'
            gaps.append({'video': [cursor, start], 'status': 'unmatched',
                         'reason': reason})
        cursor = end
        last_period = interval['period']
        last_elapsed = interval['elapsed'][1]
    finish = duration if duration is not None else max((float(r['video_seconds']) for r in observations), default=0)
    if finish > cursor:
        gaps.append({'video': [cursor, finish], 'status': 'unmatched', 'reason': 'unobserved_tail'})
    # Preserve same-clock ordering to distinguish stoppage -> restart without inventing wall time.
    groups = {}
    for i, event in enumerate(events):
        groups.setdefault((event.get('period_number'), event.get('time_in_period')), []).append(i)
    boundaries = []
    for (period, clock), indexes in groups.items():
        ordered = sorted(indexes, key=lambda i: (events[i].get('sort_order') or 0, events[i].get('event_id') or 0))
        if any(matched_events[i]['timing_role'] != 'event' for i in ordered):
            boundaries.append({'period': period, 'elapsed_clock': clock, 'event_indexes': ordered})
    result = {'schema_version': 2, 'status': 'needs_review',
            'note': 'All video matches are candidates. Held clocks may be normal OCR quantization or stoppages. Repeated footage, missing clocks, and intermissions are not automatically verified. Shift membership is game-clock evidence, not visual player detection.',
            'clock_intervals': intervals, 'gaps': gaps, 'boundary_groups': boundaries,
            'events': matched_events, 'shifts': matched_shifts,
            'coverage': {kind: {status: sum(x['status'] == status for x in records)
                               for status in ('candidate', 'ambiguous', 'unmatched')}
                         for kind, records in (('events', matched_events), ('shifts', matched_shifts))}}
    return add_candidates(result, readings, conflicts, seconds)
