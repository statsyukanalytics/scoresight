"""Match NHL records to observed clock intervals; never invent a unique replay time."""
import math


def seconds(value):
    minutes, seconds = value.split(':')
    return int(minutes) * 60 + float(seconds)


def clock_intervals(observations, period_lengths, max_gap=1.5):
    intervals = []
    previous = None
    for row in observations:
        if str(row['valid']).lower() not in ('true', '1'):
            previous = None
            continue
        period = int(row['period'])
        length = period_lengths.get(period)
        if length is None:
            previous = None
            continue
        current = (float(row['video_seconds']), length - float(row['clock_seconds']), period)
        if not all(math.isfinite(n) for n in current) or not 0 <= current[1] <= length:
            previous = None
            continue
        if previous:
            t0, c0, p0 = previous
            t1, c1, p1 = current
            if p0 == p1 and 0 < t1 - t0 <= max_gap and 0 <= c1 - c0 <= t1 - t0 + 1:
                intervals.append({'period': p1, 'video': [t0, t1], 'elapsed': [c0, c1]})
        previous = current
    return intervals


def align(observations, events, shifts, playoff=False):
    lengths = {p: 1200 if p <= 3 or playoff else 300 for p in range(1, 21) if playoff or p <= 4}
    intervals = clock_intervals(observations, lengths)
    matched_events = []
    for event in events:
        candidates = []
        try:
            clock = seconds(event['time_in_period'])
        except (TypeError, ValueError, AttributeError):
            clock = None
        if clock is not None and event.get('period_type') != 'SO':
            for interval in intervals:
                if interval['period'] == event['period_number'] and interval['elapsed'][0] <= clock <= interval['elapsed'][1]:
                    candidates.append(interval['video'])
        matched_events.append({'record': event, 'video_intervals': candidates,
                               'status': 'candidate' if candidates else 'unmatched'})
    matched_shifts = []
    for shift in shifts:
        candidates = []
        start, end = shift.get('start_time'), shift.get('end_time')
        if start is not None and end is not None and end > start:
            for interval in intervals:
                a, b = interval['elapsed']
                if interval['period'] == shift['period'] and start <= b and a < end:
                    candidates.append(interval['video'])
        matched_shifts.append({'record': shift, 'video_intervals': candidates})
    return {'schema_version': 1, 'status': 'needs_review',
            'note': 'Clock matches are candidate intervals. OCR cannot distinguish live footage from replay or identify the exact event frame during a stopped clock.',
            'clock_intervals': intervals, 'events': matched_events, 'shifts': matched_shifts}
