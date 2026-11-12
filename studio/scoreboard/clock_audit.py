"""Keep clock discontinuities and repeated occurrences as explicit evidence."""

def analyze(rows):
    flags, intervals = [], []
    previous = None
    open_intervals = {}
    for row in rows:
        if str(row['valid']).lower() != 'true':
            continue
        t, clock, period = float(row['video_seconds']), float(row['clock_seconds']), int(row['period'])
        if previous:
            t0, c0, p0 = previous
            dt = t - t0
            if dt <= 0:
                raise ValueError('Observation timestamps must increase')
            reasons = []
            if period < p0:
                reasons.append('period_regression')
            elif period > p0:
                reasons.append('period_change')
            if dt > 2:
                reasons.append('reading_gap')
            elif period == p0:
                if clock > c0 + 2:
                    reasons.append('clock_increased')
                elif c0 - clock > dt + 2:
                    reasons.append('countdown_jump')
            if reasons:
                flags.append(dict(video_seconds=t, previous_video_seconds=t0,
                                  previous_period=p0, period=period, previous_clock=c0,
                                  clock=clock, reasons=';'.join(reasons)))
        key = (period, clock)
        interval = open_intervals.get(key)
        # Observed repeats remain ambiguous; gaps are never filled or interpolated.
        if interval is None or t - interval['end_video_seconds'] > 1.5:
            interval = dict(period=period, clock_seconds=clock, start_video_seconds=t,
                            end_video_seconds=t, observations=0)
            intervals.append(interval)
            open_intervals[key] = interval
        interval['end_video_seconds'] = t
        interval['observations'] += 1
        previous = (t, clock, period)
    counts = {}
    for item in intervals:
        key = (item['period'], item['clock_seconds'])
        counts[key] = counts.get(key, 0) + 1
    for item in intervals:
        item['ambiguous'] = counts[(item['period'], item['clock_seconds'])] > 1
    return flags, intervals

