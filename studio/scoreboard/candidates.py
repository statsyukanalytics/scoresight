"""Review locations from incomplete clocks, kept separate from interval evidence."""


def period_conflicts(readings, lengths):
    """Flag a sustained reset after period end that joins the next period's clock.

    This can be a stale broadcast label or edited/repeated footage. It proposes
    a period interpretation for review; it never changes the OCR observation.
    """
    conflicts = []
    for i in range(3, len(readings) - 3):
        t, elapsed, period = readings[i]
        previous = readings[i - 3:i]
        if (period >= 3 or period + 1 not in lengths or elapsed > 30
                or not all(p == period and e >= lengths[period] - 1 for _, e, p in previous)
                or not 0 < t - previous[-1][0] <= 1800):
            continue
        j = i
        while j < len(readings) and readings[j][2] == period:
            j += 1
        segment = readings[i:j]
        following = readings[j:j + 3]
        if len(segment) < 20 or len(following) < 3:
            continue
        last_t, last_e, _ = segment[-1]
        next_t, next_e, next_p = following[0]
        if (last_t - t < 60 or last_t - t > 1200 or next_p != period + 1
                or not all(p == next_p for _, _, p in following)
                or not 0 < next_t - last_t <= 120
                or not 0 <= next_e - last_e <= next_t - last_t + 1):
            continue
        plausible = sum(0 <= b[1] - a[1] <= b[0] - a[0] + 1
                        for a, b in zip(segment, segment[1:]))
        if plausible / (len(segment) - 1) < .98:
            continue
        conflicts.append(dict(video=[t, last_t], observed_period=period,
                              candidate_period=period + 1,
                              reason='period_reset_then_next_period_continuity',
                              prior_end_video=previous[-1][0], next_period_video=next_t,
                              observations=len(segment), status='needs_review'))
    return conflicts


def conflict_at(t, conflicts):
    return next((c for c in conflicts if c['video'][0] <= t <= c['video'][1]), None)


def _groups(points, basis, target=None):
    groups = []
    for point in points:
        t, elapsed, period, observed = point
        contextual = observed != period
        if (groups and t - groups[-1]['video'][1] <= 1.5
                and groups[-1]['period_context'] == contextual):
            groups[-1]['video'][1] = t
            groups[-1]['elapsed'][1] = elapsed
            groups[-1]['observations'] += 1
        else:
            groups.append(dict(video=[t, t], elapsed=[elapsed, elapsed], basis=basis,
                               period_context=contextual, observed_period=observed,
                               candidate_period=period, observations=1,
                               clock_error_seconds=abs(elapsed - target) if target is not None else None))
        if target is not None:
            groups[-1]['clock_error_seconds'] = max(groups[-1]['clock_error_seconds'], abs(elapsed - target))
    return groups


def _brackets(points, period, start, end):
    candidates = []
    # Adjacent accepted readings only: never skip intervening periods or reversals.
    for a, b in zip(points, points[1:]):
        t0, e0, p0, o0 = a
        t1, e1, p1, o1 = b
        if (p0 != period or p1 != period or not 0 < t1 - t0 <= 120
                or not 0 < e1 - e0 <= min(60, t1 - t0 + 1)):
            continue
        overlaps = e0 < start < e1 if start == end else start < e1 and e0 < end
        if overlaps:
            candidates.append(dict(video=[t0, t1], elapsed=[e0, e1], basis='bracketed_clock',
                                   period_context=o0 != p0 or o1 != p1, observed_period=o0,
                                   candidate_period=period, observations=2, clock_error_seconds=None))
    return candidates


def review_candidates(points, period, start, end):
    same_period = [p for p in points if p[2] == period]
    if start != end:
        inside = [p for p in same_period if start <= p[1] < end]
        return _groups(inside, 'shift_clock_overlap') if inside else _brackets(points, period, start, end)
    exact = [p for p in same_period if abs(p[1] - start) < 1e-6]
    if exact:
        return _groups(exact, 'exact_reading', start)
    nearby = [p for p in same_period if abs(p[1] - start) <= 1]
    if nearby:
        return _groups(nearby, 'nearby_clock', start)
    brackets = _brackets(points, period, start, end)
    if brackets:
        return brackets
    # An explicit review anchor, not an extrapolated event timestamp.
    distance = min((abs(p[1] - start) for p in same_period), default=float('inf'))
    if distance <= 10:
        return _groups([p for p in same_period if abs(abs(p[1] - start) - distance) < 1e-6],
                       'nearby_clock', start)
    return []


def add_candidates(result, readings, conflicts, parse_seconds):
    points = []
    for t, elapsed, period in readings:
        conflict = conflict_at(t, conflicts)
        points.append((t, elapsed, conflict['candidate_period'] if conflict else period, period))
    for kind in ('events', 'shifts'):
        for match in result[kind]:
            match['clock_status'] = match['status']
            match['review_candidates'] = []
            if match['status'] != 'unmatched':
                continue
            record = match['record']
            try:
                if kind == 'events':
                    if record.get('period_type') == 'SO':
                        continue
                    period = int(record['period_number'])
                    start = end = parse_seconds(record['time_in_period'])
                else:
                    period = int(record['period'])
                    start, end = float(record['start_time']), float(record['end_time'])
                    if end < start or (end == start and match['record_kind'] != 'point_marker'):
                        continue
                candidates = review_candidates(points, period, start, end)
            except (KeyError, TypeError, ValueError):
                continue
            match['review_candidates'] = candidates
            match['status'] = 'ambiguous' if len(candidates) > 1 else 'candidate' if candidates else 'unmatched'
    result['period_conflicts'] = conflicts
    result['candidate_policy'] = dict(max_bracket_video_seconds=120, max_bracket_game_seconds=60,
                                    max_nearby_clock_seconds=10,
                                    note='Review windows and nearby anchors are not event timestamps or on-ice evidence. Period-context candidates retain the conflicting broadcast label.')
    result['coverage'] = {kind: {status: sum(x['status'] == status for x in records)
                               for status in ('candidate', 'ambiguous', 'unmatched')}
                          for kind, records in (('events', result['events']), ('shifts', result['shifts']))}
    result['connection_coverage'] = {kind: dict(total=len(records),
        clock_interval=sum(x['clock_status'] != 'unmatched' for x in records),
        review_only=sum(bool(x['review_candidates']) for x in records),
        unresolved=sum(x['status'] == 'unmatched' for x in records))
        for kind, records in (('events', result['events']), ('shifts', result['shifts']))}
    return result
