"""Clock-supported play boundaries, ported from nhl-tracking-data.

Supported is OCR evidence, never a verified live/replay classification.
"""

def clock_seconds(text):
    minutes, seconds = map(int, text.split(':'))
    if minutes < 0 or not 0 <= seconds < 60:
        raise ValueError(f'Invalid NHL clock: {text}')
    return minutes * 60 + seconds


def countdown_restart(interval, observations):
    """Find a nearby supported countdown, without crossing an OCR gap or clock jump."""
    end = float(interval['end_video_seconds'])
    baseline = float(interval['clock_seconds'])
    previous_t, previous_clock = end, baseline
    last_same, first_lower = end, None
    first_lower_clock = None
    for row in observations:
        t = float(row['video_seconds'])
        if t <= end or str(row['valid']).lower() != 'true':
            continue
        if t - end > 30 or t - previous_t > 2 or int(row['period']) != int(interval['period']):
            return None
        clock = float(row['clock_seconds'])
        if clock > previous_clock or previous_clock - clock > t - previous_t + 1.5:
            return None
        if first_lower is None:
            if clock == baseline:
                last_same = t
            elif 0 < baseline - clock <= 2:
                first_lower, first_lower_clock = t, clock
            else:
                return None
        elif clock < first_lower_clock:
            return dict(last_same_clock_video_seconds=last_same,
                        first_lower_clock_video_seconds=first_lower,
                        countdown_support_until_video_seconds=t)
        previous_t, previous_clock = t, clock
    return None


def clip_window(event_type, cue):
    before, after = (15, 3) if event_type == 'goal' else (4, 8)
    start = max(0, cue - before)
    return start, cue + after - start


def candidates_for(event, intervals, tolerance=1, observations=None, closest_only=True):
    target = clock_seconds(event['time_remaining'])
    matching = [i for i in intervals if int(i['period']) == event['period_number']
                and abs(float(i['clock_seconds']) - target) <= tolerance]
    # Exact matches take precedence; when absent, retain every equally close interval.
    if not matching:
        return []
    error = min(abs(float(i['clock_seconds']) - target) for i in matching)
    if closest_only:
        matching = [i for i in matching if abs(float(i['clock_seconds'])-target) == error]
    result = []
    for index, interval in enumerate(sorted(matching, key=lambda i:float(i['start_video_seconds']))):
        start, end = float(interval['start_video_seconds']), float(interval['end_video_seconds'])
        # Review cues only; faceoffs prefer a supported countdown over a stopped-clock endpoint.
        cue = start if event['type_desc_key'] == 'goal' else end
        restart = countdown_restart(interval, observations) if event['type_desc_key']=='faceoff' and observations is not None else None
        if restart:
            cue = restart['first_lower_clock_video_seconds']
        result.append(dict(candidate_id=f"{event['event_id']}-c{index+1:03d}",
                           start_video_seconds=start, end_video_seconds=end,
                           observed_clock_seconds=float(interval['clock_seconds']),
                           clock_error_seconds=float(interval['clock_seconds'])-target,
                           observations=int(interval['observations']), review_cue_seconds=cue,
                           cue_rule='first_observation' if event['type_desc_key']=='goal' else ('supported_countdown' if restart else 'last_observation_restart_unconfirmed'),
                           countdown_evidence=restart,
                           review_status='unreviewed'))
    return result



STOPS = {'stoppage', 'goal', 'period-end', 'game-end', 'penalty'}


def derive_boundaries(snapshot):
    events = sorted(snapshot['events'], key=lambda e: e['sort_order'])
    if not events or len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Full play-by-play with unique event IDs is required')
    if any(e['game_id'] != snapshot['game']['id'] for e in events):
        raise ValueError('Event game identity mismatch')
    boundaries, issues = [], []
    for period in sorted({e['period_number'] for e in events}):
        group = [e for e in events if e['period_number'] == period]
        if any(e.get('period_type') == 'SO' for e in group):
            issues.append(f'Period {period}: shootout requires separate labels')
        opening = next((e for e in group if e['type_desc_key'] == 'faceoff'), None)
        if opening is None or opening.get('time_in_period') != '00:00':
            issues.append(f'Period {period}: missing opening faceoff')
        if not any(e['type_desc_key'] == 'period-end' for e in group):
            issues.append(f'Period {period}: missing period end')
    active = None
    for event in events:
        kind = event['type_desc_key']
        if active and event['period_number'] != active['period_number']:
            issues.append(f"Missing stop after event {active['event_id']}")
            active = None
        if kind == 'faceoff':
            if active:
                issues.append(f"Faceoff {event['event_id']} before previous play stopped")
            boundaries.append(dict(**event, state='live'))
            active = event
        elif kind in STOPS and active:
            if clock_seconds(event['time_remaining']) > clock_seconds(active['time_remaining']):
                issues.append(f"Clock increases before stop {event['event_id']}")
            boundaries.append(dict(**event, state='not_live'))
            active = None
    if active:
        issues.append(f"Missing final stop after event {active['event_id']}")
    if not boundaries:
        raise ValueError('No faceoff/start boundaries in source feed')
    return boundaries, issues


def link_boundaries(boundaries, intervals, observations):
    linked = []
    for event in boundaries:
        # Stops use first observation at the stopped clock; starts require countdown.
        proxy = dict(event, type_desc_key='faceoff' if event['state'] == 'live' else 'goal')
        candidates = candidates_for(proxy, intervals, observations=observations)
        for candidate in candidates:
            candidate.pop('review_status', None)
        linked.append(dict(event, candidates=candidates))
    return linked


def select_matches(linked, observations):
    valid = [r for r in observations if str(r['valid']).lower() == 'true']
    for event in linked:
        supported = []
        for candidate in event['candidates']:
            if candidate['clock_error_seconds'] != 0:
                continue
            start = candidate['start_video_seconds']
            end = candidate['end_video_seconds']
            if end - start < 2 or candidate['observations'] < 3:
                continue
            if event['state'] == 'live':
                if not candidate['countdown_evidence']:
                    continue
            else:
                # A genuine stop needs an approaching countdown, then a plateau.
                before = [r for r in valid if start - 3 <= float(r['video_seconds']) < start]
                if len(before) < 2:
                    continue
                clocks = [float(r['clock_seconds']) for r in before]
                times = [float(r['video_seconds']) for r in before] + [start]
                target = candidate['observed_clock_seconds']
                if (any(int(r['period']) != event['period_number'] for r in before)
                        or max(b-a for a,b in zip(times,times[1:])) > 2
                        or len(set(clocks)) < 2 or clocks[-1] <= target
                        or any(b > a for a,b in zip(clocks, clocks[1:]))
                        or clocks[0] - target > start - times[0] + 1.5):
                    continue
            supported.append(candidate)
        event.pop('review_status', None)
        event['selected'] = supported[0] if len(supported) == 1 else None
        event['status'] = 'matched' if len(supported) == 1 else 'skipped'
        event['skip_reason'] = None if len(supported) == 1 else ('ambiguous_clock' if len(supported)>1 else 'insufficient_clock_evidence')
    # Reject inverted selections; do not choose a later replay over source order.
    selected = [e for e in linked if e['selected']]
    rejected = set()
    for a,b in zip(selected,selected[1:]):
        if a['selected']['review_cue_seconds'] >= b['selected']['review_cue_seconds']:
            rejected.update((a['event_id'], b['event_id']))
    for e in linked:
        if e['event_id'] in rejected:
            e.update(selected=None, status='skipped', skip_reason='video_order_conflict')
    return linked


def collect_segments(linked, observations):
    """Only adjacent supported boundaries define labels; skipped spans stay unknown."""
    segments = []
    valid = [r for r in observations if str(r['valid']).lower() == 'true']
    for a,b in zip(linked, linked[1:]):
        if not a['selected'] or not b['selected'] or a['state'] == b['state'] or a['period_number'] != b['period_number']:
            continue
        start, end = a['selected']['review_cue_seconds'], b['selected']['review_cue_seconds']
        if end <= start:
            continue
        ca, cb = clock_seconds(a['time_remaining']), clock_seconds(b['time_remaining'])
        if a['state'] == 'live':
            # A live stretch must advance clock in real time without missing OCR
            # or replay jumps. Breaks cannot be silently included as live.
            if ca <= cb or abs((end-start)-(ca-cb)) > 3:
                continue
            rows = [r for r in valid if start <= float(r['video_seconds']) <= end]
            if len(rows) < 3:
                continue
            times = [start] + [float(r['video_seconds']) for r in rows] + [end]
            clocks = [float(r['clock_seconds']) for r in rows]
            if (any(int(r['period']) != a['period_number'] for r in rows)
                    or max(y-x for x,y in zip(times,times[1:])) > 2
                    or any(y > x or x-y > float(rb['video_seconds'])-float(ra['video_seconds'])+1.5
                           for x,y,ra,rb in zip(clocks,clocks[1:],rows,rows[1:]))
                    or any(not cb <= c <= ca for c in clocks)):
                continue
        elif ca != cb:
            continue
        segments.append(dict(segment_id=f"{a['event_id']}-{b['event_id']}", state=a['state'],
                             is_live=int(a['state']=='live'), period_number=a['period_number'],
                             start_video_seconds=start, end_video_seconds=end,
                             start_event_id=a['event_id'], end_event_id=b['event_id']))
    return segments
