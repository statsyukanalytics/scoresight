"""Build the complete NHL timeline with independent boundary and strength evidence."""
import math

from .align import align, _reading, seconds
from .boundaries import derive_boundaries, link_boundaries, select_matches, collect_segments, candidates_for
from .clock_audit import analyze
from .penalty_schedule import derive_windows
from .candidates import conflict_at


def snapshot_for(bundle, game_id):
    game = bundle['context'].get('game')
    if not game or game.get('id') != game_id:
        raise ValueError('NHL game context does not match the requested game')
    raw = {r['event_id']: r for r in bundle['context'].get('raw_events', [])}
    events = []
    for event in bundle['events']:
        if event.get('game_id') != game_id:
            raise ValueError('Event belongs to a different NHL game')
        details = raw.get(event['event_id'], {}).get('details') or event.get('details') or {}
        if not isinstance(details, dict):
            raise ValueError('NHL event details must be an object')
        # The expanded view omits penalty typeCode; preserve it from raw events.
        details = {**{k: v for k, v in event.items() if v is not None}, **details}
        events.append({**event, 'details': details})
    return {'game': game, 'events': events}


def strength_boundary(game_seconds, ending=False):
    period = min(3, int((game_seconds - (1e-6 if ending and game_seconds else 0)) // 1200) + 1)
    remaining = int(period * 1200 - game_seconds)
    return period, f'{remaining // 60:02d}:{remaining % 60:02d}'


def match_strength_window(window, index, observations):
    """Rank evidence without choosing between multiple plausible occurrences."""
    ps, cs = strength_boundary(window['start_game_seconds'])
    pe, ce = strength_boundary(window['end_game_seconds'], True)
    starts = candidates_for(dict(event_id=0, type_desc_key='faceoff', period_number=ps, time_remaining=cs),
                            index, tolerance=0, observations=observations)
    ends = candidates_for(dict(event_id=1, type_desc_key='goal', period_number=pe, time_remaining=ce),
                          index, tolerance=0)
    valid = [(float(r['video_seconds']), (int(r['period']) - 1) * 1200 + 1200 - float(r['clock_seconds']))
             for r in observations if str(r['valid']).lower() == 'true' and int(r['period']) <= 3]
    pairs = []
    for start in starts:
        for end in ends:
            a, b = start['review_cue_seconds'], end['review_cue_seconds']
            if b <= a or b - a < window['game_duration_seconds'] - 3:
                continue
            between = [t for v, t in valid if a <= v <= b]
            inside = [t for t in between if window['start_game_seconds'] - 1 <= t <= window['end_game_seconds'] + 1]
            covered = {math.floor(t) for t in inside if window['start_game_seconds'] <= t <= window['end_game_seconds']}
            coverage = min(1, len(covered) / (window['game_duration_seconds'] + 1))
            purity = len(inside) / max(1, len(between))
            if purity < .75:
                continue
            pairs.append(dict(video=[a, b], observed_game_seconds_coverage=coverage,
                              clock_range_fraction=purity, start_countdown_observed=bool(start['countdown_evidence']),
                              start_stopped_clock_observed=start['end_video_seconds'] - start['start_video_seconds'] >= 2))
    pairs.sort(key=lambda p: (p['start_stopped_clock_observed'] and p['start_countdown_observed'],
                             p['start_countdown_observed'], p['clock_range_fraction'],
                             p['observed_game_seconds_coverage'], -(p['video'][1] - p['video'][0])), reverse=True)
    return dict(start_period=ps, start_clock=cs, end_period=pe, end_clock=ce,
                boundary_candidates=pairs, selected=pairs[0] if len(pairs) == 1 else None,
                status='unmatched' if not pairs else 'candidate' if len(pairs) == 1 else 'ambiguous')


def add_evidence(result, observations, bundle, game_id):
    snapshot = snapshot_for(bundle, game_id)
    lengths = {p: 1200 if p <= 3 or str(game_id)[4:6] == '03' else 300
               for p in range(1, 21) if str(game_id)[4:6] == '03' or p <= 4}
    rows = sorted(observations, key=lambda r: float(r['video_seconds']))
    rows = [dict(r, valid=_reading(r, lengths) is not None
                 and conflict_at(float(r['video_seconds']), result.get('period_conflicts', [])) is None)
            for r in rows]
    audit_issue = None
    try:
        flags, index = analyze(rows)
    except ValueError as error:
        audit_issue = str(error)
        flags, index = [], []
    result['clock_flags'] = flags
    try:
        boundaries, issues = derive_boundaries(snapshot)
        if audit_issue:
            issues.append(audit_issue)
        linked = select_matches(link_boundaries(boundaries, index, rows), rows)
        segments = collect_segments(linked, rows)
        evidence = []
        for boundary in linked:
            status = 'supported' if boundary['selected'] else 'ambiguous' if boundary['skip_reason'] == 'ambiguous_clock' else 'unmatched'
            evidence.append(dict(event_id=boundary['event_id'], state=boundary['state'], status=status,
                                 reason=boundary.get('skip_reason'), selected=boundary['selected'], candidates=boundary['candidates']))
        by_id = {e['event_id']: e for e in evidence}
        for match in result['events']:
            if match['record']['event_id'] in by_id:
                match['boundary_evidence'] = by_id[match['record']['event_id']]
        for segment in segments:
            segment['video'] = [segment['start_video_seconds'], segment['end_video_seconds']]
            segment['status'] = 'supported_candidate'
        result['play_context'] = dict(status='available', boundaries=evidence, segments=segments, issues=issues,
            basis='Clock-supported start/stop candidates. Unknown spans are excluded; no visual live/replay verification.')
    except (ValueError, KeyError, TypeError) as error:
        result['play_context'] = dict(status='unavailable', reason=str(error), boundaries=[], segments=[], issues=[])

    regulation = [e for e in snapshot['events'] if e['period_type'] == 'REG' and e['period_number'] <= 3]
    other_periods = sorted({e['period_number'] for e in snapshot['events'] if e not in regulation})
    strength = dict(status='available', windows=[], unsupported_periods=other_periods,
                    basis='Penalty-derived regulation manpower; goalie pulls remain separate event observations.',
                    issues=['Overtime and shootout strength scheduling is unsupported.'] if other_periods else [])
    try:
        windows, active_penalties = derive_windows({**snapshot, 'events': regulation}, include_state=True)
        end = max((e['period_number'] - 1) * 1200 + seconds(e['time_in_period']) for e in regulation)
        complete_regulation = any(e['period_number'] == 3 and e['type_desc_key'] == 'period-end'
                                  and seconds(e['time_in_period']) == 1200 for e in regulation)
        for window in windows:
            window['complete'] = not (window['end_game_seconds'] == end and (not complete_regulation or (other_periods and active_penalties)))
            if not window['complete']:
                window['end_reason'] = 'unresolved_continuation'
            window.update(match_strength_window(window, index, rows))
            a, b = window['start_game_seconds'], window['end_game_seconds']
            window['event_indexes'] = [i for i, e in enumerate(bundle['events']) if e['period_type'] == 'REG'
                and a <= (e['period_number'] - 1) * 1200 + seconds(e['time_in_period']) <= b]
            window['shift_indexes'] = [i for i, s in enumerate(bundle['shifts']) if s.get('period', 4) <= 3
                and s.get('start_time') is not None and s.get('end_time') is not None and s['end_time'] > s['start_time']
                and a < (s['period'] - 1) * 1200 + s['end_time'] and (s['period'] - 1) * 1200 + s['start_time'] < b]
        strength.update(windows=windows, status='partial' if other_periods or not complete_regulation else 'available')
        if not complete_regulation:
            strength['issues'].append('Regulation feed has no observed final period-end; windows may be truncated.')
        if other_periods and active_penalties:
            strength['issues'].append('Active penalties carry beyond regulation; final window remains incomplete.')
        for kind in ('events', 'shifts'):
            for match in result[kind]:
                match['strength_window_ids'] = []
            for window in windows:
                for i in window['event_indexes' if kind == 'events' else 'shift_indexes']:
                    result[kind][i]['strength_window_ids'].append(window['segment_id'])
    except (ValueError, KeyError, TypeError) as error:
        strength.update(status='unsupported', reason=str(error), windows=[])
    if audit_issue:
        strength['issues'].append(audit_issue)
    result['strength_context'] = strength
    result['pipeline_version'] = 'scoreboard-evidence-v2'
    return result


def build_timeline(observations, bundle, game_id, duration=None):
    result = align(observations, bundle['events'], bundle['shifts'],
                   playoff=str(game_id)[4:6] == '03', duration=duration)
    result.update(context=bundle['context'], unavailable_context=bundle['unavailable_context'], nhl_game_id=game_id)
    return add_evidence(result, observations, bundle, game_id)
