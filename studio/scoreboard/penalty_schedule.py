"""Regulation manpower scheduling for batch extraction (NHL rules 16–20).

Includes bench minors, double minors, majors and coincidental penalties.
Recorded live-play situations validate the schedule; unsupported cases fail.
"""
from itertools import groupby

from .strength import decode_strength


def event_time(event):
    if event['period_type']!='REG' or not 1<=event['period_number']<=3:
        raise ValueError('Strength scheduling currently supports regulation games only')
    minutes, seconds=map(int,event['time_in_period'].split(':'))
    return (event['period_number']-1)*1200+minutes*60+seconds


def derive_windows(snapshot, include_state=False):
    game = snapshot['game']
    away, home = game['awayTeam_id'], game['homeTeam_id']
    events = sorted(snapshot['events'], key=lambda e: e['sort_order'])
    regulation = [e for e in events if e['period_type'] == 'REG']
    overtime = [e for e in events if e['period_type'] == 'OT']
    if any(e['type_desc_key'] == 'penalty' for e in overtime):
        raise ValueError('Overtime penalties require an overtime manpower scheduler')
    active, windows = [], []
    cursor = 0
    expiry_before = None
    expiry_time = None

    def counts():
        return tuple(5 - sum(p['team'] == team for p in active) for team in (away, home))

    def emit(start, end, reason, event_id=None):
        a, h = counts()
        if end <= start or a == h:
            return
        team = away if a > h else home
        phase = dict(start_game_seconds=start, end_game_seconds=end, away_skaters=a, home_skaters=h)
        if windows and windows[-1]['pp_team_id'] == team and windows[-1]['end_game_seconds'] == start:
            w = windows[-1]
            w.update(end_game_seconds=end, end_reason=reason, end_event_id=event_id)
            w['penalty_event_ids'] = sorted(set(w['penalty_event_ids']) | {p['id'] for p in active})
            prior = w['phases'][-1]
            if (prior['away_skaters'], prior['home_skaters']) == (a, h):
                prior['end_game_seconds'] = end
            else:
                w['phases'].append(phase)
        else:
            windows.append(dict(segment_id=f'pp-{len(windows)+1:03d}', pp_team_id=team,
                                pk_team_id=home if team == away else away,
                                start_game_seconds=start, end_game_seconds=end,
                                end_reason=reason, end_event_id=event_id,
                                penalty_event_ids=sorted({p['id'] for p in active}), phases=[phase]))

    def finish(p, at):
        if p['remaining']:
            p['remaining'] -= 1
            p['expires'] = at + 120
        else:
            active.remove(p)

    def advance(to, reason, event_id=None):
        nonlocal cursor, expiry_before, expiry_time
        if to < cursor:
            raise ValueError('NHL event times are not chronological')
        while active and min(p['expires'] for p in active) <= to:
            expiry = min(p['expires'] for p in active)
            emit(cursor, expiry, 'penalty_expired')
            expiry_before, expiry_time = counts(), expiry
            for p in list(active):
                if p['expires'] == expiry:
                    finish(p, expiry)
            cursor = expiry
        emit(cursor, to, reason, event_id)
        cursor = to

    def assess(group, t):
        penalties = []
        for e in group:
            d = e['details']; kind = d.get('typeCode'); duration = d.get('duration')
            if kind in ('MIS', 'GAM'):
                continue
            if (kind, duration) not in [('MIN', 2), ('MIN', 4), ('BEN', 2), ('MAJ', 5)]:
                raise ValueError(f"Unsupported penalty {e['event_id']}: {kind}/{duration}")
            penalties.append(dict(id=e['event_id'], team=d['eventOwnerTeamId'], kind=kind,
                                  duration=duration, player=d.get('committedByPlayerId')))
        # One minor each with no clocks running produces 4v4. Other equal
        # coincidental minors/majors are substituted and do not reduce manpower.
        simple_pair = (not active and len(penalties) == 2
                       and all(p['kind'] == 'MIN' and p['duration'] == 2 for p in penalties)
                       and penalties[0]['team'] != penalties[1]['team'])
        if not simple_pair:
            for duration in (2, 4, 5):
                a = [p for p in penalties if p['team'] == away and p['duration'] == duration]
                h = [p for p in penalties if p['team'] == home and p['duration'] == duration]
                for pa, ph in zip(a, h):
                    penalties.remove(pa); penalties.remove(ph)
        for p in penalties:
            if p['team'] not in (away, home):
                raise ValueError('Unknown penalized team')
            if sum(q['team'] == p['team'] for q in active) >= 2:
                raise ValueError('Queued third penalties require explicit scheduling')
            if p['player'] and any(q['player'] == p['player'] for q in active):
                raise ValueError('Multiple sequential penalties to one player require explicit scheduling')
            p.update(expires=t + (300 if p['kind'] == 'MAJ' else 120),
                     remaining=1 if p['duration'] == 4 else 0, releasable=p['kind'] != 'MAJ' and not simple_pair)
            active.append(p)

    live = {'faceoff', 'shot-on-goal', 'missed-shot', 'blocked-shot', 'goal', 'hit', 'giveaway', 'takeaway'}
    for t, grouped in groupby(regulation, event_time):
        group = list(grouped)
        penalties = [e for e in group if e['type_desc_key'] == 'penalty']
        assessed = False
        for e in group:
            kind = e['type_desc_key']
            advance(t, kind, e['event_id'])
            if kind == 'penalty':
                if not assessed:
                    assess(penalties, t); assessed = True
                continue
            recorded = decode_strength(e.get('situation_code'))
            if kind in live and recorded['classification'] in ('even_strength', 'power_play'):
                observed = (recorded['away_skaters'], recorded['home_skaters'])
                # An event logged exactly at expiry can describe the instant
                # before the player returns. Do not move the actual expiry.
                if counts() != observed and not (t == expiry_time and observed == expiry_before):
                    raise ValueError(f"Penalty schedule disagrees at event {e['event_id']} ({t}s): {counts()} vs {observed}")
            if kind == 'goal':
                a, h = counts(); scoring = e['details']['eventOwnerTeamId']
                short = home if scoring == away and a > h else away if scoring == home and h > a else None
                candidates = [p for p in active if p['team'] == short and p['releasable']]
                if candidates:
                    finish(min(candidates, key=lambda p: p['expires']), t)
    end = max(event_time(e) for e in regulation)
    advance(end, 'game_end')
    if overtime and active:
        raise ValueError('Penalty carries into overtime; explicit overtime scheduling required')
    for w in windows:
        if w['end_game_seconds'] == end:
            w['end_reason'] = 'game_end'
        w['game_duration_seconds'] = w['end_game_seconds'] - w['start_game_seconds']
        w['pp_team'] = game['awayTeam_abbrev'] if w['pp_team_id'] == away else game['homeTeam_abbrev']
        w['pk_team'] = game['awayTeam_abbrev'] if w['pk_team_id'] == away else game['homeTeam_abbrev']
    return (windows, [dict(p) for p in active]) if include_state else windows
