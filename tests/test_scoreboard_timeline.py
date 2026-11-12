import unittest
from copy import deepcopy
from studio.scoreboard.timeline import build_timeline, match_strength_window


def event(i, kind, elapsed, period=1, details=None, code=None):
    length=1200 if period<=3 else 300
    clock=length-elapsed
    return dict(game_id=2026020001,event_id=i,sort_order=i,type_desc_key=kind,
        period_number=period,period_type='REG' if period<=3 else 'OT',
        time_in_period=f'{elapsed//60:02d}:{elapsed%60:02d}',time_remaining=f'{clock//60:02d}:{clock%60:02d}',
        situation_code=code,details=details or {})


def bundle(events):
    return dict(events=events,shifts=[dict(player_id=10,team_id=4,period=1,start_time=10,end_time=60,type_code=517)],
        context={'game':dict(id=2026020001,awayTeam_id=4,homeTeam_id=13,awayTeam_abbrev='PHI',homeTeam_abbrev='FLA'),
                 'raw_events':deepcopy(events)},unavailable_context={})


class TimelineEvidenceTests(unittest.TestCase):
    def test_raw_penalty_details_join_and_source_records_remain_unchanged(self):
        data=bundle([event(1,'faceoff',0),event(2,'penalty',10,details={'typeCode':'MIN','duration':2,'eventOwnerTeamId':4}),
                     event(3,'faceoff',10,code='1451'),event(4,'period-end',1200,period=3)])
        data['events'][1].pop('details')
        original=deepcopy(data)
        result=build_timeline([],data,2026020001,duration=100)
        self.assertEqual(data,original)
        self.assertEqual([m['record'] for m in result['events']],data['events'])
        self.assertEqual(result['strength_context']['windows'][0]['pk_team'],'PHI')
        self.assertEqual(result['shifts'][0]['strength_window_ids'],['pp-001'])
        self.assertEqual(result['strength_context']['windows'][0]['status'],'unmatched')

    def test_overtime_carry_preserves_partial_regulation_without_guessing_overtime(self):
        data=bundle([event(1,'faceoff',0),event(2,'penalty',1143,period=3,
                     details={'typeCode':'MIN','duration':2,'eventOwnerTeamId':4}),
                     event(3,'period-end',1200,period=3),event(4,'faceoff',0,period=4)])
        strength=build_timeline([],data,2026020001)['strength_context']
        self.assertEqual(strength['status'],'partial')
        self.assertEqual(strength['unsupported_periods'],[4])
        self.assertFalse(strength['windows'][0]['complete'])
        self.assertEqual(strength['windows'][0]['end_game_seconds'],3600)

    def test_unsupported_penalty_does_not_discard_events_shifts_or_boundary_evidence(self):
        data=bundle([event(1,'faceoff',0),event(2,'penalty',10,details={'typeCode':'UNKNOWN','duration':9,'eventOwnerTeamId':4})])
        result=build_timeline([],data,2026020001)
        self.assertEqual(result['strength_context']['status'],'unsupported')
        self.assertEqual(len(result['events']),2)
        self.assertEqual(len(result['shifts']),1)
        self.assertEqual(result['play_context']['status'],'available')

    def test_two_viable_strength_occurrences_stay_ambiguous(self):
        window=dict(start_game_seconds=10,end_game_seconds=20,game_duration_seconds=10)
        rows=[];index=[]
        for offset in (0,100):
            rows.extend(dict(video_seconds=t+offset,clock_seconds=1200-t,period=1,valid=True) for t in range(10,21))
            index.extend([dict(period=1,clock_seconds=1190,start_video_seconds=10+offset,end_video_seconds=10+offset,observations=1),
                          dict(period=1,clock_seconds=1180,start_video_seconds=20+offset,end_video_seconds=20+offset,observations=1)])
        result=match_strength_window(window,index,rows)
        self.assertEqual(result['status'],'ambiguous')
        self.assertIsNone(result['selected'])
        self.assertGreaterEqual(len(result['boundary_candidates']),2)

    def test_pulled_goalie_does_not_cancel_penalty_window(self):
        data=bundle([event(1,'faceoff',0),event(2,'penalty',10,details={'typeCode':'MIN','duration':2,'eventOwnerTeamId':4}),
                     event(3,'shot-on-goal',30,code='0551'),event(4,'period-end',1200,period=3)])
        result=build_timeline([],data,2026020001)
        self.assertEqual(result['events'][2]['strength_window_ids'],['pp-001'])
        self.assertEqual(result['events'][2]['record']['situation_code'],'0551')
        self.assertEqual(result['strength_context']['windows'][0]['end_game_seconds'],130)

    def test_duplicate_observation_timestamps_do_not_discard_source_records(self):
        data=bundle([event(1,'faceoff',0),event(2,'stoppage',10)])
        rows=[dict(video_seconds=0,clock_seconds=1200,period=1,valid=True)]*2
        result=build_timeline(rows,data,2026020001)
        self.assertEqual(len(result['events']),2)
        self.assertTrue(result['play_context']['issues'])
        self.assertEqual(result['play_context']['segments'],[])

if __name__=='__main__':unittest.main()
