import unittest
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from studio.scoreboard.boundaries import derive_boundaries, link_boundaries, select_matches, collect_segments


def event(i, kind, clock, period=1):
    return dict(game_id=1, event_id=i, sort_order=i, type_desc_key=kind,
                period_number=period, period_type='REG', time_remaining=clock,
                time_in_period='00:00' if i == 1 else '01:00')


def snapshot(events):
    return dict(game=dict(id=1), events=events)


class LivePlayChecks(unittest.TestCase):
    def setUp(self):
        self.events = [event(1, 'faceoff', '20:00'), event(2, 'stoppage', '19:00'),
                       event(3, 'penalty', '19:00'), event(4, 'faceoff', '19:00'),
                       event(5, 'goal', '18:00'), event(6, 'faceoff', '18:00'),
                       event(7, 'period-end', '00:00')]

    def test_binary_boundaries_deduplicate_stops(self):
        boundaries, issues = derive_boundaries(snapshot(self.events))
        self.assertEqual(issues, [])
        self.assertEqual([e['event_id'] for e in boundaries], [1, 2, 4, 5, 6, 7])
        self.assertEqual([e['state'] for e in boundaries], ['live', 'not_live'] * 3)

    def test_penalty_closes_play_but_delayed_penalty_does_not(self):
        events = [self.events[0], event(2, 'delayed-penalty', '19:30'),
                  event(3, 'penalty', '19:00'), *self.events[3:]]
        boundaries, _ = derive_boundaries(snapshot(events))
        self.assertEqual(boundaries[1]['event_id'], 3)

    def test_missing_stops_and_incomplete_feeds_are_flagged(self):
        _, issues = derive_boundaries(snapshot([self.events[0], self.events[3]]))
        self.assertTrue(issues)
        with self.assertRaises(ValueError):
            derive_boundaries(snapshot([self.events[1]]))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            derive_boundaries(snapshot([self.events[0], self.events[0]]))

    def test_unmatched_and_multiple_candidates_are_skipped(self):
        boundaries, _ = derive_boundaries(snapshot(self.events))
        intervals = [dict(period=1, clock_seconds=1200, start_video_seconds=t,
                          end_video_seconds=t+2, observations=4) for t in (10, 200)]
        linked = link_boundaries(boundaries, intervals, [])
        self.assertEqual(len(linked[0]['candidates']), 2)
        self.assertEqual(linked[1]['candidates'], [])
        select_matches(linked, [])
        self.assertTrue(all(e['status'] == 'skipped' for e in linked))

    def test_confident_matching_and_unknown_spans(self):
        boundaries, _ = derive_boundaries(snapshot(self.events))
        # Stable opening clock -> countdown -> stable stopped clock -> restart.
        values = [(0,1200),(1,1200),(2,1200),(3,1199),(4,1198),
                  (60,1142),(61,1141),(62,1140),(63,1140),(64,1140),
                  (65,1140),(66,1139),(67,1138)]
        rows = [dict(video_seconds=t,clock_seconds=c,period=1,valid=True) for t,c in values]
        intervals = [dict(period=1,clock_seconds=1200,start_video_seconds=0,end_video_seconds=2,observations=3),
                     dict(period=1,clock_seconds=1140,start_video_seconds=62,end_video_seconds=65,observations=4)]
        linked = select_matches(link_boundaries(boundaries, intervals, rows), rows)
        self.assertEqual([e['status'] for e in linked[:3]], ['matched']*3)
        segments = collect_segments(linked, rows)
        # Running-clock OCR gap excludes live stretch; stopped stretch is usable.
        self.assertEqual([s['state'] for s in segments], ['not_live'])
        dense = [dict(video_seconds=t,clock_seconds=1202-t,period=1,valid=True) for t in range(3,63)]
        full = rows[:3] + dense + rows[8:]
        segments = collect_segments(linked, full)
        self.assertEqual([s['state'] for s in segments], ['live','not_live'])
        linked[1]['selected'] = None
        self.assertEqual(collect_segments(linked, full), [])

    def test_ambiguous_supported_restart_is_skipped(self):
        boundary = self.events[0] | dict(state='live')
        intervals = [dict(period=1,clock_seconds=1200,start_video_seconds=t,end_video_seconds=t+2,observations=3) for t in (0,100)]
        rows = [dict(video_seconds=t+offset,clock_seconds=c,period=1,valid=True)
                for t in (0,100) for offset,c in [(3,1199),(4,1198)]]
        result = select_matches(link_boundaries([boundary], intervals, rows), rows)[0]
        self.assertEqual(result['skip_reason'], 'ambiguous_clock')
        self.assertIsNone(result['selected'])

    def test_nearest_clock_and_no_plateau_are_skipped(self):
        boundary = self.events[0] | dict(state='live')
        for clock,end in [(1199,3),(1200,1)]:
            intervals=[dict(period=1,clock_seconds=clock,start_video_seconds=0,end_video_seconds=end,observations=4)]
            linked=select_matches(link_boundaries([boundary],intervals,[]),[])
            self.assertEqual(linked[0]['status'],'skipped')


if __name__ == "__main__": unittest.main()
