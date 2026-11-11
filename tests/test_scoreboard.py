import unittest
from studio.scoreboard.align import align
from studio.scoreboard.readings import parse_reading


def row(t, remaining, period=1, valid=True):
    return dict(video_seconds=t, clock_seconds=remaining, period=period, valid=valid)


class AlignmentTests(unittest.TestCase):
    def test_stoppage_and_replay_remain_multiple_candidates(self):
        observations = [row(0, 1100), row(.5, 1100), row(1, 1100),
                        row(2, 1120), row(3, 1100), row(3.5, 1100)]
        result = align(observations, [dict(period_number=1, time_in_period='01:40')], [])
        self.assertEqual(result['events'][0]['video_intervals'], [[0, .5], [.5, 1], [3, 3.5]])
        self.assertEqual(result['status'], 'needs_review')

    def test_missing_clock_does_not_bridge_gap(self):
        result = align([row(0, 1101), row(.5, 0, valid=False), row(1, 1100)],
                       [dict(period_number=1, time_in_period='01:40')], [])
        self.assertEqual(result['events'][0]['status'], 'unmatched')

    def test_regular_and_playoff_overtime(self):
        events = [dict(period_number=4, time_in_period='00:01')]
        for playoff, length in [(False, 300), (True, 1200)]:
            result = align([row(0, length, 4), row(1, length-1, 4)], events, [], playoff)
            self.assertEqual(result['events'][0]['video_intervals'], [[0, 1]])

    def test_shift_end_is_exclusive_at_stopped_clock(self):
        shifts = [dict(period=1, start_time=90, end_time=100),
                  dict(period=1, start_time=100, end_time=110)]
        result = align([row(0, 1100), row(.5, 1100)], [], shifts)
        self.assertEqual(result['shifts'][0]['video_intervals'], [])
        self.assertEqual(result['shifts'][1]['video_intervals'], [[0, .5]])

    def test_unreadable_and_low_confidence_clock(self):
        self.assertFalse(parse_reading('12:99', '1ST', .99, .9)[2])
        self.assertFalse(parse_reading('12:00', '1ST', .2, .9)[2])

    def test_boundary_order_and_bidirectional_links(self):
        events = [dict(period_number=1, time_in_period='00:44', type_desc_key='faceoff', sort_order=22),
                  dict(period_number=1, time_in_period='00:44', type_desc_key='stoppage', sort_order=20)]
        shifts = [dict(period=1, start_time=40, end_time=50)]
        result = align([row(0, 1156), row(.5, 1156), row(1, 1155)], events, shifts)
        self.assertEqual(result['boundary_groups'][0]['event_indexes'], [1, 0])
        self.assertEqual(result['events'][0]['timing_role'], 'restart')
        self.assertEqual(result['events'][1]['timing_role'], 'stop')
        for i in result['events'][0]['interval_ids']:
            self.assertIn(0, result['clock_intervals'][i]['event_indexes'])
        self.assertEqual(result['clock_intervals'][0]['shift_indexes'], [0])

    def test_replay_has_separate_occurrences(self):
        result = align([row(0, 1101), row(.5, 1100), row(1, 1100),
                        row(2, 1101), row(2.5, 1100)],
                       [dict(period_number=1, time_in_period='01:40')], [])
        self.assertEqual(result['events'][0]['status'], 'ambiguous')
        self.assertEqual(len(result['events'][0]['occurrences']), 2)
        self.assertEqual(result['gaps'][0]['reason'], 'clock_reversal_possible_replay')

    def test_full_video_gaps_and_period_transition(self):
        result = align([row(2, 1), row(2.5, 0), row(10, 1200, 2), row(10.5, 1200, 2)], [], [], duration=12)
        self.assertEqual([g['video'] for g in result['gaps']], [[0, 2], [2.5, 10], [10.5, 12]])
        self.assertEqual(result['gaps'][1]['reason'], 'period_transition')

    def test_no_running_segment_before_shift_start(self):
        result = align([row(0, 1101), row(.5, 1100)], [], [dict(period=1, start_time=100, end_time=120)])
        self.assertEqual(result['shifts'][0]['status'], 'unmatched')

    def test_multiple_overtime_and_shootout(self):
        self.assertEqual(parse_reading('19:59', '2OT', .99, .9), (1199, 5, True))
        shootout = [{**row(0, 1200, 5), 'period_raw': 'SO'}, {**row(.5, 1200, 5), 'period_raw': 'SO'}]
        self.assertEqual(align(shootout, [], [], playoff=True)['clock_intervals'], [])

    def test_shift_chart_goal_marker_is_not_on_ice_evidence(self):
        marker = dict(period=1, start_time=100, end_time=100, type_code=505)
        result = align([row(0, 1100), row(.5, 1100)], [], [marker])
        self.assertEqual(result['shifts'][0]['status'], 'candidate')
        self.assertEqual(result['shifts'][0]['record_kind'], 'point_marker')
        self.assertEqual(result['clock_intervals'][0]['shift_indexes'], [])
        self.assertEqual(result['clock_intervals'][0]['marker_indexes'], [0])


if __name__ == '__main__':
    unittest.main()
