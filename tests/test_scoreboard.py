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


if __name__ == '__main__':
    unittest.main()
