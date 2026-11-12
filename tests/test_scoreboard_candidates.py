import unittest

from studio.scoreboard.align import align


def row(t, elapsed, period=1, valid=True):
    return dict(video_seconds=t, clock_seconds=1200-elapsed, period=period, valid=valid)


def event(elapsed, period=1):
    return dict(period_number=period, time_in_period=f'{elapsed//60:02d}:{elapsed%60:02d}')


class CandidateTests(unittest.TestCase):
    def test_isolated_reading_keeps_exact_clock_and_no_interval(self):
        result = align([row(10, 100)], [event(100)], [])
        match = result['events'][0]
        self.assertEqual(match['status'], 'candidate')
        self.assertEqual(match['clock_status'], 'unmatched')
        self.assertEqual(match['review_candidates'][0]['basis'], 'exact_reading')
        self.assertEqual(match['interval_ids'], [])

    def test_fractional_clock_preserves_error(self):
        result = align([row(10, 100.2)], [event(100)], [])
        candidate = result['events'][0]['review_candidates'][0]
        self.assertEqual(candidate['basis'], 'nearby_clock')
        self.assertAlmostEqual(candidate['clock_error_seconds'], .2)

    def test_gap_returns_bounds_without_interpolating_or_on_ice(self):
        result = align([row(10, 100), row(20, 0, valid=False), row(40, 110)],
                       [event(105)], [dict(period=1, start_time=103, end_time=108)])
        for match in [result['events'][0], result['shifts'][0]]:
            self.assertEqual(match['review_candidates'][0]['video'], [10, 40])
            self.assertEqual(match['review_candidates'][0]['basis'], 'bracketed_clock')
            self.assertEqual(match['occurrences'], [])
        self.assertEqual(result['clock_intervals'], [])

    def test_distant_gap_reversed_clock_and_other_period_do_not_bridge(self):
        cases = [[row(10, 100), row(200, 150)],
                 [row(10, 150), row(40, 100)],
                 [row(10, 100), row(11, 0, 2), row(40, 150)]]
        for rows in cases:
            self.assertEqual(align(rows, [event(125)], [])['events'][0]['status'], 'unmatched')

    def test_repeated_singletons_remain_ambiguous(self):
        result = align([row(10, 100), row(50, 100)], [event(100)], [])
        self.assertEqual(result['events'][0]['status'], 'ambiguous')
        self.assertEqual(len(result['events'][0]['review_candidates']), 2)

    def test_period_conflict_quarantines_strict_links_and_proposes_context(self):
        rows = [row(t, 1200, 2) for t in (0, .5, 1)]
        rows += [row(100+t, t, 2) for t in range(100)]
        rows += [row(200+t, 100+t, 3) for t in range(3)]
        result = align(rows, [event(50, 3), event(50, 2)], [])
        self.assertEqual(len(result['period_conflicts']), 1)
        candidate = result['events'][0]['review_candidates'][0]
        self.assertTrue(candidate['period_context'])
        self.assertEqual(candidate['observed_period'], 2)
        self.assertEqual(candidate['candidate_period'], 3)
        self.assertEqual(result['events'][0]['interval_ids'], [])
        self.assertEqual(result['events'][1]['status'], 'unmatched')
        self.assertEqual(rows[53]['period'], 2)

    def test_reset_without_following_period_does_not_infer_period(self):
        rows = [row(t, 1200, 2) for t in (0, .5, 1)]
        rows += [row(100+t, t, 2) for t in range(100)]
        self.assertEqual(align(rows, [event(50, 3)], [])['period_conflicts'], [])

    def test_no_shootout_or_zero_duration_shift_inference(self):
        result = align([row(10, 100)], [{**event(100), 'period_type': 'SO'}],
                       [dict(period=1, start_time=100, end_time=100, type_code=517)])
        self.assertEqual(result['events'][0]['status'], 'unmatched')
        self.assertEqual(result['shifts'][0]['status'], 'unmatched')


if __name__ == '__main__':
    unittest.main()
