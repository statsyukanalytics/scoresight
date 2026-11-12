import unittest
from unittest.mock import Mock

import numpy as np

from studio.scoreboard.readings import parse_reading
from studio.scoreboard.auto_fields import AutoReader, EMPTY, choose_pair


class AutomaticFieldsTests(unittest.TestCase):
    def test_ocr_assigns_clock_and_period_without_detector_role_labels(self):
        candidates = [((10, 10, 70, 25), ('7:34', .99)),
                      ((10, 40, 50, 25), ('3RD', .99)),
                      ((100, 10, 60, 25), ('OTT2', .99))]
        result = choose_pair(candidates)
        self.assertEqual(result[0], [candidates[0][0], candidates[1][0]])

    def test_penalty_clock_competition_is_unresolved(self):
        candidates = [((10, 10, 70, 25), ('7:34', .99)),
                      ((10, 40, 50, 25), ('3RD', .99)),
                      ((100, 10, 70, 25), ('1:12', .99))]
        self.assertIsNone(choose_pair(candidates))

    def test_missing_or_low_confidence_period_is_unresolved(self):
        self.assertIsNone(choose_pair([((10, 10, 70, 25), ('7:34', .99))]))
        self.assertIsNone(choose_pair([((10, 10, 70, 25), ('7:34', .99)),
                                       ((10, 40, 50, 25), ('3RD', .3))]))

    def test_disappearing_scoreboard_does_not_reuse_cached_readings(self):
        reader = AutoReader(Mock(), Mock())
        reader.fields = [(0, 0, 10, 10), (20, 0, 10, 10)]
        reader.until_search = 15
        reader.recognizer.recognize.return_value = [('', 0), ('', 0)]
        reader._search = Mock(return_value=None)
        reading, fields = reader.read(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(reading, EMPTY)
        self.assertIsNone(fields)
        reader._search.assert_not_called()

    def test_batch_matches_sequential_across_missing_and_moved_boards(self):
        def make_reader():
            reader = AutoReader(Mock(), Mock())
            first = [(0, 0, 2, 2), (2, 0, 2, 2)]
            moved = [(4, 0, 2, 2), (6, 0, 2, 2)]
            def recognize(crops, whitelists=None):
                return [(('10:00' if i % 2 == 0 else '1ST'), .99)
                        if crop[0, 0, 0] else ('', 0) for i, crop in enumerate(crops)]
            reader.recognizer.recognize.side_effect = recognize
            def search(image):
                fields = first if image[0, 0, 0] else moved if image[0, 4, 0] else None
                return (fields, (('10:00', .99), ('1ST', .99))) if fields else None
            reader._search = Mock(side_effect=search)
            return reader
        images = []
        for i in range(60):
            frame = np.zeros((4, 8, 3), dtype=np.uint8)
            if i < 9:
                frame[:, :4] = 1
            elif i >= 14:
                frame[:, 4:] = 1
            images.append(frame)
        sequential, batched = make_reader(), make_reader()
        expected = [sequential.read(frame) for frame in images]
        actual = []
        for start in range(0, len(images), 16):
            actual.extend(batched.read_many(images[start:start+16]))
        self.assertEqual(actual, expected)
        self.assertEqual(batched.until_search, sequential.until_search)
        self.assertLess(batched.recognizer.recognize.call_count, sequential.recognizer.recognize.call_count)

    def test_first_period_ordinal_aliases_do_not_relax_clock_or_confidence(self):
        for period in ('1ST', 'IST', 'lST', 'i st'):
            self.assertEqual(parse_reading('16:58', period, .96, .9), (1018, 1, True))
            self.assertFalse(parse_reading('16:58', period, .89, .9)[2])
        for period in ('I', 'FIRST', '1', '1ND', 'TSN'):
            self.assertFalse(parse_reading('16:58', period, .99, .9)[2])
        self.assertFalse(parse_reading('I6:58', 'IST', .99, .9)[2])

    def test_inset_recovers_same_period_but_rejects_conflicting_period(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        for alternative, expected in [(('IST', .96), ('IST', .96)),
                                      (('2ND', .99), ('IST', .85)),
                                      (('IST', .70), ('IST', .85))]:
            reader = AutoReader(Mock(), Mock())
            reader.recognizer.recognize.side_effect = [[('16:58', .99), ('IST', .85)], [alternative]]
            readings = reader._ocr(image, [(0, 0, 40, 20), (40, 0, 30, 20)])
            self.assertEqual(readings, [('16:58', .99), expected])

    def test_known_geometry_recovers_after_missing_frame_then_relocates_on_schedule(self):
        reader = AutoReader(Mock(), Mock())
        reader.fields = [(0, 0, 10, 10), (20, 0, 10, 10)]
        reader.until_search = 3
        reader.recognizer.recognize.side_effect = [[('', 0), ('', 0)],
                                                 [('16:58', .99), ('IST', .97)],
                                                 [('', 0), ('', 0)]]
        reader._search = Mock(return_value=None)
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        self.assertEqual(reader.read(image), (EMPTY, None))
        self.assertEqual(reader.read(image)[0], (('16:58', .99), ('IST', .97)))
        reader._search.assert_not_called()
        self.assertEqual(reader.read(image), (EMPTY, None))
        reader._search.assert_called_once()


if __name__ == '__main__':
    unittest.main()
