import unittest
from unittest.mock import Mock

import numpy as np

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
        reader._search.assert_called_once()


if __name__ == '__main__':
    unittest.main()
