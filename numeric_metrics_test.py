import unittest

from numeric_metrics import NumericMetrics


class TestStringMethods(unittest.TestCase):

    def test1(self):
        actual_pitches = [[5]]
        found_pitches = [[5, 7]]
        self.assertEqual(NumericMetrics.containsAnyActualRatio(actual_pitches,
                                                               found_pitches), 1)
        self.assertEqual(NumericMetrics.containsAnyActualRatio(actual_pitches,
                                                               found_pitches), 1)
        self.assertEqual(NumericMetrics.containsAllActualRatio(actual_pitches,
                                                               found_pitches), 1)
        self.assertEqual(NumericMetrics.totalEqualityActualRatio(actual_pitches,
                                                                 found_pitches), 0)

    def test2(self):
        actual_pitches = [[6]]
        found_pitches = [[5, 7]]
        self.assertEqual((NumericMetrics.containsAnyActualRatio(actual_pitches,
                                                                found_pitches)), 0)
        self.assertEqual(NumericMetrics.containsAllActualRatio(actual_pitches,
                                                               found_pitches), 0)
        self.assertEqual(NumericMetrics.totalEqualityActualRatio(actual_pitches,
                                                                 found_pitches), 0)

    def test3(self):
        actual_pitches = [[5, 6]]
        found_pitches = [[5, 7]]
        self.assertEqual(NumericMetrics.containsAnyActualRatio(actual_pitches,
                                                               found_pitches), 1)
        self.assertEqual(NumericMetrics.containsAllActualRatio(actual_pitches,
                                                               found_pitches), 0)
        self.assertEqual((NumericMetrics.totalEqualityActualRatio(actual_pitches,
                                                                  found_pitches)), 0)

    def test4(self):
        actual_pitches = [[45]]
        found_pitches = [[45, 45]]
        self.assertEqual(NumericMetrics.containsAnyActualRatio(actual_pitches, found_pitches), 1)
        self.assertEqual(NumericMetrics.containsAllActualRatio(actual_pitches, found_pitches), 1)
        self.assertEqual(NumericMetrics.totalEqualityActualRatio(actual_pitches, found_pitches), 0)

    def test5(self):
        actual_pitches = [[45], [46, 47]]
        found_pitches = [[45, 45], [46, 48]]
        self.assertEqual(NumericMetrics.containsAnyActualRatio(actual_pitches, found_pitches), 1)
        self.assertEqual(NumericMetrics.containsAllActualRatio(actual_pitches, found_pitches), 0.5)
        self.assertEqual(NumericMetrics.totalEqualityActualRatio(actual_pitches, found_pitches), 0)


if __name__ == '__main__':
    unittest.main()
