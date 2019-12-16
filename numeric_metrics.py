import numpy as np
import unittest


class NumericMetrics:

    @staticmethod
    def containsAnyActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for actual, found in zip(actual_pitches, found_pitches):
            if any(elem in actual for elem in found):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)

    @staticmethod
    def containsAllActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for actual, found in zip(actual_pitches, found_pitches):
            if all(elem in found for elem in actual):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)

    @staticmethod
    def totalEqualityActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for actual, found in zip(actual_pitches, found_pitches):
            if np.array_equal(actual, found):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)
