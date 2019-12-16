import numpy as np
import unittest


class Metrics:
    @staticmethod
    def containsAnyActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for index, actual in enumerate(actual_pitches):
            if any(elem in actual for elem in found_pitches[index]):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)

    @staticmethod
    def containsAllActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for index, found in enumerate(found_pitches):
            if all(elem in found for elem in actual_pitches[index]):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)

    @staticmethod
    def totalEqualityActualRatio(actual_pitches, found_pitches):
        contains_n = 0
        for index, found in enumerate(found_pitches):
            if np.array_equal(actual_pitches[index], found):
                contains_n = contains_n + 1
        return float(contains_n) / len(actual_pitches)