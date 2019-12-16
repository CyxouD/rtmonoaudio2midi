from numeric_metrics import NumericMetrics
from tabulate import tabulate


class TableMetrics:
    @staticmethod
    def numeric_metrics_in_table(actual_pitches, found_pitches):
        footer = ['Total', '', NumericMetrics.containsAnyActualRatio(actual_pitches, found_pitches),
                  NumericMetrics.containsAllActualRatio(actual_pitches, found_pitches),
                  NumericMetrics.totalEqualityActualRatio(actual_pitches, found_pitches)]
        data = list(map(lambda actual_found: [actual_found[0], actual_found[1],
                                              '+' if NumericMetrics.containsAnyActualRatio(
                                                  [actual_found[0]], [actual_found[1]]) != 0 else '',
                                              '+' if NumericMetrics.containsAllActualRatio(
                                                  [actual_found[0]], [actual_found[1]]) != 0 else '',
                                              '+' if NumericMetrics.totalEqualityActualRatio(
                                                  [actual_found[0]], [actual_found[1]]) != 0 else ''
                                              ],
                        zip(actual_pitches, found_pitches)))
        data.extend([footer])
        print(tabulate(data, headers=['Actual', 'Found', 'Any', 'Total', 'All']))


if __name__ == '__main__':
    actual_pitches = [[45], [46, 47]]
    found_pitches = [[45, 45], [46, 48]]
    TableMetrics.numeric_metrics_in_table(actual_pitches, found_pitches)
