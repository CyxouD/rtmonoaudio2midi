from app_setup import SAMPLE_RATE
import numpy as np

from process_folder import process_folder

DELAYS_SECONDS_BETWEEN_PLAYING = 0

PENALTY = 50
MISSED_TO_EXTRA_PENALTY_RATIO = 2 / 1


class TuningHyperparameters(object):

    def __init__(self):
        (min_combine_results, combine_results) = self.combine_optimization_results()
        print('min_combine_results', min_combine_results)
        print('combined results', combine_results)

    def combine_optimization_results(self):
        (missed_and_extra_and_other_notes_objective_minResult,
         missed_and_extra_and_other_notes_objective_results) = self.brute_optimization(
            self.missed_and_extra_and_other_notes_objective)
        (other_notes_difference_objective_minResult,
         other_notes_difference_objective_results) = self.brute_optimization(
            self.other_notes_difference_objective)
        mapped_missed_and_extra_and_other_notes_objective_results = self.map_results_from_zero_to_one(
            missed_and_extra_and_other_notes_objective_results)
        mapped_other_notes_difference_objective_results = self.map_results_from_zero_to_one(
            other_notes_difference_objective_results)

        combine_results = {
            k: [missed_and_extra_and_other_notes_objective_results[k], other_notes_difference_objective_results[k],
                mapped_missed_and_extra_and_other_notes_objective_results[k],
                mapped_other_notes_difference_objective_results[k],
                v + mapped_other_notes_difference_objective_results[k]] for k, v in
            mapped_missed_and_extra_and_other_notes_objective_results.items()}
        min_combine_results = min(combine_results.items(), key=lambda x: x[1]) # TODO fix incorrect getting of min
        return min_combine_results, combine_results

    def map_results_from_zero_to_one(self, results):
        return {k: ((v - min(results.values())) / (max(results.values()) - min(results.values()))) for k, v in
                results.items()}

    def brute_optimization(self, objective_function):
        window_sizes = [1024, 2048]
        local_mean_thresholds = np.arange(0, 100001, 5000).tolist()
        local_max_windows = [3]
        local_mean_range_multipliers = [2, 3]
        exponential_decay_thresholds = np.arange(0.0, 1.01, 0.25).tolist()
        spectral_flux_norm_levels = [1, 2]

        total_number_of_experiments = len(window_sizes) * len(local_mean_thresholds) * len(local_max_windows) * len(
            local_mean_range_multipliers) * len(exponential_decay_thresholds) * len(spectral_flux_norm_levels)

        results = {}

        min_objective = - 1005000
        min_inputs = None
        experiment_n = 1
        for window_size in window_sizes:
            for local_mean_threshold in local_mean_thresholds:
                for local_max_window in local_max_windows:
                    for local_mean_range_multiplier in local_mean_range_multipliers:
                        for exponential_decay_threshold in exponential_decay_thresholds:
                            for spectral_flux_norm_level in spectral_flux_norm_levels:
                                print('Experiment #' + str(experiment_n) + ' of ' + str(total_number_of_experiments))
                                inputs = [window_size, local_mean_threshold, local_max_window,
                                          local_mean_range_multiplier,
                                          exponential_decay_threshold, spectral_flux_norm_level]

                                result = self.tuning_function(
                                    inputs)
                                result_objective = objective_function(result[0], result[1], window_size, SAMPLE_RATE)
                                if min_inputs is None:
                                    min_inputs = inputs
                                    min_objective = result_objective
                                results[str(inputs)] = result_objective

                                if result_objective <= min_objective:
                                    min_objective = result_objective
                                    min_inputs = inputs
                                print('results', results)
                                experiment_n += 1

        return (min_inputs, min_objective), results

    def tuning_function(self, inputs, *params):
        (window_size, local_mean_threshold, local_max_window, local_mean_range_multiplier,
         exponential_decay_threshold, spectral_flux_norm_level) = inputs
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
                                                                       bitDepth=24,
                                                                       window_size=window_size,
                                                                       local_max_window=local_max_window,
                                                                       local_mean_range_multiplier=
                                                                       local_mean_range_multiplier,
                                                                       local_mean_threshold=local_mean_threshold,
                                                                       exponential_decay_threshold=exponential_decay_threshold,
                                                                       filesSubstrings=['Lick1_', 'Lick11_',
                                                                                        'Lick3_', 'Lick4_',
                                                                                        'Lick5_',
                                                                                        'Lick6_', "Lick2_"],
                                                                       spectral_flux_norm_level=spectral_flux_norm_level,
                                                                       show_chart=False)

        return allFoundPitchesInfos, allActualPitchesInfos

    def other_notes_difference_objective(self, allFoundPitchesInfos, allActualPitchesInfos, window_size,
                                         sample_rate):
        penalties = []
        for (actual_pitches_infos, found_pitches_infos) in zip(allFoundPitchesInfos, allActualPitchesInfos):
            _, _, (_, other_notes_pairs) = self.find_missed_and_extra_and_other_notes(
                found_pitches_infos,
                actual_pitches_infos,
                window_size,
                sample_rate)
            penalty = sum(map(lambda pair: abs(pair[0] - pair[1]), other_notes_pairs))
            penalties.append(penalty)
        return np.average(penalties)

    def missed_and_extra_and_other_notes_objective(self, allFoundPitchesInfos, allActualPitchesInfos, window_size,
                                                   sample_rate):
        penalties = []
        for (actual_pitches_infos, found_pitches_infos) in zip(allFoundPitchesInfos, allActualPitchesInfos):
            actual_onsets = list(map(lambda info: info.onset_sec, actual_pitches_infos))
            missed_notes_number, extra_notes_number, (
                other_notes_number, _) = self.find_missed_and_extra_and_other_notes(
                found_pitches_infos,
                actual_pitches_infos,
                window_size,
                sample_rate)
            mistakes_penalty = missed_notes_number * MISSED_TO_EXTRA_PENALTY_RATIO + extra_notes_number + other_notes_number
            penalty = mistakes_penalty / len(actual_onsets)
            penalties.append(penalty)
        return np.average(penalties)

    def missed_and_extra_notes_objective(self, allFoundPitchesInfos, allActualPitchesInfos, window_size,
                                         sample_rate):
        penalties = []
        for (actual_pitches_infos, found_pitches_infos) in zip(allFoundPitchesInfos, allActualPitchesInfos):
            actual_onsets = list(map(lambda info: info.onset_sec, actual_pitches_infos))
            missed_notes_number, extra_notes_number, _ = self.find_missed_and_extra_and_other_notes(
                found_pitches_infos,
                actual_pitches_infos,
                window_size,
                sample_rate)
            penalty = (missed_notes_number * MISSED_TO_EXTRA_PENALTY_RATIO + extra_notes_number) / len(
                actual_onsets)
            penalties.append(penalty)
        return np.average(penalties)

    @staticmethod
    def find_missed_and_extra_and_other_notes(found_pitches_infos, actual_pitches_infos, window_size,
                                              sample_rate):
        found_onsets = list(map(lambda info: info.onset_sec, found_pitches_infos))
        actual_onsets = list(map(lambda info: info.onset_sec, actual_pitches_infos))

        window_size_in_sec = float(window_size) / sample_rate
        onsets_equals = 0
        other_notes_number = 0
        other_notes_pairs = []
        m = 2

        min_distance = 100500
        for i in range(0, len(actual_onsets) - 1):
            pair_distance = actual_onsets[i + 1] - actual_onsets[i]
            if pair_distance < min_distance:
                min_distance = pair_distance

        # step = window_size_in_sec * m
        step = min_distance / 2
        associated_found_onsets = []
        for (actual_pitch, actual_onset) in actual_pitches_infos:
            for (found_pitch, found_onset) in found_pitches_infos:
                if found_onset - step <= actual_onset <= found_onset + step and found_onset not in associated_found_onsets:
                    if found_pitch != actual_pitch:
                        other_notes_number += 1
                        other_notes_pairs.append((actual_pitch, found_pitch))
                    onsets_equals = onsets_equals + 1
                    associated_found_onsets.append(found_onset)
                    break  # TODO should we stop if we found note in interval, what if more than 1 is found?

        missed_notes_number = len(actual_onsets) - onsets_equals
        extra_notes_number = len(found_onsets) - onsets_equals
        print('associated_found_onsets', associated_found_onsets)
        print('min actual distance', min_distance)
        print('window_size_in_sec', window_size_in_sec)
        print('window_size_in_sec * m', step)
        print('actual size', len(actual_onsets))
        print('found size', len(found_onsets))
        print('onsets equals', onsets_equals)
        print('other pairs', other_notes_pairs)
        print('other_notes_number', other_notes_number)
        print('missed onsets', missed_notes_number)  # missed
        print('extra_onsets', extra_notes_number)  # extra

        return missed_notes_number, extra_notes_number, (other_notes_number, other_notes_pairs)


if __name__ == '__main__':
    test = TuningHyperparameters()
