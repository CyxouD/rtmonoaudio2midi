import collections
import os
from os.path import isfile, join, splitext, exists

from tabulate import tabulate

from app_setup import WINDOW_SIZE, LOCAL_MAX_WINDOW, LOCAL_MEAN_RANGE_MULTIPLIER, LOCAL_MEAN_THRESHOLD, \
    EXPONENTIAL_DECAY_THRESHOLD_PARAMETER, TUNE_HYPERPARAMETERS, SAMPLE_RATE
from audiostream import StreamProcessor
import xml.etree.ElementTree as ET
import numpy as np
from mingus.midi import fluidsynth
import time

from chart import Chart
from midi import hz_to_midi
from numeric_metrics import NumericMetrics
# from midi import create_midi_file_with_notes, Note, hz_to_midi
from table_metrics import TableMetrics
from scipy import optimize

DELAYS_SECONDS_BETWEEN_PLAYING = 0

PENALTY = 50
MISSED_TO_EXTRA_PENALTY_RATIO = 2 / 1


class Test(object):

    def __init__(self):
        if not TUNE_HYPERPARAMETERS:
            # Single notes are played on each string from the 0th fret (empty string) to the 12th fret.
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC", bitDepth=16,
                                show_chart=False, print_logs=True)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU", bitDepth=16,
                                show_chart=False, print_logs=True)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge+Neck SC",
                                bitDepth=16, show_chart=False, print_logs=True)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU", bitDepth=16,
                                show_chart=False, print_logs=True)

            # monophonic songs
            # TODO include Lick1 but handle it differently from Lick10,
            #  add other Lick3, Lick4, Lick5, Lick6, Lick11, but handle that some annotations are missing
            # (allActualPitchesInfos, allFoundPitchesInfos) = self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
            #                                                                     bitDepth=24,
            #                                                                     filesSubstrings=["AR_Lick2"],
            #                                                                     show_chart=False, print_logs=True)
            self.play_found_and_actual_pitches(allActualPitchesInfos, allFoundPitchesInfos)
            self.show_table(allActualPitchesInfos, allFoundPitchesInfos)

        else:
            objective_function = self.missed_and_extra_and_other_notes_objective
            # objective_function = self.mean_squared_error
            (minResult, results) = self.brute_optimization(objective_function)
            print('minResult', minResult)
            print('results', results)

    # def mean_squared_error(self, Y_pred, Y_real):
    #     summed_all_found_pitches = map(lambda pitches: np.sum(pitches), Y_pred)
    #     summed_all_actual_pitches = map(lambda pitches: np.sum(pitches), Y_real)
    #     return np.square(
    #         np.subtract(summed_all_found_pitches, summed_all_actual_pitches)).mean()

    def brute_optimization(self, objective_function):
        window_sizes = [1024, 2048]
        local_mean_thresholds = np.arange(0, 100001, 5000).tolist()
        local_max_windows = [3]
        local_mean_range_multipliers = [2, 3]
        exponential_decay_thresholds = np.arange(0.0, 1.01, 0.25).tolist()

        total_number_of_experiments = len(window_sizes) * len(local_mean_thresholds) * len(local_max_windows) * len(
            local_mean_range_multipliers) * len(exponential_decay_thresholds)

        results = {}

        min_objective = - 1005000
        min_inputs = None
        experiment_n = 1
        for window_size in window_sizes:
            for local_mean_threshold in local_mean_thresholds:
                for local_max_window in local_max_windows:
                    for local_mean_range_multiplier in local_mean_range_multipliers:
                        for exponential_decay_threshold in exponential_decay_thresholds:
                            print('Experiment #' + str(experiment_n) + ' of ' + str(total_number_of_experiments))
                            inputs = [window_size, local_mean_threshold, local_max_window,
                                      local_mean_range_multiplier,
                                      exponential_decay_threshold]

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
         exponential_decay_threshold) = inputs
        (allActualPitchesInfos, allFoundPitchesInfos) = self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
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
                                                                            show_chart=False)

        return allFoundPitchesInfos, allActualPitchesInfos

    def process_folder(self, folderPath, bitDepth, window_size=WINDOW_SIZE, local_max_window=LOCAL_MAX_WINDOW,
                       local_mean_range_multiplier=LOCAL_MEAN_RANGE_MULTIPLIER,
                       local_mean_threshold=LOCAL_MEAN_THRESHOLD,
                       exponential_decay_threshold=EXPONENTIAL_DECAY_THRESHOLD_PARAMETER, filesSubstrings=None,
                       show_chart=True,
                       print_logs=False):
        print(folderPath)
        path = folderPath + "/annotation/"
        files = []
        for filename in sorted(os.listdir(path)):
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.xml') and (any(
                    filename.find(
                        substring) != -1 for substring in filesSubstrings) if filesSubstrings is not None else True):
                files.append(filename)

        all_found_pitches_infos = []
        all_actual_pitches_infos = []
        for filename in files:
            print(filename)
            filename_without_ext = splitext(filename)[0]

            path_to_wav = os.path.join(folderPath + "/audio/", filename_without_ext + ".wav")

            if not exists(path_to_wav):
                print(path_to_wav + '- not exists')
                continue

            result = StreamProcessor(path_to_wav, bits_per_sample=bitDepth, window_size=window_size,
                                     local_max_window=local_max_window,
                                     local_mean_range_multiplier=local_mean_range_multiplier,
                                     local_mean_threshold=local_mean_threshold,
                                     exponential_decay_threshold_parameter=exponential_decay_threshold).run()
            Pitch_info = collections.namedtuple('Pitch_info',
                                                ['pitch', 'onset_sec'])
            # TODO improve round function
            found_pitches_infos = map(
                lambda info: Pitch_info(self.round_midi(hz_to_midi(info.fundamental_frequency)), info.onset_sec),
                result.fundamental_frequencies_infos)
            all_found_pitches_infos.append(found_pitches_infos)
            tree = ET.parse(os.path.join(path, filename))
            actual_pitches_infos = []
            for event in tree.getroot().find('transcription').findall('event'):
                actual_pitches_infos.append(
                    Pitch_info(pitch=int(event.find('pitch').text),
                               onset_sec=float(event.find('onsetSec').text)))

            all_actual_pitches_infos.append(actual_pitches_infos)

            if show_chart:
                Chart.showSignalAndFlux(result.amplitudes, result.flux_values,
                                        result.window_size, result.onset_flux, result.local_mean_thresholds,
                                        result.exponential_decay_thresholds)

            if print_logs:
                self.print_logs(actual_pitches_infos, found_pitches_infos)

        return all_actual_pitches_infos, all_found_pitches_infos

    def round_midi(self, midi):
        return int(round(midi))

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

    def find_missed_and_extra_and_other_notes(self, found_pitches_infos, actual_pitches_infos, window_size,
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

    def play_found_and_actual_pitches(self, allActualPitchesInfos, allFoundPitchesInfos):
        for (actual_pitches_info, found_pitches_info) in zip(allActualPitchesInfos, allFoundPitchesInfos):
            for pitch in map(lambda info: info.pitch, actual_pitches_info):
                print('Playing actual pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

            for pitch in map(lambda info: info.pitch, found_pitches_info):
                print('Playing found pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            # create_midi_file_with_notes('test', [Note(pitches[0], 100, 0.2, 0.5)] , 140)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

    def show_table(self, allActualPitchesInfos, allFoundPitchesInfos):
        if len(allActualPitchesInfos) != 0:
            TableMetrics.numeric_metrics_in_table(
                map(lambda actual_pitches_infos: map(lambda info: info.pitch, actual_pitches_infos),
                    allActualPitchesInfos),
                map(lambda found_pitches_infos: map(
                    lambda info: info.pitch, found_pitches_infos),
                    allFoundPitchesInfos))
        else:
            print('no actual pitches')

        print("\n" * 10)

    def print_logs(self, actual_pitches_infos, found_pitches_infos):
        found_pitches = map(lambda info: info.pitch, found_pitches_infos)
        print('found = ' + str(found_pitches))
        actual_pitches = map(lambda info: info.pitch, actual_pitches_infos)
        print('actual = ' + str(actual_pitches))
        print('found_pitches_infos', found_pitches_infos)
        print('actual_pitches_infos', actual_pitches_infos)
        found_onsets = map(lambda info: info.onset_sec, found_pitches_infos)
        print('found_onsets', found_onsets)
        actual_onsets = map(lambda info: info.onset_sec, actual_pitches_infos)
        print('actual_onsets = ' + str(actual_onsets))
        print('find_missed_and_extra_and_other_notes',
              self.find_missed_and_extra_and_other_notes(found_pitches_infos, actual_pitches_infos,
                                                         WINDOW_SIZE,
                                                         SAMPLE_RATE))

if __name__ == '__main__':
    test = Test()
