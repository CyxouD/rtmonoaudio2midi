import collections
import os
from os.path import isfile, join, splitext, exists

from tabulate import tabulate

from app_setup import WINDOW_SIZE, LOCAL_MAX_WINDOW, LOCAL_MEAN_RANGE_MULTIPLIER, LOCAL_MEAN_THRESHOLD, \
    EXPONENTIAL_DECAY_THRESHOLD_PARAMETER, TUNE_HYPERPARAMETERS
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


class Test(object):

    def __init__(self):
        if not TUNE_HYPERPARAMETERS:
            # Single notes are played on each string from the 0th fret (empty string) to the 12th fret.
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC", bitDepth=16)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU", bitDepth=16)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge+Neck SC",
                                bitDepth=16)
            self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU", bitDepth=16)

            # monophonic songs
            # TODO include Lick1 but handle it differently from Lick10,
            #  add other Lick3, Lick4, Lick5, Lick6, Lick11, but handle that some annotations are missing
            # self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
            #                     bitDepth=24,
            #                     filesSubstrings=['AR_Lick2_FN'])
        else:
            # rranges = ((1024, 2048), slice(5000, 50000, 5000), slice(0, 1, 0.1))
            # rranges = (slice(1024.0, 2048.0, 512.0), slice(5000.0, 10000.0, 5000.0))
            # params = (3, 3)
            # params = (3, 3, 0.3)
            # (resbrute, some, grid, jout) = optimize.brute(self.f, rranges, args=params, full_output=True)
            # print('resbrute', resbrute)
            # print(tabulate(grid))

            (minResult, results) = self.brute_optimization(self.mean_squared_error)
            print('minResult', minResult)
            print('results', results)

    def mean_squared_error(self, Y_pred, Y_real):
        return np.square(
            np.subtract(Y_pred, Y_real)).mean()

    def brute_optimization(self, objective_function):
        window_sizes = [1024]
        local_mean_thresholds = np.arange(0, 100000, 5000).tolist()
        local_max_windows = [3]
        local_mean_range_multipliers = [3]
        exponential_decay_thresholds = np.arange(0.0, 1.0, 0.24).tolist()

        results = {}

        min_objective = - 1005000
        min_inputs = None
        for window_size in window_sizes:
            for local_mean_threshold in local_mean_thresholds:
                for local_max_window in local_max_windows:
                    for local_mean_range_multiplier in local_mean_range_multipliers:
                        for exponential_decay_threshold in exponential_decay_thresholds:
                            inputs = [window_size, local_mean_threshold, local_max_window,
                                      local_mean_range_multiplier,
                                      exponential_decay_threshold]

                            result = self.tuning_function(
                                inputs)
                            result_objective = objective_function(result[0], result[1])
                            if min_inputs is None:
                                min_inputs = inputs
                                min_objective = result_objective
                            results[str(inputs)] = result_objective
                            print('results', results)

                            if result_objective <= min_objective:
                                min_objective = result_objective
                                min_inputs = inputs

        return (min_inputs, min_objective), results

    def tuning_function(self, inputs, *params):
        (window_size, local_mean_threshold, local_max_window, local_mean_range_multiplier,
         exponential_decay_threshold) = inputs
        (allActualPitches, allFoundPitches) = self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
                                                                  bitDepth=24,
                                                                  window_size=window_size,
                                                                  local_max_window=local_max_window,
                                                                  local_mean_range_multiplier=
                                                                  local_mean_range_multiplier,
                                                                  local_mean_threshold=local_mean_threshold,
                                                                  exponential_decay_threshold=exponential_decay_threshold,
                                                                  # TODO exclude Lick12 from here
                                                                  filesSubstrings=['Lick1', 'Lick3', 'Lick4', 'Lick5',
                                                                                   'Lick6'],
                                                                  show_chart=False)

        summed_all_actual_pitches = map(lambda pitches: np.sum(pitches), allActualPitches)
        summed_all_found_pitches = map(lambda pitches: np.sum(pitches), allFoundPitches)

        return summed_all_found_pitches, summed_all_actual_pitches

    def process_folder(self, folderPath, bitDepth, window_size=WINDOW_SIZE, local_max_window=LOCAL_MAX_WINDOW,
                       local_mean_range_multiplier=LOCAL_MEAN_RANGE_MULTIPLIER,
                       local_mean_threshold=LOCAL_MEAN_THRESHOLD,
                       exponential_decay_threshold=EXPONENTIAL_DECAY_THRESHOLD_PARAMETER, filesSubstrings=None,
                       show_chart=True):
        print(folderPath)
        path = folderPath + "/annotation/"
        files = []
        for filename in sorted(os.listdir(path)):
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.xml') and (any(
                    filename.find(
                        substring) != -1 for substring in filesSubstrings) if filesSubstrings is not None else True):
                files.append(filename)

        all_found_pitches = []
        all_actual_pitches = []
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
            # TODO improve round function
            found_fundamental_frequencies = map(lambda info: info.fundamental_frequency,
                                                result.fundamental_frequencies_infos)
            found_onsets = map(lambda info: info.onset_sec, result.fundamental_frequencies_infos)
            print('found_onsets', found_onsets)
            found_pitches = map(lambda midi: int(round(midi)), list(hz_to_midi(found_fundamental_frequencies)))
            all_found_pitches.append(found_pitches)
            print('found = ' + str(found_pitches))
            tree = ET.parse(os.path.join(path, filename))
            actual_pitches_infos = []
            for event in tree.getroot().find('transcription').findall('event'):
                Pitch_info = collections.namedtuple('Pitch_info',
                                                    ['pitch', 'onset_sec'])
                actual_pitches_infos.append(
                    Pitch_info(pitch=int(event.find('pitch').text), onset_sec=float(event.find('onsetSec').text)))
            actual_onsets = map(lambda info: info.onset_sec, actual_pitches_infos)
            print('actual_onsets = ' + str(actual_onsets))
            actual_pitches = map(lambda info: info.pitch, actual_pitches_infos)
            print('actual = ' + str(actual_pitches))

            all_actual_pitches.append(actual_pitches)

            for pitch in actual_pitches:
                print('Playing actual pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

            for pitch in found_pitches:
                print('Playing found pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            # create_midi_file_with_notes('test', [Note(pitches[0], 100, 0.2, 0.5)] , 140)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

            if show_chart:
                Chart.showSignalAndFlux(result.amplitudes, result.flux_values,
                                        result.window_size, result.onset_flux, result.local_mean_thresholds,
                                        result.exponential_decay_thresholds)

        if len(all_actual_pitches) != 0:
            TableMetrics.numeric_metrics_in_table(all_actual_pitches, all_found_pitches)
        else:
            print('no actual pitches')

        print("\n" * 10)

        return all_actual_pitches, all_found_pitches


if __name__ == '__main__':
    test = Test()
