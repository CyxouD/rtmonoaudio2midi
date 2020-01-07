import collections
import os
from os.path import splitext, exists

from app_setup import WINDOW_SIZE, LOCAL_MAX_WINDOW, LOCAL_MEAN_RANGE_MULTIPLIER, LOCAL_MEAN_THRESHOLD, \
    EXPONENTIAL_DECAY_THRESHOLD_PARAMETER, SAMPLE_RATE, SPECTRAL_FLUX_NORM_LEVEL
from audiostream import StreamProcessor
import xml.etree.ElementTree as ET

from chart import Chart
from midi import hz_to_midi


def process_folder(folderPath, bitDepth, window_size=WINDOW_SIZE, local_max_window=LOCAL_MAX_WINDOW,
                   local_mean_range_multiplier=LOCAL_MEAN_RANGE_MULTIPLIER,
                   local_mean_threshold=LOCAL_MEAN_THRESHOLD,
                   exponential_decay_threshold=EXPONENTIAL_DECAY_THRESHOLD_PARAMETER,
                   spectral_flux_norm_level=SPECTRAL_FLUX_NORM_LEVEL, filesSubstrings=None,
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
                                 exponential_decay_threshold_parameter=exponential_decay_threshold,
                                 spectral_flux_norm_level=spectral_flux_norm_level).run()
        Pitch_info = collections.namedtuple('Pitch_info',
                                            ['pitch', 'onset_sec'])
        # TODO improve round function
        found_pitches_infos = map(
            lambda info: Pitch_info(round_midi(hz_to_midi(info.fundamental_frequency)), info.onset_sec),
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
            print_info_logs(actual_pitches_infos, found_pitches_infos)

    return all_actual_pitches_infos, all_found_pitches_infos


def print_info_logs(actual_pitches_infos, found_pitches_infos):
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
    from tuning_hyperparameters import TuningHyperparameters
    print('find_missed_and_extra_and_other_notes',
          TuningHyperparameters.find_missed_and_extra_and_other_notes(found_pitches_infos, actual_pitches_infos,
                                                                      WINDOW_SIZE,
                                                                      SAMPLE_RATE))


def round_midi(midi):
    return int(round(midi))
