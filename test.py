import os
from os.path import isfile, join, splitext
from audiostream import StreamProcessor
import xml.etree.ElementTree as ET
import numpy as np
from mingus.midi import fluidsynth
import time

from chart import Chart
from numeric_metrics import NumericMetrics
from midi import create_midi_file_with_notes, Note
from table_metrics import TableMetrics

DELAYS_SECONDS_BETWEEN_PLAYING = 2;


class Test(object):

    def __init__(self):
        # Single notes are played on each string from the 0th fret (empty string) to the 12th fret.
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge+Neck SC")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU")

        # monophonic songs
        # TODO include Lick1 but handle it differently from Lick10,
        #  add other Lick3, Lick4, Lick5, Lick6, Lick11, but handle that some annotations are missing
        # self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
        #                     ['AR_Lick2_FN'])

    def process_folder(self, folderPath, filesSubstrings=None):
        print(folderPath)
        path = folderPath + "/audio/"
        files = [];
        for filename in sorted(os.listdir(path)):
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.wav') and (any(
                    filename.find(
                        substring) != -1 for substring in filesSubstrings) if filesSubstrings is not None else True):
                files.append(filename)

        allFoundPitches = []
        allActualPitches = []
        for filename in files:
            print(filename)
            result = StreamProcessor(os.path.join(path, filename), bits_per_sample=16).run()
            allFoundPitches.append(result.fundamental_frequencies)
            print('found = ' + str(result.fundamental_frequencies))
            filenameWithoutExt = splitext(filename)[0]
            tree = ET.parse(os.path.join(folderPath + "/annotation/", filenameWithoutExt + ".xml"))
            actualPitches = []
            for event in tree.getroot().find('transcription').findall('event'):
                actualPitches.append(int(event.find('pitch').text))
            print('actual = ' + str(actualPitches))

            allActualPitches.append(actualPitches)

            for pitch in actualPitches:
                print('Playing actual pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

            for pitch in result.fundamental_frequencies:
                print('Playing found pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            # create_midi_file_with_notes('test', [Note(pitches[0], 100, 0.2, 0.5)] , 140)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)

            Chart.showSignalAndFlux(result.amplitudes, result.flux_values,
                                    result.window_size, result.onset_flux)

        TableMetrics.numeric_metrics_in_table(allActualPitches, allFoundPitches)

        # print('actual = ' + actualPitches)
        # print(
        #     np.sum(np.array(allFoundPitches) == np.array(allActualPitches)) / len(allActualPitches))  # not working now

        print("\n" * 10)


if __name__ == '__main__':
    test = Test()
