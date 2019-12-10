import os
from os.path import isfile, join, splitext
from audiostream import StreamProcessor
import xml.etree.ElementTree as ET
import numpy as np
from mingus.midi import fluidsynth
import time
from midi import create_midi_file_with_notes, Note

DELAYS_SECONDS_BETWEEN_PLAYING = 2;


class Test(object):

    def __init__(self):
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge+Neck SC")
        self.process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU")

    def process_folder(self, folderPath):
        print(folderPath)
        path = folderPath + "/audio/"
        files = [];

        for filename in sorted(os.listdir(path)):
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.wav'):
                files.append(filename)
        allFoundPitches = []
        allActualPitches = []
        for filename in files:
            print(filename)
            pitches = StreamProcessor(os.path.join(path, filename)).run()
            allFoundPitches.append(pitches)
            # print('found = ' + str(pitches))
            filenameWithoutExt = splitext(filename)[0]
            tree = ET.parse(os.path.join(folderPath + "/annotation/", filenameWithoutExt + ".xml"))
            actualPitches = [int(tree.getroot()[1][0][0].text)]
            allActualPitches.append(actualPitches)

            for pitch in actualPitches:
                print('Playing actual pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)
            for pitch in pitches:
                print('Playing found pitch: ' + str(pitch) + '...')
                fluidsynth.play_Note(pitch, 0, 100)
            # create_midi_file_with_notes('test', [Note(pitches[0], 100, 0.2, 0.5)] , 140)
            time.sleep(DELAYS_SECONDS_BETWEEN_PLAYING)
        # print('actual = ' + actualPitches)
        print(
            np.sum(np.array(allFoundPitches) == np.array(allActualPitches)) / len(allActualPitches))  # not working now

        print("\n" * 10)


if __name__ == '__main__':
    test = Test()
