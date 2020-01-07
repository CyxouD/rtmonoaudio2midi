from mingus.midi import fluidsynth
from process_folder import process_folder
from table_metrics import TableMetrics
import time

DELAYS_SECONDS_BETWEEN_PLAYING = 0

PENALTY = 50
MISSED_TO_EXTRA_PENALTY_RATIO = 2 / 1


class Test(object):

    def __init__(self):
        # Single notes are played on each string from the 0th fret (empty string) to the 12th fret.
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder(
            "test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC", bitDepth=16,
            show_chart=False, print_logs=False)
        self.show_table(allActualPitchesInfos, allFoundPitchesInfos)
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder(
            "test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU", bitDepth=16,
            show_chart=False, print_logs=False)
        self.show_table(allActualPitchesInfos, allFoundPitchesInfos)
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder(
            "test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge+Neck SC",
            bitDepth=16, show_chart=False, print_logs=False)
        self.show_table(allActualPitchesInfos, allFoundPitchesInfos)
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder(
            "test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU", bitDepth=16,
            show_chart=False, print_logs=False)
        self.show_table(allActualPitchesInfos, allFoundPitchesInfos)

        # monophonic songs
        # TODO include Lick1 but handle it differently from Lick10,
        #  add other Lick3, Lick4, Lick5, Lick6, Lick11, but handle that some annotations are missing
        (allActualPitchesInfos, allFoundPitchesInfos) = process_folder("test_data/IDMT-SMT-GUITAR_V2/dataset2",
                                                                            bitDepth=24,
                                                                            filesSubstrings=["AR_Lick2"],
                                                                            show_chart=False, print_logs=True)
        self.play_found_and_actual_pitches(allActualPitchesInfos, allFoundPitchesInfos)
        self.show_table(allActualPitchesInfos, allFoundPitchesInfos)

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


if __name__ == '__main__':
    test = Test()
