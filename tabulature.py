import time

import mingus.extra.lilypond as LilyPond
from mingus.containers.bar import Bar
from mingus.containers.note import Note
from mingus.containers.track import Track

from audiostream import StreamProcessor


# LilyPond app should be installed
class Tabulature:
    @staticmethod
    def show(notation, fundamental_frequencies_infos, filename):
        print('fundamental_frequencies_infos', fundamental_frequencies_infos)
        print('len(fundamental_frequencies_infos)', len(fundamental_frequencies_infos))
        track = Track()
        notes = map(
            lambda fundamental_frequency_info: Note().from_hertz(fundamental_frequency_info.fundamental_frequency),
            fundamental_frequencies_infos)
        print('notes', notes)
        for note in notes:
            track + note

        print('track', track)
        lyString = ("\\new TabStaff " if notation == 'tablature' else "") + LilyPond.from_Track(track)
        print('lyString', lyString)
        LilyPond.to_pdf(lyString, "lilypond_generated/%s" % filename)


if __name__ == '__main__':
    filename = "AR_Lick1_KN.wav"
    result = StreamProcessor(
        "./test_data/IDMT-SMT-GUITAR_V2/dataset2/audio/%s" % filename,
        bits_per_sample=24).run()
    Tabulature.show('tablature', result.fundamental_frequencies_infos, filename)
