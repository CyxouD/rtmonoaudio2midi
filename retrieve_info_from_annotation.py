import collections
import os
import xml.etree.ElementTree as ET

Pitch_info = collections.namedtuple('Pitch_info',
                                    ['pitch', 'onset_sec'])


def retrieve_info_from_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    actual_pitches_infos = []
    for event in tree.getroot().find('transcription').findall('event'):
        actual_pitches_infos.append(
            Pitch_info(pitch=int(event.find('pitch').text),
                       onset_sec=float(event.find('onsetSec').text)))
