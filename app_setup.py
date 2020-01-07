# start: best results retrieved from TuningHyperparameters.py
WINDOW_SIZE = 2048
LOCAL_MAX_WINDOW = 3  # window used to find a local maximum
LOCAL_MEAN_RANGE_MULTIPLIER = 3
LOCAL_MEAN_THRESHOLD = 50000
EXPONENTIAL_DECAY_THRESHOLD_PARAMETER = 0.75
SPECTRAL_FLUX_NORM_LEVEL = 1
# end: best results retrieved from TuningHyperparameters.py

RING_BUFFER_SIZE = 40
SAMPLE_RATE = 44100

SOUNDFONT = "sound_fonts/SpanishClassicalGuitar.sf2"
# don't set it to False, broke it right now
FROM_FILE = True
