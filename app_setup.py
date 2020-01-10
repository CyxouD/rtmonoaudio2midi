# start: best results retrieved from TuningHyperparameters.py
WINDOW_SIZE = 2048
LOCAL_MEAN_THRESHOLD = 100000  # TODO do not make it constant? 'δ is the threshold above the local mean which an onset must reach' (https://pdfs.semanticscholar.org/2f5d/2c3884181f19a78efc17ce07c54f249edb0e.pdf)
LOCAL_MAX_WINDOW = 3  # window used to find a local maximum
LOCAL_MEAN_RANGE_MULTIPLIER = 2
EXPONENTIAL_DECAY_THRESHOLD_PARAMETER = 0.75
SPECTRAL_FLUX_NORM_LEVEL = 1
# end: best results retrieved from TuningHyperparameters.py

RING_BUFFER_SIZE = 40
SAMPLE_RATE = 44100

SOUNDFONT = "sound_fonts/SpanishClassicalGuitar.sf2"
# don't set it to False, broke it right now
FROM_FILE = True
