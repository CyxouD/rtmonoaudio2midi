import time
import itertools
from collections import deque

import numpy as np
from pyaudio import PyAudio, paContinue, paInt16

from app_setup import (
    RING_BUFFER_SIZE,
    WINDOW_SIZE,
    FROM_FILE,
    LOCAL_MAX_WINDOW,
    LOCAL_MEAN_RANGE_MULTIPLIER,
    LOCAL_MEAN_THRESHOLD,
    EXPONENTIAL_DECAY_THRESHOLD_PARAMETER,
    SAMPLE_RATE)
from midi import hz_to_midi, RTNote
from mingus.midi import fluidsynth
from app_setup import SOUNDFONT
import wave
from chart import Chart
import collections
from math import sqrt, pow


class SpectralAnalyser(object):
    # guitar frequency range
    FREQUENCY_RANGE = (80, 1200)

    def __init__(self, window_size, sample_rate, local_max_window, local_mean_range_multiplier, local_mean_threshold,
                 exponential_decay_threshold_parameter):
        self._window_size = window_size
        self._sample_rate = sample_rate
        self._w = local_max_window
        self._m = local_mean_range_multiplier
        self._e = local_mean_threshold
        self._a = exponential_decay_threshold_parameter

        self.spectrums = []
        self._amplitudes = []
        self._fluxs = []
        self._onset_flux = []

        self._hanning_window = np.hanning(window_size)
        # The zeros which will be used to double each segment size
        self._inner_pad = np.zeros(window_size)

    """
        Took spectral flux method from here http://www.mazurka.org.uk/software/sv/plugin/MzSpectralFlux/ 
        in 'Peak detection' section (same described in this author's work https://pdfs.semanticscholar.org/2f5d/2c3884181f19a78efc17ce07c54f249edb0e.pdf).
    """

    def is_onset(self, n):

        is_flux_local_max = self.is_flux_local_max(n)
        is_more_local_mean_threshold = self.is_more_local_mean_threshold(n)
        is_more_exponential_decay_threshold = self.is_more_exponential_decay_threshold(n)

        is_onset = is_flux_local_max and is_more_local_mean_threshold and is_more_exponential_decay_threshold
        if is_onset:
            print('n', n)
            print('flux', self._fluxs[n])
            print('find_spectral_flux_local_max', self.find_spectral_flux_local_max(n))
            print('is_flux_local_max', is_flux_local_max)
            print('local_mean_threshold', self.local_mean_threshold(n))
            print('is_more_local_mean_threshold', is_more_local_mean_threshold)
            print('self.exponential_decay_threshold(n-1)', self.exponential_decay_threshold(n - 1))
            print('is_more_exponential_decay_threshold', is_more_exponential_decay_threshold)
            print('\n' * 3)
            self._onset_flux.append(self._fluxs[n])

        return is_onset

    def is_flux_local_max(self, n):
        flux = self._fluxs[n]
        return flux >= self.find_spectral_flux_local_max(n)

    def find_spectral_flux_local_max(self, n):
        range_fluxs = self._fluxs[slice(max(0, n - self._w), n + self._w + 1)]
        return max(range_fluxs)

    def is_more_local_mean_threshold(self, n):
        flux = self._fluxs[n]
        return flux >= self.local_mean_threshold(n)

    def local_mean_threshold(self, n):
        return np.mean(self._fluxs[slice(max(0, n - self._m * self._w), n + self._w + 1)]) + self._e

    def is_more_exponential_decay_threshold(self, n):
        flux = self._fluxs[n]
        return flux >= self.exponential_decay_threshold(n - 1)

    def exponential_decay_threshold(self, n):
        flux = self._fluxs[n]
        return max(flux, self._a * self.exponential_decay_threshold(n - 1) + (1 - self._a) * flux) if n > 0 else flux

    def find_fundamental_freq(self, samples):
        cepstrum = self.cepstrum(samples)
        # search for maximum between 0.08ms (=1200Hz) and 12.5ms (=80Hz)
        # as it's about the recorder's frequency range of one octave
        min_freq, max_freq = self.FREQUENCY_RANGE
        start = int(self._sample_rate / max_freq)
        end = int(self._sample_rate / min_freq)
        narrowed_cepstrum = cepstrum[start:end]

        peak_ix = narrowed_cepstrum.argmax()
        freq0 = self._sample_rate / (start + peak_ix)

        if freq0 < min_freq or freq0 > max_freq:
            # Ignore the note out of the desired frequency range
            return

        return freq0

    def process_data(self, windows_data):
        self._amplitudes = self.flatten(windows_data)

        self.spectrums = self.calculate_spectrums(windows_data)
        self.calculate_flexs()

        fund_frequencies = []
        for n in range(0, len(windows_data)):
            onset = self.is_onset(n)
            if onset:
                fund_frequencies.append(self.find_fundamental_freq(windows_data[n]))

        return fund_frequencies

    def calculate_spectrums(self, windows_data):
        """
        Calculates a power spectrum of the given data using the Hamming window.
        """
        # TODO: find another way to treat differently if not
        # equal to the window size
        return list(map(
            lambda samples: self._autopower_spectrum(samples) if (np.size(samples) == self._window_size) else np.zeros(
                self._window_size), windows_data)
        )

    def _autopower_spectrum(self, samples):
        windowed = samples * self._hanning_window
        # Add 0s to double the length of the data
        padded = np.append(windowed, self._inner_pad)
        # Take the Fourier Transform and scale by the number of samples
        spectrum = np.fft.fft(padded) / self._window_size
        autopower = np.abs(spectrum * np.conj(spectrum))
        return autopower[:self._window_size]

    def calculate_flexs(self):
        for i in range(0, len(self.spectrums)):
            """
                   Calculates the difference between the current and last spectrum (Spectral Flux),
                   then applies a thresholding function and checks if a peak occurred.
                   my comment: took from http://www.mazurka.org.uk/software/sv/plugin/MzSpectralFlux/ (Positive Flux)
            """
            last_spectrum = self.spectrums[i - 1] if i > 0 else np.zeros(self._window_size,
                                                                         dtype=np.int16)
            spectrum = self.spectrums[i]

            flux = sqrt(sum([max(pow(spectrum[n] - last_spectrum[n], 2), 0) for n in xrange(self._window_size)]))
            self._fluxs.append(flux)

    def cepstrum(self, samples):
        """
        Calculates the complex cepstrum of a real sequence.
        """
        spectrum = np.fft.fft(samples)
        log_spectrum = np.log(np.abs(
            spectrum))  # TODO check divide by zero error in test_data/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Neck HU
        cepstrum = np.fft.ifft(log_spectrum).real
        return cepstrum

    def flatten(self, l):
        flat_list = []
        for sublist in l:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def getFluxValues(self):
        return list(self._fluxs)

    def getOnsetFluxValues(self):
        return self._onset_flux;

    def getAmplitudes(self):
        return self._amplitudes


class StreamProcessor:
    def __init__(self, pathWav, bits_per_sample, local_max_window=LOCAL_MAX_WINDOW,
                 local_mean_range_multiplier=LOCAL_MEAN_RANGE_MULTIPLIER, local_mean_threshold=LOCAL_MEAN_THRESHOLD,
                 exponential_decay_threshold_parameter=EXPONENTIAL_DECAY_THRESHOLD_PARAMETER,
                 play_notes=False):
        self._bits_per_sample = bits_per_sample;
        self._play_notes = play_notes
        self._local_max_window = local_max_window
        self._wf = wave.open(pathWav, 'rb')
        if FROM_FILE:
            self._sample_rate = self._wf.getframerate()
        else:
            self._sample_rate = SAMPLE_RATE
        self._spectral_analyser = SpectralAnalyser(
            window_size=WINDOW_SIZE,
            sample_rate=self._sample_rate,
            local_max_window=self._local_max_window,
            local_mean_range_multiplier=local_mean_range_multiplier,
            local_mean_threshold=local_mean_threshold,
            exponential_decay_threshold_parameter=exponential_decay_threshold_parameter)

        fluidsynth.init(SOUNDFONT)

    def run(self):
        fundament_freqs = []
        if FROM_FILE:
            frames = self._wf.readframes(-1)
            fundament_freqs = self._process_wav_window_frames(frames)
            self._wf.close()
        else:
            pya = PyAudio()
            self._stream = pya.open(
                format=paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=self._process_stream_frame,
            )
            self._stream.start_stream()

            while self._stream.is_active() and not raw_input():
                time.sleep(0.1)

            self._stream.stop_stream()
            self._stream.close()
            pya.terminate()

        Result = collections.namedtuple('Result',
                                        ['fundamental_frequencies', 'amplitudes', 'flux_values', 'window_size',
                                         'onset_flux'])
        return Result(filter(lambda x: x is not None, fundament_freqs), self._spectral_analyser.getAmplitudes(),
                      self._spectral_analyser.getFluxValues(), WINDOW_SIZE,
                      self._spectral_analyser.getOnsetFluxValues())

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _process_wav_window_frames(self, frames):
        bits_in_byte = 8
        frames_chunks = list(self.chunks(frames, WINDOW_SIZE * self._bits_per_sample / bits_in_byte))
        data = list(map(self._get_wav_frames, frames_chunks))
        return self._process_data(data)

    def _get_wav_frames(self, frames):
        if self._bits_per_sample == 24:
            return np.frombuffer(frames, 'b').reshape(-1, 3)[:, 1:].flatten().view('i2')
        elif self._bits_per_sample == 16:
            return np.frombuffer(frames, dtype=np.int16)
        else:
            raise Exception('Not handled bits per sample = ' + self._bits_per_sample)

    def _process_stream_frame(self, data, frame_count, time_info, status_flag):
        data_array = np.fromstring(data, dtype=np.int16)
        self._process_data(data_array)
        return (data, paContinue)

    def _process_data(self, frames_data):
        return self._spectral_analyser.process_data(frames_data)
        # if freq0:
        #     # Onset detected
        #     if not FROM_FILE or self._play_notes:
        #         print("Note detected; fundamental frequency: ", freq0)
        #     midi_note_value = int(hz_to_midi(freq0)[0])
        #     if not FROM_FILE or self._play_notes:
        #         print("Midi note value: ", midi_note_value)
        #         fluidsynth.play_Note(midi_note_value, 0, 100)
        #     return midi_note_value


if __name__ == '__main__':
    stream_proc = StreamProcessor(
        "test_data/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-40100-1111-00001.wav",
        bits_per_sample=16,
        play_notes=True)
    result = stream_proc.run()
    print(result.onset_flux)
    # Chart.showSignalAndFlux(result.amplitudes, result.flux_values,
    #                         result.window_size, result.onset_flux)
