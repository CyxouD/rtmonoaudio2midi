###
# for matplotlib to be fixed https://github.com/matplotlib/matplotlib/issues/9196#issuecomment-448457722
import sys
reload(sys)
sys.setdefaultencoding('utf8')
####
import matplotlib.pyplot as plt
import numpy as np


class Chart:
    @staticmethod
    def showSignalAndFlux(amplitudes, flux_values, window_size, onset_flux_values, local_mean_thresholds,
                          exponential_decay_thresholds):
        plt.plot(
            amplitudes, "#ff9933", label="Amplitudes")

        scale = max(flux_values) / max(amplitudes)
        scaled_flux_values = [flux / scale for flux in flux_values]
        markers = filter(lambda index: index != -1,
                         map(lambda flux_index: flux_index[0] if flux_index[1] in onset_flux_values else -1,
                             enumerate(flux_values)))
        x = np.arange(0, len(flux_values) * window_size, window_size)
        # TODO center values?
        plt.plot(x,
                 scaled_flux_values, ':b*', markevery=markers, label="Spectral Flux function")

        scaled_local_mean_values = [local_mean / scale for local_mean in local_mean_thresholds]
        plt.plot(x, scaled_local_mean_values, '#804000', label="Local mean threshold function")

        scaled_exponential_decays_values = [exponential_decay / scale for exponential_decay in
                                            exponential_decay_thresholds]
        plt.plot(x, scaled_exponential_decays_values, 'g', label="Exponential decay threshold function")

        plt.legend(loc="lower right")
        plt.xlabel('frames')
        plt.ylabel('frequency and scaled thresholds')

        plt.show()
