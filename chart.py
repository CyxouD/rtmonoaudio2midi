import matplotlib.pyplot as plt
import numpy as np


class Chart:
    @staticmethod
    def showSignalAndFlux(amplitudes, flux_values, window_size, onset_flux_values):
        plt.plot(
            amplitudes)

        scale = max(flux_values) / max(amplitudes)
        scaled_flux_values = [flux / scale for flux in flux_values]
        markers = filter(lambda index: index != -1,
                         map(lambda flux_index: flux_index[0] if flux_index[1] in onset_flux_values else -1,
                             enumerate(flux_values)))
        # TODO center values?
        plt.plot(np.arange(0, len(flux_values) * window_size, window_size),
                 scaled_flux_values, '-rD',
                 markevery=markers)

        plt.show()
