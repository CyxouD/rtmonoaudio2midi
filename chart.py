import matplotlib.pyplot as plt
import numpy as np


class Chart:
    @staticmethod
    def showSignalAndFlux(amplitudes, flux_values, window_size):
        plt.plot(
            amplitudes)

        scale = max(flux_values) / max(amplitudes)
        scaled_flux_values = [flux / scale for flux in flux_values]
        # TODO center values?
        plt.plot(np.arange(0, len(flux_values) * window_size, window_size),
                 scaled_flux_values, 'r')

        plt.show()
