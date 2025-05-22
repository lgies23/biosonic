import matplotlib.pyplot as plt
import numpy as np
# from numpy.typing import ArrayLike, NDArray
# from typing import Union, Dict
# from compute.spectrotemporal import spectrogram
# from compute.utils import extract_all_features, check_signal_format, check_sr_format


def plot_spectrogram(Sx, t, f, db_scale=True, cmap='viridis', vmin=None, vmax=None, title="Spectrogram"):
    """
    Plot a spectrogram.
    
    Parameters:
    - Sx: 2D array, spectrogram (complex or magnitude)
    - t: 1D array, time axis
    - f: 1D array, frequency axis
    - db_scale: bool, convert magnitude to dB
    - cmap: str, colormap
    - vmin, vmax: float, optional color limits
    - title: str, title of the plot
    """
    # Convert to magnitude if complex
    magnitude = np.abs(Sx)

    # Convert to dB scale
    if db_scale:
        magnitude = 20 * np.log10(magnitude + 1e-30)  # Avoid log(0)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, magnitude, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.imshow(magnitude, 
    #        origin='lower', 
    #        aspect='auto', 
    #        extent=[t[0], t[-1], f[0], f[-1]], 
    #        cmap='viridis', 
    #        vmin=vmin, 
    #        vmax=vmax)
    plt.colorbar(label='Amplitude (dB)' if db_scale else 'Amplitude')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_filterbank_and_cepstrum(
        fbanks, 
        sr, 
        n_fft, 
        ceps, 
        fmax=None, 
        title_prefix=""):
    """
    Plots:
    1. The filter bank filters (rows of the filter bank matrix)
    2. The resulting cepstral coefficients

    Parameters:
        fbanks (np.ndarray): Filter bank matrix of shape (n_filters, n_fft//2+1)
        sr (int): Sampling rate
        n_fft (int): FFT size
        ceps (np.ndarray): Cepstral coefficients (1D array)
        fmax (float): Optional frequency max for plotting x-axis
        title_prefix (str): Optional prefix for plot titles
    """
    # Set frequency axis (only positive freqs from rfft)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # Limit frequency axis
    if fmax is not None:
        max_bin = np.searchsorted(freqs, fmax)
        freqs = freqs[:max_bin]
        fbanks = fbanks[:, :max_bin]

    # Plot filter bank
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for i in range(fbanks.shape[0]):
        plt.plot(freqs, fbanks[i])
    plt.title(f"{title_prefix}Filter Bank")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot cepstral coefficients
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(ceps)), ceps, marker='o')
    plt.title(f"{title_prefix}Cepstral Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# def plot_features(
#         data: ArrayLike, 
#         sr: int, 
#         features: Dict[str, Union[float, NDArray[np.float64]]]
#     ) -> None:
#     """
#     Plot audio signal features using precomputed feature dictionary.
#     """
#     data = check_signal_format(data)
#     sr = check_sr_format(sr)
#     features = extract_all_features(data, sr)
#     spec, times, freqs = spectrogram(data, sr)
#     spectrogram_db = 20 * np.log10(np.abs(spec))

#     # Spectrogram with Dominant Frequencies
#     plt.figure(figsize=(12, 12))
#     plt.subplot(3, 1, 1)
#     plt.title("Spectrogram with Dominant Frequencies")
#     plt.pcolormesh(times, freqs, spectrogram_db, shading='gouraud', cmap='viridis')
#     plt.scatter(times, features["dominant_freqs"], color=(0.7, 0.1, 0.1, 0.3), marker="o", label='Dominant Frequency')
#     plt.colorbar(label="Amplitude (dB)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Frequency (Hz)")
#     plt.legend()

#     # Power Spectrum
#     plt.subplot(3, 1, 2)
#     plt.title("Power Spectrum with Spectral Features")
#     freq_ps = features["freqs_ps"]
#     filtered_ps = features["filtered_pow_spectrum"]
#     if not isinstance(freq_ps, np.ndarray):
#         raise TypeError("Expected 'freqs_ps' to be an ndarray.")
#     if not isinstance(filtered_ps, np.ndarray):
#         raise TypeError("Expected 'freqs_ps' to be an ndarray.")
#     cutoff = len(freq_ps) - len(freq_ps) // 3
#     plt.plot(freq_ps[:cutoff], filtered_ps[:cutoff], label="Power Spectrum", color='blue')
    
#     plt.axvline(features["mean_f"] * 1000, color='green', linestyle="--", label="Mean Frequency (kHz)")
#     plt.axvline(features["mean_peak_f"] * 1000, color='orange', linestyle="-", label="Peak Frequency (kHz)")
#     plt.axvline(features["f_median"], color='red', linestyle="--", label="Median")
#     plt.axvline(features["f_q1"], color='yellow', linestyle="--", label="Q1")
#     plt.axvline(features["f_q3"], color='purple', linestyle="--", label="Q3")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power")
#     plt.legend()

#     # Waveform
#     plt.subplot(3, 1, 3)
#     plt.title("Waveform with Energy Envelope and Time-domain Features")
#     times_waveform = np.linspace(0, len(data) / sr, num=len(data))
#     plt.plot(times_waveform, data, label="Waveform", color="gray", alpha=0.3)
#     plt.plot(times_waveform, features["amplitude_envelope"], label="Amplitude Envelope", color="blue")
#     plt.axvline(features["t_median"], color='red', linestyle="--", label="Median")
#     plt.axvline(features["t_q1"], color='orange', linestyle="--", label="Q1")
#     plt.axvline(features["t_q3"], color='green', linestyle="--", label="Q3")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()