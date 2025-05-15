# import matplotlib.pyplot as plt
# import numpy as np
# from numpy.typing import ArrayLike, NDArray
# from typing import Union, Dict
# from compute.spectral import spectrogram
# from compute.utils import extract_all_features, check_signal_format, check_sr_format

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