import matplotlib.pyplot as plt
import numpy as np
from compute.spectral import spectrogram
from compute.utils import extract_all_features

def plot_features(data, sr, features):
    """
    Plot audio signal features using precomputed feature dictionary.
    """
    features = extract_all_features(data, sr)
    spectrogram_db = 20 * np.log10(np.abs(spectrogram(data)))

    # Spectrogram with Dominant Frequencies
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.title("Spectrogram with Dominant Frequencies")
    plt.pcolormesh(features["spectrogram_times"], features["spectrogram_freqs"], spectrogram_db, shading='gouraud', cmap='viridis')
    plt.scatter(features["spectrogram_times"], features["dominant_freqs"], color=(0.7, 0.1, 0.1, 0.3), marker="o", label='Dominant Frequency')
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()

    # Power Spectrum
    plt.subplot(3, 1, 2)
    plt.title("Power Spectrum with Spectral Features")
    cutoff = len(features["freqs_ps"]) - len(features["freqs_ps"]) // 3
    plt.plot(features["freqs_ps"][:cutoff], features["filtered_pow_spectrum"][:cutoff], label="Power Spectrum", color='blue')
    plt.axvline(features["mean_f"] * 1000, color='green', linestyle="--", label="Mean Frequency (kHz)")
    plt.axvline(features["mean_peak_f"] * 1000, color='orange', linestyle="-", label="Peak Frequency (kHz)")
    plt.axvline(features["f_median"], color='red', linestyle="--", label="Median")
    plt.axvline(features["f_q1"], color='yellow', linestyle="--", label="Q1")
    plt.axvline(features["f_q3"], color='purple', linestyle="--", label="Q3")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()

    # Waveform
    plt.subplot(3, 1, 3)
    plt.title("Waveform with Energy Envelope and Time-domain Features")
    times_waveform = np.linspace(0, len(data) / sr, num=len(data))
    plt.plot(times_waveform, data, label="Waveform", color="gray", alpha=0.3)
    plt.plot(times_waveform, features["amplitude_envelope"], label="Amplitude Envelope", color="blue")
    plt.axvline(features["t_median"], color='red', linestyle="--", label="Median")
    plt.axvline(features["t_q1"], color='orange', linestyle="--", label="Q1")
    plt.axvline(features["t_q3"], color='green', linestyle="--", label="Q3")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()