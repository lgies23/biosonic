import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Any, Union, Literal
from scipy import signal

from biosonic.compute.utils import check_signal_format, check_sr_format
# from compute.spectrotemporal import spectrogram
# from compute.utils import extract_all_features, check_signal_format, check_sr_format


def plot_spectrogram(
        data : ArrayLike, 
        sr : int,
        window: Union[str, ArrayLike] = "hann", 
        window_length : int = 1024,
        overlap: float = .5, 
        scaling: Optional[Literal['psd', 'magnitude', 'angle', 'phase']] = "magnitude",
        db_scale : bool = True, 
        cmap : str = "grey", 
        vmin : Optional[float] = None, 
        vmax : Optional[float] = None, 
        title : str = "Spectrogram",
        *args : Any, 
        **kwargs : Any
    ) -> None:
    """
    Plot a spectrogram using matplotlibs "specgram" function.
    
    Parameters:
    # TODO fix
    - Sx: 2D array, spectrogram (complex or magnitude)
    - t: 1D array, time axis
    - f: 1D array, frequency axis
    - db_scale: bool, convert magnitude to dB
    - cmap: str, colormap
    - vmin, vmax: float, optional color limits
    - title: str, title of the plot
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    hop_length = int(window_length * (1 - overlap))
    noverlap_ = window_length - hop_length

    if isinstance(window, str):
        try:
            window = signal.windows.get_window(window, window_length)
        except ValueError as e:
            raise ValueError(f"Invalid window type: {window}") from e
    else:
        window = np.asarray(window) 
        if not isinstance(window, np.ndarray):
            raise TypeError("'window' must be either a string or a 1D NumPy array.")
 
    plt.specgram(
        data, 
        Fs=sr, 
        window=window, 
        NFFT=window_length, 
        noverlap=noverlap_, 
        mode=scaling, 
        scale='dB' if db_scale else 'linear',
        cmap=cmap
        )
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]" if db_scale else "Magnitude")
    


def plot_filterbank_and_cepstrum(
        fbanks : ArrayLike, 
        sr : int, 
        n_fft : int, 
        ceps : ArrayLike, 
        fmax : Optional[float] = None, 
        title_prefix : str = ""
    ) -> None:
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