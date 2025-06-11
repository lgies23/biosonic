import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Any, Union, Literal, Tuple

from biosonic.compute.spectrotemporal import cepstrum, spectrogram, cepstral_coefficients

# from compute.spectrotemporal import spectrogram
# from compute.utils import extract_all_features, check_signal_format, check_sr_format


# def plot_spectrogram(
#         data : ArrayLike, 
#         sr : int,
#         window: Union[str, ArrayLike] = "hann", 
#         window_length : int = 1024,
#         overlap: float = .5, 
#         scaling: Optional[Literal['psd', 'magnitude', 'angle', 'phase']] = "magnitude",
#         db_scale : bool = True, 
#         cmap : str = "grey", 
#         vmin : Optional[float] = None, 
#         vmax : Optional[float] = None, 
#         title : str = "Spectrogram",
#         *args : Any, 
#         **kwargs : Any
#     ) -> None:
#     """
#     Plot a spectrogram using matplotlibs "specgram" function.
    
#     Parameters:
#     # TODO fix
#     - Sx: 2D array, spectrogram (complex or magnitude)
#     - t: 1D array, time axis
#     - f: 1D array, frequency axis
#     - db_scale: bool, convert magnitude to dB
#     - cmap: str, colormap
#     - vmin, vmax: float, optional color limits
#     - title: str, title of the plot
#     """
#     data = check_signal_format(data)
#     sr = check_sr_format(sr)

#     hop_length = int(window_length * (1 - overlap))
#     noverlap_ = window_length - hop_length

#     if isinstance(window, str):
#         try:
#             window = signal.windows.get_window(window, window_length)
#         except ValueError as e:
#             raise ValueError(f"Invalid window type: {window}") from e
#     else:
#         window = np.asarray(window) 
#         if not isinstance(window, np.ndarray):
#             raise TypeError("'window' must be either a string or a 1D NumPy array.")
 
#     plt.specgram(
#         data, 
#         Fs=sr, 
#         window=window, 
#         NFFT=window_length, 
#         noverlap=noverlap_, 
#         mode=scaling, 
#         scale='dB' if db_scale else 'linear',
#         cmap=cmap
#         )
#     plt.title(title)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.colorbar(label="Intensity [dB]" if db_scale else "Magnitude")
    
def plot_spectrogram(
        data : ArrayLike, 
        sr: int, 
        db_scale : bool = True, 
        cmap : str = 'viridis', 
        vmin : Optional[float] = None, 
        vmax : Optional[float] = None, 
        title : str = "Spectrogram",
        db_ref : Optional[float] = None,
        flim: Optional[Tuple[float, float]] = None,
        tlim: Optional[Tuple[float, float]] = None,
        freq_scale: Literal["linear", "log", "mel"] = "linear",
        window_length: int = 512,
        window: Union[str, ArrayLike] = "hann",
        zero_padding: int = 0,
        overlap: float = 50,
        noisereduction: Optional[int] = None,
        n_fft: Optional[int] = None,
        n_bands: int = 40,
        corner_frequency: Optional[float] = None,
        **kwargs: Any
    ) -> None:
    """
    Plot a time-frequency spectrogram with optional dB scaling and frequency axis transformations.

    Parameters
    ----------
    data : ArrayLike
        Input signal.
    sr : int
        Sampling rate in Hz.
    db_scale : bool, optional
        Whether to convert the spectrogram to decibel scale.
    cmap : str, optional
        Colormap for the spectrogram.
    vmin, vmax : float or None
        Color limits.
    title : str, optional
        Title of the plot.
    db_ref : float or None
        Reference for dB scaling (max if None).
    flim : tuple of float or None
        Frequency limits in Hz.
    tlim : tuple of float or None
        Time limits in seconds.
    freq_scale : {"mel", "log", None}
        Apply mel or log frequency scaling.
    n_fft : int, optional
        FFT size used to generate the spectrogram (required for mel/log scaling).
    n_bands : int, optional
        Number of mel or log bands to use.
    corner_frequency : float, optional
        Corner frequency used for perceptual scaling.
    **kwargs : dict
        Extra arguments passed to filterbank functions.
    """
    Sx, t, f = spectrogram(
        data=data,
        sr=sr,
        window_length=window_length,
        window=window,
        zero_padding=zero_padding,
        overlap=overlap,
        noisereduction=noisereduction,
        complex_output=False
    )

    if n_fft is None:
        n_fft = window_length + zero_padding

    if freq_scale in ("mel", "log"):
        if sr is None or n_fft is None:
            raise ValueError("Sample rate and n_fft must be provided for mel or log frequency scale.")

        fmin = flim[0] if flim else 0.0
        fmax = flim[1] if flim and flim[1] else sr / 2

        if freq_scale == "mel":
            from biosonic.filter import mel_filterbank
            fb, f_centers = mel_filterbank(n_bands, n_fft, sr, fmin=fmin, fmax=fmax, corner_frequency=corner_frequency, **kwargs)
        elif freq_scale == "log":
            from biosonic.filter import log_filterbank
            fb, f_centers = log_filterbank(n_bands, n_fft, sr, fmin=fmin, fmax=fmax, **kwargs)
            print(f_centers[:5])

        f = f_centers
        print(f_centers[0], f_centers[-1])
        Sx = fb @ Sx

    # Apply dB scale
    if db_scale:
        ref = np.max(Sx) if db_ref is None else db_ref
        Sx = 20 * np.log10(Sx / ref)
        # Sx[Sx < -100] = -100

    # Apply frequency limits
    if flim is not None and freq_scale=="linear":  # already handled in mel/log cases
        fmin, fmax = flim
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        Sx = Sx[mask, :]

    # Apply time limits
    if tlim is not None:
        tmin, tmax = tlim
        mask = (t >= tmin) & (t <= tmax)
        t = t[mask]
        Sx = Sx[:, mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = plt.imshow(Sx, aspect='auto', origin='lower', extent=(t[0], t[-1], f[0], f[-1]), cmap=cmap)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")

    # if freq_scale == "log":
    #     ax.set_yscale("log")

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude (dB)" if db_scale else "Amplitude")
    plt.tight_layout()
    plt.show()
    
    return ax


def plot_cepstrum(
        data : ArrayLike,
        sr : int, 
        **kwargs : Any
) -> None:
    
    t = np.arange(len(data)) / sr
    ceps, _ = cepstrum(data, sr, **kwargs)

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, data)
    ax0.set_xlabel('time in seconds')
    # ax0.set_xlim(0.0, 0.05)
    ax1 = fig.add_subplot(212)
    ax1.plot(t, ceps)
    ax1.set_xlabel('quefrency in seconds')
    ax1.set_xlim(0.005, 0.015)
    # ax1.set_ylim(-5., +10.)
    # plt.figure(figsize=(14, 5))
    # plt.plot(t, ceps)
    # plt.title("Cepstrum")
    # plt.xlabel("Coefficient Index")
    # plt.ylabel("Amplitude")
    # plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_cepstral_coefficients(
        data : ArrayLike, 
        sr : int, 
        n_fft : int, 
        n_filters : int = 32, 
        n_ceps : int = 40
    ) -> None:
        ceps = cepstral_coefficients(data, sr, n_fft, n_filters, n_ceps)
        print(ceps.shape)


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


def plot_pitch_on_spectrogram(
    data,
    sr,
    time_points,
    all_candidates,
    window_length=1024,
    overlap=50,
    show_strongest=True,
    db_scale=True,
    flim=None,
    cmap='viridis'
):
    # Compute spectrogram
    #Sx, t, f= spectrogram(data, sr=sr, window_length=window_length, overlap=overlap)

    # Plot base spectrogram
    ax = plot_spectrogram(
        data,
        sr,
        overlap=overlap,
        db_scale=db_scale, 
        cmap=cmap, 
        flim=flim,
        title="Spectrogram with Pitch Candidates",
        n_fft=window_length
    )
    
    for t_frame, candidates in zip(time_points, all_candidates):
        for pitch, _ in candidates:
            if pitch > 0:
                ax.plot(t_frame, pitch, 'w.', alpha=0.4, markersize=4)

    # Optionally overlay the strongest voiced candidate
    if show_strongest:
        times = []
        pitches = []
        for t_frame, candidates in zip(time_points, all_candidates):
            voiced = [c for c in candidates if c[0] > 0]
            if voiced:
                best = max(voiced, key=lambda x: x[1])
                times.append(t_frame)
                pitches.append(best[0])
        ax.plot(times, pitches, 'r-', linewidth=1.5, label="Strongest pitch")
        ax.legend()

    plt.show()


def plot_pitch_candidates(
        time_points : ArrayLike, 
        all_candidates : ArrayLike, 
        show_strongest : bool = True
    ) -> None:
    """
    Plot pitch candidates over time.
    
    Parameters:
    - time_points: list of time stamps for each frame.
    - all_candidates: list of lists of (pitch, strength) tuples.
    - show_strongest: if True, highlight the strongest voiced candidate per frame.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot all candidates
    for t, candidates in zip(time_points, all_candidates):
        for pitch, strength in candidates:
            if pitch > 0:
                plt.plot(t, pitch, 'k.', alpha=0.3)

    # Optionally plot the strongest voiced candidate
    if show_strongest:
        times = []
        pitches = []
        for t, candidates in zip(time_points, all_candidates):
            voiced = [c for c in candidates if c[0] > 0]
            if voiced:
                best = max(voiced, key=lambda x: x[1])
                times.append(t)
                pitches.append(best[0])
        plt.plot(times, pitches, 'b-', label="Strongest candidate", linewidth=1.5)

    plt.title("Praat-style Pitch Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()