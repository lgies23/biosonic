import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Any, Union, Literal, Tuple, List, Dict
from pandas import DataFrame

from biosonic.compute.spectrotemporal import cepstrum, spectrogram, cepstral_coefficients
from biosonic.compute.spectral import spectrum
from biosonic.filter import mel_filterbank

from biosonic.compute.utils import extract_all_features, check_signal_format, check_sr_format

    
def plot_spectrogram(
        data: Union[NDArray[np.float32], Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]],
        sr: int, 
        db_scale : bool = True, 
        cmap : str = 'binary', 
        title : Optional[str] = None,
        db_ref : Optional[float] = None,
        dynamic_range: Optional[float] = 100,
        flim: Optional[Tuple[float, float]] = None,
        tlim: Optional[Tuple[float, float]] = None,
        freq_scale: Literal["linear", "log", "mel"] = "linear",
        window_length: int = 512,
        window: Union[str, ArrayLike] = "hann",
        overlap: float = 50,
        noisereduction: Optional[bool] = False,
        n_bands: int = 40,
        corner_frequency: Optional[float] = None,
        plot : Optional[Tuple[Figure, Axes]] = None,
        **kwargs: Any
    ) -> Tuple[Figure, Axes]:
    """
    Plot a time-frequency spectrogram with optional dB scaling and frequency axis transformations.

    Parameters
    ----------
    data : Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        Input signal. Either as precomputed spectrogram (S, t, f) or 1D signal array.
    sr : int
        Sampling rate in Hz.
    db_scale : bool, optional
        Whether to convert the spectrogram to decibel scale. Default is True.
    cmap : str, optional
        Colormap for the spectrogram.
    title : str, optional
        Title of the plot. If None, title is generated based on 'freq_scale'.
    db_ref : float or None
        Reference for dB scaling (max if None).
    dynamic_range : float or None, optional
        Clip values below (max_dB - dynamic_range) after dB conversion. Set to None if no clipping is desired.
        Useful for suppressing low-amplitude noise. Default is 100.
    flim : tuple of float or None
        Frequency limits in Hz.
    tlim : tuple of float or None
        Time limits in seconds.
    freq_scale : {"linear", "log", "mel"}, optional
        Frequency axis scale. Choose "linear", "log", or "mel". Default is "linear".
    window_length : int, optional
        Length of the window for STFT (in samples). Default is 512.
    window : str or ArrayLike, optional
        Windowing function used for the STFT. Default is "hann".
    zero_padding : int, optional
        Number of zeros to pad each windowed frame. Default is 0.
    overlap : float, optional
        Percentage of overlap between successive windows (0 to 100). Default is 50.
    noisereduction : bool, optional
        Whether to apply noise reduction. Default is False (no reduction).
    n_bands : int, optional
        Number of frequency bands for mel or log scaling. Default is 40.
    corner_frequency : float, optional
        Corner frequency for perceptual frequency scaling (used in mel scale).
    plot : tuple(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        Existing matplotlib Figure and Axes objects to plot into. If None, a new figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to `matplotlib.pyplot.imshow`.
    """
    # Precomputed spectrogram
    if isinstance(data, tuple) and len(data) == 3:
        Sx, t, f = data
    
    # Raw signal + sr
    elif isinstance(data, np.ndarray):
        if sr is None:
            raise ValueError("sr must be provided when passing a signal array.")
        Sx, t, f = spectrogram(
        data=data,
        sr=sr,
        window_length=window_length,
        window=window,
        overlap=overlap,
        noisereduction=noisereduction,
        complex_output=False
    )
    
    else:
        raise TypeError("data must be either a (S, t, f) tuple or a 1D np.ndarray signal")

    if freq_scale=="mel":
        if sr is None:
            raise ValueError("Sample rate must be provided for mel frequency scale.")

        fmin = flim[0] if flim else 0.0
        fmax = flim[1] if flim and flim[1] else sr / 2
        
        fb, f_centers = mel_filterbank(n_bands, window_length, sr, fmin=fmin, fmax=fmax, corner_frequency=corner_frequency)
        f = f_centers
        Sx = fb @ Sx

    # Apply dB scale
    if db_scale:
        ref = np.max(Sx) if db_ref is None else db_ref
        Sx = 20 * np.log10(Sx / ref)
        
        if dynamic_range is not None:
            if dynamic_range is not None:
                Sx = np.maximum(Sx, -dynamic_range)

    # Apply frequency limits
    if flim is not None and freq_scale in ("linear", "log"):  # already handled in mel case
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

    # Plot
    if plot is not None:
        fig, ax = plot
    else: 
        fig, ax = plt.subplots(figsize=(10, 5))

    extent = (
        float(t[0]),
        float(t[-1]),
        float(f[0]),
        float(f[-1]),
    )

    im = ax.imshow(Sx, aspect='auto', origin='lower', extent=extent, cmap=cmap, **kwargs)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")

    if freq_scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(float(min(f[f>0])), float(f[-1]))

    if title is None:
        title = f"{freq_scale}-scaled spectrogram"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label=("Amplitude [dB]" if db_scale else "Magnitude"))
    plt.tight_layout()

    if plot is None:
        plt.show()
    
    return fig, ax


def plot_cepstrum(
        data : ArrayLike,
        sr : int, 
        max_quefrency: float = 0.05,
        log_scale : bool = True, 
        ylim : Optional[Tuple[float, float]] = None,
        title : Optional[str] = None,
        **kwargs : Any
) -> None:
    """
    Plot the cepstrum against quefrency (in seconds).

    Parameters
    ----------
    data : ArrayLike
        Signal to use for cepstrum calculation.
    sr : int
        Sampling rate of the original signal.
    max_quefrency : float, optional
        Maximum quefrency (in seconds) to plot. Defaults to 0.05s.
    log_scale : bool, optional
        Wether to log-scale the y-axis. Defaults to True.
    ylim : tuple, optional
        Limits for the y-axis.
    title : str or None
        Title for the plot. If None, a default will be generated.
    """
    ceps, quefs = cepstrum(data, sr, **kwargs)

    mask = quefs <= max_quefrency

    plt.plot(quefs[mask], ceps[mask], color='steelblue')
    plt.xlabel("Quefrency (s)")
    if log_scale:
        plt.yscale("log")
    if ylim:
        plt.ylim(ylim)
    plt.ylabel("Cepstrum")
    plt.title(title or f"Cepstrum (Sampling rate: {sr} Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cepstral_coefficients(
        data : ArrayLike, 
        sr : int, 
        window_length : int, 
        n_filters : int = 32, 
        n_ceps : int = 40,
        pre_emphasis: float = 0.97,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        filterbank_type: Literal["mel", "linear", "log"] = "mel",
        cmap : Optional[str] = "grey",
        **kwargs : Any
    ) -> None:
        """
        Compute and plot cepstral coefficients over time from audio data.

        This function calculates cepstral coefficients using a specified filterbank 
        type and visualizes them with time on the x-axis and 
        cepstral coefficient indices on the y-axis.

        Parameters
        ----------
        data : ArrayLike
            Input audio signal (1D array-like).
        sr : int
            Sampling rate of the audio signal in Hz.
        window_length : int
            Length of the analysis window in samples.
        n_filters : int, optional
            Number of filters in the filterbank. Default is 32.
        n_ceps : int, optional
            Number of cepstral coefficients to compute and plot. Default is 40.
        pre_emphasis : float, optional
            Pre-emphasis filter coefficient. Default is 0.97.
        fmin : float, optional
            Minimum frequency for the filterbank in Hz. Default is 0.0.
        fmax : float, optional
            Maximum frequency for the filterbank in Hz. Defaults to Nyquist frequency if None.
        filterbank_type : {'mel', 'linear', 'log'}, optional
            Type of filterbank to use for cepstral coefficient calculation.
            Default is 'mel'.
        cmap : str or None, optional
            Matplotlib colormap name for plotting. Default is 'grey'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the cepstral_coefficients function.
        """
        #TODO axis labels
        ceps = cepstral_coefficients(
            data, 
            sr, 
            window_length, 
            n_filters, 
            n_ceps, 
            pre_emphasis=pre_emphasis, 
            fmin=fmin,
            fmax=fmax,
            filterbank_type=filterbank_type,
            **kwargs)
        
        times = np.linspace(0, len(data) / sr, ceps.shape[0])
        plt.xlabel("Time [s]")
        plt.ylabel("Cepstral Coefficient Index")
        plt.imshow(ceps, origin="lower", aspect="auto", extent=(times[0], times[-1], 0, n_ceps), cmap=cmap)


def plot_features(
        data: ArrayLike, 
        sr: int,
    ) -> None:
    """
    Plot audio signal features using precomputed feature dictionary.

    Parameters
    ----------
    data : ArrayLike
        Audio time series data.
    sr : int
        Sampling rate of the audio data in Hz.
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)
    features = extract_all_features(data, sr)
    spec, times, freqs = spectrogram(data, sr)
    freq_ms, ms = spectrum(data, sr)#, mode="power")
    spectrogram_db = 20 * np.log10(np.abs(spec))
    #dynamic range to 100 dB
    spectrogram_db = np.maximum(spectrogram_db, -100)

    # Spectrogram with Dominant Frequencies
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(3, 1, 1)
    plt.title("Spectrogram with Dominant Frequencies")
    # plt.pcolormesh(times, freqs, spectrogram_db, shading='gouraud', cmap='grey')
    im = ax.imshow(spectrogram_db, aspect='auto', origin='lower', extent=(times[0], times[-1], freqs[0], freqs[-1]), cmap="grey")
    plt.scatter(times, features["dominant_freqs"], color=(0.7, 0.1, 0.1, 0.3), marker="o", label='Dominant Frequency')
    plt.colorbar(im, ax=ax, label="Amplitude [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency (Hz)")
    plt.legend()

    # Power Spectrum
    plt.subplot(3, 1, 2)
    plt.title("Magnitude Spectrum with Spectral Features")
    if not isinstance(freq_ms, np.ndarray):
        raise TypeError("Expected 'freqs_ps' to be an ndarray.")
    if not isinstance(freq_ms, np.ndarray):
        raise TypeError("Expected 'freqs_ps' to be an ndarray.")
    cutoff = len(freq_ms) // 3
    plt.plot(freq_ms[:-cutoff], ms[:-cutoff], label="Magnitude Spectrum", color='blue')
    
    plt.axvline(features["peak_frequency"], color='orange', linestyle="-", label="Peak Frequency (kHz)")
    plt.axvline(features["fq_median"], color='red', linestyle="--", label="Median")
    plt.axvline(features["fq_q1"], color='yellow', linestyle="--", label="Q1")
    plt.axvline(features["fq_q3"], color='purple', linestyle="--", label="Q3")
    plt.fill_betweenx(
        y=[0, max(ms)],
        x1=features["spectral_centroid"] - features["spectral_sd"],
        x2=features["spectral_centroid"] + features["spectral_sd"],
        color='grey',
        alpha=0.2,
        label='Bandwidth'
    )
    plt.axvline(features["spectral_centroid"], color='pink', linestyle="--", label="Centroid")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()

    # Waveform
    plt.subplot(3, 1, 3)
    plt.title("Waveform with Energy Envelope and Time-domain Features")
    times_waveform = np.linspace(0, len(data) / sr, num=len(data))
    plt.plot(times_waveform, data, label="Waveform", color="gray", alpha=0.3)
    plt.plot(times_waveform, features["amplitude_envelope"], label="Amplitude Envelope", color="blue")
    plt.axvline(features["t_median"], color='red', linestyle="--", label="Median")
    plt.axvline(features["t_q1"], color='yellow', linestyle="--", label="Q1")
    plt.axvline(features["t_q3"], color='purple', linestyle="--", label="Q3")
    plt.fill_betweenx(
        y=[0, np.max(features["amplitude_envelope"])],
        x1=features["temporal_centroid"] - features["temporal_sd"],
        x2=features["temporal_centroid"] + features["temporal_sd"],
        color='grey',
        alpha=0.2,
        label='Bandwidth'
    )
    plt.axvline(features["temporal_centroid"], color='pink', linestyle="--", label="Centroid")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_pitch_candidates(
        time_points : ArrayLike, 
        all_candidates : ArrayLike, 
        show_strongest : bool = True,
        tlim : Optional[Tuple[float, float]] = None,
        ax : Optional[Axes] = None
    ) -> Optional[Axes]:
    """
    Plot pitch candidates over time.
    
    Parameters
    ----------
    time_points : list of float
        Time stamps for each frame.
    all_candidates : list of list of tuple(float, float)
        List containing, for each frame, a list of (pitch, strength) tuples.
    show_strongest : bool
        If True, highlight the strongest voiced candidate per frame.
    """    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if all(isinstance(p, (int, float, np.number)) for p in all_candidates):
        all_candidates = [[(p, 1.0)] for p in all_candidates]

    # Plot all candidates
    for t, candidates in zip(time_points, all_candidates):
        if tlim and not (tlim[0] <= t <= tlim[1]):
            continue
        for pitch, _ in candidates:
            if pitch > 0:
                ax.plot(t, pitch, 'k.', alpha=0.3)

    # Optionally plot the strongest voiced candidate
    if show_strongest:
        times = []
        pitches = []
        for t, candidates in zip(time_points, all_candidates):
            if tlim and not (tlim[0] <= t <= tlim[1]):
                continue
            voiced = [c for c in candidates if c[0] > 0]
            if voiced:
                best = max(voiced, key=lambda x: x[1])
                times.append(t)
                pitches.append(best[0])
        ax.scatter(times, pitches, color=(0.7, 0.1, 0.1, 0.3), marker="o", label='Strongest pitch candidate')
    
    if tlim:
        ax.set_xlim(tlim)
        
    if ax is None:
        plt.title("Autocorrelation based pitch tracking")
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch [Hz]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    return ax


def plot_pitch_on_spectrogram(
    data : ArrayLike,
    sr : int,
    time_points : ArrayLike,
    all_candidates : ArrayLike,
    window_length : int = 512,
    overlap : int = 50,
    show_strongest : bool = True,
    db_scale : bool = True,
    flim : Optional[Tuple[float, float]] = None,
    tlim : Optional[Tuple[float, float]] = None,
    title : str = "Spectrogram with Pitch Candidates",
    cmap : str = 'grey',
    plot : Optional[Tuple[Figure, Axes]] = None,
) -> None:
    """
    Plot a spectrogram of the input audio data and overlay pitch candidates.

    This function computes and displays a spectrogram of the given audio data,
    then overlays pitch candidates over time. It can optionally highlight the
    strongest pitch candidate per time frame.

    Parameters
    ----------
    data : ArrayLike
        Audio time series data.
    sr : int
        Sampling rate of the audio data in Hz.
    time_points : ArrayLike
        Time stamps corresponding to each frame of pitch candidates.
    all_candidates : ArrayLike
        List or array of pitch candidate tuples (pitch, strength) for each time frame.
    window_length : int, optional
        Window length (in samples) for the spectrogram. Default is 512.
    overlap : int, optional
        Overlap between windows (in samples) for the spectrogram. Default is 50.
    show_strongest : bool, optional
        If True, highlights the strongest voiced pitch candidate per frame. Default is True.
    db_scale : bool, optional
        Whether to display the spectrogram in decibel scale. Default is True.
    flim : tuple of float, optional
        Frequency limits (min_freq, max_freq) to display in the spectrogram. Default is None (no limit).
    tlim : tuple of float, optional
        Time limits (start_time, end_time) for the plot. Default is None (full duration).
    title : str, optional
        Title of the plot. Default is "Spectrogram with Pitch Candidates".
    cmap : str, optional
        Colormap to use for the spectrogram. Default is 'grey'.
    plot : tuple of (Figure, Axes), optional
        Existing matplotlib Figure and Axes to plot on. If None, a new figure is created.
    """
    if plot is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else: 
        fig, ax = plot
    
    plot_spectrogram(
        data,
        sr,
        overlap=overlap,
        db_scale=db_scale, 
        cmap=cmap, 
        flim=flim,
        tlim=tlim,
        title=title,
        window_length=window_length,
        plot=(fig, ax)
    )

    plot_pitch_candidates(
        time_points=time_points, 
        all_candidates=all_candidates, 
        show_strongest=show_strongest,
        tlim=tlim,
        ax=ax,
        )
    
    plt.show()


def plot_boundaries_on_spectrogram(
    data : ArrayLike,
    sr : int,
    segments : List[Dict[str, float]],
    **kwargs : Any
    ) -> None:
    """
    Plot a spectrogram of the input audio data and overlay vertical lines indicating segment boundaries.

    Parameters
    ----------
    data : ArrayLike
        Audio time series data.
    sr : int
        Sampling rate of the audio data.
    segments : List[Dict[str, float]]
        A list of segment boundary dictionaries. Each dictionary should
        contain keys "begin" and "end", representing the start and end
        times (in seconds or frames, depending on the spectrogram scale).
    **kwargs : Any
        Additional keyword arguments passed to the spectrogram plotting function.
    """
    fig, ax = plt.subplots()
    
    plot_spectrogram(data, sr, plot=(fig, ax), **kwargs)

    tlim = kwargs.get("tlim", None)

    for segment in segments: 
        if tlim is None or (segment["begin"] >= tlim[0] and segment["end"] <= tlim[1]):
            ax.axvline(segment["begin"], color='green', linestyle="--")
            ax.axvline(segment["end"], color='orange', linestyle="-")
            
        # TODO labels
        
    plt.show()


def plot_spectrogram_catalogue(
    df: DataFrame,
    column: str = "waveform",
    per_page: int = 20,
    ncols: int = 2,
    save_dir: Optional[str] = None,
    show: bool = True,
    prefix: str = "spectrograms",
    title_columns: List[str] = ["filename"],
    **kwargs: Any
) -> None:
    """
    Plot a catalogue of spectrograms from a DataFrame.

    This function generates spectrogram plots for waveforms stored in a DataFrame.
    The plots are arranged in a paginated grid and can be saved to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing waveform data and associated metadata.
    column : str, default="waveform"
        Column name in `df` that contains the waveform arrays.
    per_page : int, default=20
        Number of spectrograms to display per page.
    ncols : int, default=2
        Number of subplot columns per page.
    save_dir : str or None, optional
        Directory to save the generated figure pages. If None, figures are not saved.
    show : bool, default=True
        If True, display the figures interactively. If False, figures are closed after saving.
    prefix : str, default="spectrograms"
        Prefix for saved figure filenames if `save_dir` is provided.
    title_columns : list of str, default=["filename"]
        List of column names from `df` to include in subplot titles, joined by a vertical bar (`|`).
    **kwargs : dict, optional
        Additional keyword arguments passed to the underlying `plot_spectrogram` function.

    Returns
    -------
    None
        This function does not return anything. It produces matplotlib figures,
        optionally displaying them or saving them to disk.

    Notes
    -----
    - The function expects each row of the DataFrame to contain:
        * `df[column]`: waveform array-like
        * `df["sr"]`: sampling rate (int)
    - If a waveform entry is `None`, its subplot will be empty.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     "ID": [1, 2],
    ...     "waveform": [wave1, wave2],
    ...     "sr": [16000, 16000]
    ... }
    >>> df = pd.DataFrame(data)
    >>> plot_spectrogram_catalogue(df, per_page=2, ncols=2, show=False, save_dir="plots")
    Saved plots/spectrograms_001.png
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    n = len(df)
    n_pages = int(np.ceil(n / per_page))

    for page in range(n_pages):
        start = page * per_page
        end = min((page + 1) * per_page, n)
        subset = df.iloc[start:end]

        nrows = int(np.ceil(len(subset) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 2*nrows))
        axes = np.atleast_1d(axes).reshape(nrows, ncols)

        for idx, (_, row) in enumerate(subset.iterrows()):
            r, c = divmod(idx, ncols)
            ax_spec = axes[r, c]   

            snippet = row[column]
            if snippet is None:
                ax_spec.axis("off")
                continue

            sr = row["sr"]

            plot_spectrogram(
                row["waveform"],
                sr,
                title="|".join(str(row[col]) for col in title_columns),
                plot=(fig, ax_spec), 
                **kwargs
            )

        plt.tight_layout()

        if save_dir is not None:
            filename = os.path.join(save_dir, f"{prefix}_{page+1:03d}.png")
            fig.savefig(filename, dpi=150)
            print(f"Saved {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)