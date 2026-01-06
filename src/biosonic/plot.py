import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ModuleNotFoundError:
    plt = None  # type: ignore
    Axes = None  # type: ignore
    Figure = None  # type: ignore

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame

from biosonic.compute.spectral import spectrum
from biosonic.compute.spectrotemporal import cepstral_coefficients, cepstrum, spectrogram
from biosonic.compute.utils import check_signal_format, check_sr_format, extract_all_features
from biosonic.filter import mel_filterbank


def plot_spectrogram(
        data: ArrayLike,
        sr: Optional[int] = None,
        db_scale: bool = True,
        cmap: str = 'binary',
        title: Optional[str] = None,
        db_ref: Optional[float] = None,
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
        after: Optional[Literal["fant", "koenig", "oshaughnessy", "umesh"]] = "oshaughnessy",
        plot: Optional[Any] = None,
        show_amplitude_bar: Optional[bool] = True,
        **kwargs: Any
    ) -> Any:
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
        Number of frequency bands for mel scaling. Default is 40.
    corner_frequency : float, optional
        Corner frequency for perceptual frequency scaling (used in mel scale).
    after (Literal): Choice of Mel scale formula to use.
            - 'fant': Classic formula with `a=b=1000`: `F_m = a * np.log(1 + f / b)`
            - 'koenig': (Not yet implemented)
            - 'oshaughnessy' or 'beranek': Commonly used formula as fant, but with `a=2595`, `b=700`
            - 'umesh': Formula using rational function with `a=0.0004`, `b=0.603`.
    plot : tuple(matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        Existing matplotlib Figure and Axes objects to plot into. If None, a new figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to `matplotlib.pyplot.imshow`.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

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

    if freq_scale == "mel":
        if sr is None:
            raise ValueError("Sample rate must be provided for mel frequency scale.")

        fmin = flim[0] if flim else 0.0
        fmax = flim[1] if flim and flim[1] else sr / 2

        fb, f_centers = mel_filterbank(n_bands, window_length, sr, fmin=fmin, fmax=fmax, corner_frequency=corner_frequency, after=after)
        f = f_centers
        # Sx : np.ndarray = np.einsum("...ft,mf->...mt", Sx, fb, optimize=True)
        Sx = fb @ Sx

    # Apply dB scale
    if db_scale:
        ref = np.max(Sx) if db_ref is None else db_ref
        Sx = 20 * np.log10(Sx / ref + 1e-30)  # add small value to avoid log(0)

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
        ax.set_ylim(float(min(f[f > 0])), float(f[-1]))

    if title is None:
        title = f"{freq_scale}-scaled spectrogram"
    ax.set_title(title)

    if show_amplitude_bar:
        fig.colorbar(im, ax=ax, label=("Amplitude [dB]" if db_scale else "Magnitude"))
    plt.tight_layout()

    if plot is None:
        plt.show()

    return fig, ax


def plot_cepstrum(
        data: ArrayLike,
        sr: int,
        min_quefrency: Optional[float] = None,
        max_quefrency: Optional[float] = None,
        log_scale: bool = False,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs: Any
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
        Wether to log-scale the y-axis. Defaults to False.
    ylim : tuple, optional
        Limits for the y-axis.
    title : str or None
        Title for the plot. If None, a default will be generated.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

    ceps, quefs = cepstrum(data, sr, **kwargs)

    if max_quefrency is None:
        max_quefrency = len(data) / sr

    if min_quefrency is None:
        min_quefrency = 0

    mask = (quefs >= min_quefrency) & (quefs <= max_quefrency)

    plt.plot(quefs[mask], ceps[mask], color='steelblue')
    plt.xlabel("Quefrency (s)")
    if log_scale:
        plt.yscale("log")
    if ylim:
        plt.ylim(ylim)
    plt.ylabel("Amplitude")
    plt.title(title or f"Cepstrum (Sampling rate: {sr} Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cepstral_coefficients(
        data: ArrayLike,
        sr: int,
        window_length: int,
        n_filters: int = 32,
        n_ceps: int = 40,
        pre_emphasis: float = 0.97,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        filterbank_type: Literal["mel", "linear", "log"] = "mel",
        cmap: Optional[str] = "grey",
        **kwargs: Any
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
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

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
        features: Optional[dict[str, Any]] = None,
        spec_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
    """
    Plot audio signal features using precomputed feature dictionary.

    Parameters
    ----------
    data : ArrayLike
        Audio time series data.
    sr : int
        Sampling rate of the audio data in Hz.
     **kwargs : dict[str, Any]
            Optional parameters for dominant frequency estimation.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

    data = check_signal_format(data)
    sr = check_sr_format(sr)

    if not features:
        features = extract_all_features(data, sr, **kwargs)

    if "trim_indices" not in features:
        features["trim_indices"] = (0, len(data))
        features["trim_times"] = (0, len(data) / sr)

    _, times, _ = spectrogram(data, sr)
    freq_ms, ms = spectrum(data, sr)

    dom_freqs = features["dominant_freqs"]
    all_candidates = [[(float(f), 1.0) if f > 0 else (0.0, 0.0)] for f in dom_freqs]

    # Spectrogram with Dominant Frequencies
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(3, 1, 1)

    if spec_kwargs is None:
        spec_kwargs = {}

    plot_pitch_on_spectrogram(
        data=data,
        sr=sr,
        time_points=times,
        all_candidates=all_candidates,
        show_strongest=True,
        db_scale=True,
        title="Spectrogram with Dominant Frequencies",
        cmap="binary",
        plot=(fig, ax1),
        **spec_kwargs
    )

    # Spectrum
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Magnitude Spectrum with Spectral Features")
    if not isinstance(freq_ms, np.ndarray):
        raise TypeError("Expected 'freqs_ps' to be an ndarray.")
    if not isinstance(freq_ms, np.ndarray):
        raise TypeError("Expected 'freqs_ps' to be an ndarray.")
    cutoff = len(freq_ms) // 3
    ax2.plot(freq_ms[:-cutoff], ms[:-cutoff], label="Magnitude Spectrum", color="#A2A2A2")

    ax2.axvline(features["peak_frequency"], color="#7951A2C6", linestyle="-", label="Peak Frequency (kHz)")
    ax2.axvline(features["fq_median"], color="#48ad46b5", linestyle="-", label="Median")
    ax2.axvline(features["fq_q1"], color="#88d253aa", linestyle="-", label="Q1")
    ax2.axvline(features["fq_q3"], color="#267746A9", linestyle="-", label="Q3")
    ax2.fill_betweenx(
        y=[0, max(ms)],
        x1=features["spectral_centroid"] - features["spectral_sd"],
        x2=features["spectral_centroid"] + features["spectral_sd"],
        color='grey',
        alpha=0.2,
        label='Bandwidth'
    )
    ax2.axvline(features["spectral_centroid"], color="#00BFFF7E", linestyle="-", label="Centroid")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Magnitude")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Waveform
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Waveform with Amplitude Envelope and Time-domain Features")
    times_waveform = np.linspace(0, len(data) / sr, num=len(data))
    ax3.plot(times_waveform, data, label="Waveform", color="grey", alpha=0.3)
    ax3.plot(times_waveform[features["trim_indices"][0]:features["trim_indices"][1]], features["amplitude_envelope"],
             label="Amplitude Envelope", color="#A2A2A2")
    if features["trim_times"][0] >= 0 or features["trim_times"][1] <= len(data) / sr:
        ax3.axvspan(features["trim_times"][0], features["trim_times"][1], color="#696969C5", label="Processed Region", alpha=0.1)
    ax3.axvline(features["t_median"]+features["trim_times"][0], color="#48ad46b5", linestyle="-", label="Median")
    ax3.axvline(features["t_q1"]+features["trim_times"][0], color="#88d253aa", linestyle="-", label="Q1")
    ax3.axvline(features["t_q3"]+features["trim_times"][0], color="#267746A9", linestyle="-", label="Q3")
    ax3.axvspan(
        features["temporal_centroid"] - features["temporal_sd"],
        features["temporal_centroid"] + features["temporal_sd"],
        color='grey',
        alpha=0.2,
        label='Bandwidth'
    )
    ax3.axvline(features["temporal_centroid"], color="#00BFFF7E", linestyle="-", label="Centroid")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plt.show()


def plot_pitch_candidates(
        time_points: ArrayLike,
        all_candidates: ArrayLike,
        show_strongest: bool = True,
        tlim: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None
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
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

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
        return None

    return ax


def plot_pitch_on_spectrogram(
    data: ArrayLike,
    sr: int,
    time_points: ArrayLike,
    all_candidates: ArrayLike,
    window_length: int = 512,
    overlap: int = 50,
    show_strongest: bool = True,
    db_scale: bool = True,
    flim: Optional[Tuple[float, float]] = None,
    tlim: Optional[Tuple[float, float]] = None,
    title: str = "Spectrogram with Pitch Candidates",
    cmap: str = "binary",
    plot: Optional[Tuple[Figure, Axes]] = None,
    **kwargs: Any
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
        Colormap to use for the spectrogram. Default is 'binary'.
    plot : tuple of (Figure, Axes), optional
        Existing matplotlib Figure and Axes to plot on. If None, a new figure is created.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

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
        plot=(fig, ax),
        **kwargs
    )

    plot_pitch_candidates(
        time_points=time_points,
        all_candidates=all_candidates,
        show_strongest=show_strongest,
        tlim=tlim,
        ax=ax,
        )

    if plot is None:
        plt.show()


def plot_boundaries_on_spectrogram(
    data: ArrayLike,
    sr: int,
    segments: List[Dict[str, float]],
    **kwargs: Any
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
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

    fig, ax = plt.subplots()

    plot_spectrogram(data, sr, plot=(fig, ax), **kwargs)

    tlim = kwargs.get("tlim", None)

    for segment in segments:
        if tlim is None or (segment["begin"] >= tlim[0] and segment["end"] <= tlim[1]):
            ax.axvline(segment["begin"], color="#95c5478d", linestyle="-")
            ax.axvline(segment["end"], color="#ffa2007f", linestyle="-")

        plt.legend(["Segment Start", "Segment End"])

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
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install biosonic[plot]")

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
                show_amplitude_bar=False,
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
