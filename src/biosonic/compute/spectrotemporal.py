from typing import Optional, Tuple, Union, Dict, Literal, Any, List
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import signal
from scipy.fft import fft, ifft, rfft
from scipy.fftpack import dct
from scipy.spatial.distance import cdist
from scipy.linalg import solve

from .temporal import temporal_entropy
from .spectral import power_spectral_entropy
from .utils import check_signal_format, check_sr_format, window_signal
from ..filter import linear_filterbank, mel_filterbank, log_filterbank


def spectrogram(
    data: ArrayLike,
    sr: int,
    window_length: int = 512,
    window: Union[str, ArrayLike] = "hann",
    overlap: float = 50,
    noisereduction: Optional[bool] = False,
    complex_output: bool = False,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute the spectrogram of a waveform. Uses scipy.signal.stft and is oriented on seewave.

    Parameters
    ----------
    data : ArrayLike
        Input signal as a 1D array.
    sr : int
        Sampling rate in Hz.
    window_length : int, optional
        Length of the window in samples. Must be even. Default is 512.
    window : str or tuple, optional
        Type of window to use (e.g., 'hann', 'hamming' of scipy.signal.windows) or a custom window array. 
        Defaults to 'hann'.
    overlap : float, optional
        Overlap between adjacent windows as a percentage (0–100). Default is 50.
    noisereduction : bool, optional
        Apply noise reduction:
        if True, subtract median from spectrogram values. Default is False.
    complex_output : bool, optional
        If True, return the complex STFT result. If False, return magnitude. Default is False.

    Returns
    -------
    Sx : np.ndarray
        Spectrogram array (complex or magnitude depending on `complex_output`).
    t : np.ndarray
        Time vector corresponding to the columns of `Sx`, in seconds.
    f : np.ndarray
        Frequency vector corresponding to the rows of `Sx`, in Hz.

    Raises
    ------
    ValueError
        If `window_length` is not an even number.

    References
    ----------
    [1] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, 
    Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, 
    Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, 
    Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis 
    Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. 
    Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 
    (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 
    261-272. DOI: 10.1038/s41592-019-0686-2.

    [2] J. Sueur, T. Aubin, C. Simonis (2008). “Seewave: a free modular tool for sound analysis and 
    synthesis.” Bioacoustics, 18, 213-226.
    """
    if window_length % 2 != 0:
        raise ValueError("'window_length' must be even")

    noverlap = int(window_length * overlap / 100)

    if isinstance(window, str):
        try:
            window = signal.windows.get_window(window, window_length)
        except ValueError as e:
            raise ValueError(f"Invalid window type: {window}") from e
    else:
        window = np.asarray(window) 
        if not isinstance(window, np.ndarray):
            raise TypeError("'window' must be either a string or a 1D NumPy array.")

    f, t, Sx = signal.stft(
        data,
        fs=sr,
        window=window,
        nperseg=window_length,
        noverlap=noverlap,
        nfft=window_length,
        padded=False,
        boundary=None
    )

    if complex_output:
        return Sx, t, f

    S_real = np.abs(Sx)

    # Noise reduction
    if noisereduction == True:
        S_real = S_real - np.median(S_real, axis=0)

    return S_real, t, f


def cepstrum(
        data: ArrayLike,  
        sr: int,
        mode : Literal ["amplitude", "power"] = "power",
    ) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the cepstrum of a real-valued time-domain signal.

    The cepstrum is computed by taking the inverse Fourier transform 
    of the logarithm of the magnitude spectrum. Depending on the `mode`, 
    this function returns either the amplitude or power cepstrum.

    Parameters
    ----------
    data : ArrayLike
        Input real-valued time-domain signal (1D array).
    sr : int
        Sampling rate of the signal in Hz.
    mode : {"amplitude", "power"}, optional
        Type of cepstrum to compute:
        - "amplitude" : Returns the absolute value of the inverse FFT of the log-magnitude spectrum.
        - "power"     : Returns the squared magnitude of the inverse FFT of the log-power spectrum.
        Default is "power".

    Returns
    -------
    cepstrum : ArrayLike
        The computed cepstrum (amplitude or power based on the selected mode).
    quefrencies : ArrayLike
        Array of quefrency values (in seconds), corresponding to each element in the cepstrum.

    References
    ----------
    Childers DG, Skinner DP, Kemerait RC. 1977 
    The cepstrum: A guide to processing. Proc. IEEE 65, 1428–1443. 
    (doi:10.1109/PROC.1977.10747)
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    if np.all(data == data[0]):
        raise ValueError("Cannot compute cepstrum of flat signal.")

    cepstrum_ = np.asarray([])

    if mode == "power":
        cepstrum_ = np.abs(ifft(np.log(np.abs(fft(data))**2)))**2

    elif mode == "amplitude":
        cepstrum_ = np.abs(ifft(np.log(np.abs(fft(data)))))
    
    else:
        raise ValueError(f"Invalid mode for cepstrum calculation: {mode}")

    quefrencies = np.arange(len(data)) / sr

    return cepstrum_, quefrencies


def cepstral_coefficients(
    data : ArrayLike,
    sr : int,
    window_length : int = 512,
    n_filters : int = 32,
    n_ceps : int = 16,
    pre_emphasis : float = 0.97,
    fmin : float = 0.0,
    fmax : Optional[float] = None,
    filterbank_type : Literal["mel", "linear", "log"] = "mel",
    skip_first : bool = True,
    **kwargs : Any
) -> ArrayLike:
    """
    Compute cepstral coefficients from a signal using the specified filter bank.

    Parameters
    ----------
    signal : ArrayLike
        Input time-domain signal.
    sr : int
        Sampling rate in Hz.
    window_length : int
        FFT size in samples.
    n_filters : int
        Number of filters in the filter bank.
    n_ceps : int
        Number of cepstral coefficients to return.
    pre_emphasis : float
        Pre-emphasis coefficient to apply. Default is 0.97
    fmin : float
        Minimum frequency for the filter bank.
    fmax : Optional[float]
        Maximum frequency for the filter bank. Defaults to Nyquist (sr/2).
    filterbank_type : {'mel', 'linear', 'log'}
        Type of filter bank to apply before DCT.
    skip_first : bool
        Wether to excluse the first cepstral coefficient. Defaults to True.
    **kwargs : dict
        Optional keyword arguments for the filter banks, e.g. corner frequency for mel.

    Returns
    -------
    np.ndarray
        Cepstral coefficient array of shape (n_ceps,).
    """
    def liftering(cc: ArrayLike, D: int = 22) -> ArrayLike:
        """
        Apply sinusoidal liftering to cepstral coefficients.

        Parameters:
        - cc: Cepstral coefficients (2D array: frames x coefficients)
        - D: Liftering parameter (default 22, typical for MFCCs)

        Returns:
        - Lifted cepstral coefficients
        """
        cc_lift = np.zeros(cc.shape)

        n = np.arange(1, cc_lift.shape[1] + 1)
        D = 22
        w = 1 + (D / 2) * np.sin(np.pi * n / D)

        return cc * w

    # pre-emphasis - from https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/
    data_preemphasized = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    # window and fft
    data_windowed = window_signal(data_preemphasized, sr, window_length)
    mag_frames = np.abs(rfft(data_windowed, window_length))
    pow_frames = (1/window_length) * mag_frames **2

    # filter bank selection
    if fmax is None:
        fmax = sr / 2

    if filterbank_type == "mel":
        fbanks, _ = mel_filterbank(n_filters, window_length, sr, fmin, fmax, **kwargs)
    elif filterbank_type == "linear":
        fbanks, _ = linear_filterbank(n_filters, window_length, sr, fmin, fmax)
    elif filterbank_type == "log":
        raise NotImplementedError("Log frequency scale is not yet implemented for cepstral coefficients in Version 0.")
        #fbanks, _ = log_filterbank(n_filters, window_length, sr, fmin, fmax, **kwargs)
    else:
        raise ValueError(f"Unknown filterbank_type: {filterbank_type}")

    audio_filtered = np.dot(pow_frames, fbanks.T)
    audio_filtered = np.where(audio_filtered == 0, np.finfo(float).eps, audio_filtered) # avoid 0 division
    audio_filtered = 20 * np.log10(audio_filtered)  # to dB

    # DCT to cepstral domain
    ceps = dct(audio_filtered, type=2, norm="ortho", axis=1)

    if skip_first: 
        ceps = liftering(ceps)[:, 1:n_ceps+1]  # skip C0
    else:
        ceps = liftering(ceps)[:, :n_ceps]

    return np.asarray(ceps.T)


def spectrotemporal_entropy(
        data: ArrayLike,  
        sr: int, 
        *args : Any, 
        **kwargs : Any
    ) -> float:
    """
    Compute the product of temporal_entropy and power spectral entropy of the input data.

    This function computes two information-theoretic measures — temporal temporal_entropy and
    power spectral entropy — using the same unit and multiplies their values 
    to produce a combined spectrotemporal complexity measure.

    Parameters
    ----------
    data : ArrayLike
        Input signal as a 1D ArrayLike.
    sr : int
        Sampling rate in Hz.
    unit : {"bits", "nat", "dits", "bans", "hartleys"}, optional
        The logarithmic base to use for temporal_entropy calculations.
        Default is "bits".
    *args : Any
        Additional positional arguments passed to `temporal_entropy` and `power_spectral_entropy`.
    **kwargs : Any
        Additional keyword arguments passed to `temporal_entropy` and `power_spectral_entropy`.

    Returns
    -------
    float
        The temporal_entropy of the input data.

    See Also
    --------
    temporal_entropy : Computes the temporal entropy of the data.
    power_spectral_entropy : Computes the spectral entropy of the data.
    """
    H_t, _ = temporal_entropy(data, *args, **kwargs)
    H_f, _ = power_spectral_entropy(data, sr, *args, **kwargs)
    return H_t * H_f


def dominant_frequencies(
        data: ArrayLike, 
        sr: int, 
        n_freqs: int = 1, 
        min_height: float = 0.05,
        min_distance: float = 0.05,
        min_prominence: float = 0.05,
        peak_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any, 
        **kwargs: Any
    ) -> NDArray[np.float64]: 
    """
    Extracts the dominant frequency or frequencies from each time frame of a spectrogram 
    based on the scipy.signal function find_peaks.

    Parameters
    ----------
    data : ArrayLike
        Input 1D audio signal.
    sr : int
        Sample rate of the input signal in Hz.
    n_freqs : Optional[int], default=3
        Number of dominant frequencies to extract per time frame.
        If 1, a 1D array is returned. If >1, a 2D array of shape (time_frames, n_freqs) is returned.
    min_height : Optional[float], default=0.05
        Minimum normalized height of a peak (as a fraction of the spectral magnitude range).
        Must be between 0 and 1.
    min_distance : Optional[float], default=0.05
        Minimum normalized distance between peaks (as a fraction of the total number of frequency bins).
        Must be between 0 and 1.
    min_prominence : Optional[float], default=0.05
        Minimum normalized prominence of a peak (as a fraction of the spectral magnitude range).
        Must be between 0 and 1.
    peak_kwargs : Optional[dict], default=None
        Additional keyword arguments passed directly to `scipy.signal.find_peaks`.
        These override any values calculated from `min_height`, `min_distance`, and `min_prominence`.
    *args, **kwargs :
        Additional arguments passed to the scipy.signal ShortTimeFFT class.
    
    Returns
    -------
    NDArray[np.float64]
        - If n_freqs == 1:
            1D array of shape (time_frames,) containing the dominant frequency per frame.
        - If n_freqs > 1:
            2D array of shape (time_frames, n_freqs) containing the top `n_freqs` dominant
            frequencies per frame. NaNs are used to pad frames with fewer than `n_freqs` detected peaks.
    """   
    if not (0.0 <= min_height <= 1.0):
        raise ValueError("min_height must be between 0 and 1")
    
    if not (0.0 <= min_distance <= 1.0):
        raise ValueError("min_distance must be between 0 and 1")
    
    if not (0.0 <= min_prominence <= 1.0):
        raise ValueError("min_prominence must be between 0 and 1")

    spec, times, freqs = spectrogram(data, sr, *args, **kwargs)
    spec_real = np.abs(spec)
    # num_frames = spec_real.shape[1]

    if n_freqs == 1:
        dominant_freqs = np.full(len(times), np.nan)
    else:
        dominant_freqs = np.full((len(times), n_freqs), np.nan)

    peak_kwargs = peak_kwargs or {}

    for t in range(len(times)):
        spectrum = spec_real[:, t]
        magnitude_range = float(np.max(spectrum)) - np.min(spectrum)

        default_peak_params = {
            "height" : magnitude_range*min_height, 
            "distance" : max(1, len(freqs)//(1/min_distance)), 
            "prominence" : magnitude_range*min_prominence
        }

        peak_params = {**default_peak_params, **peak_kwargs}
        peaks, _ = signal.find_peaks(spectrum, **peak_params)

        if len(peaks) > 0:
            sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
            top_peaks = sorted_peaks[:n_freqs]
            top_freqs = freqs[top_peaks]

            if n_freqs == 1:
                dominant_freqs[t] = top_freqs[0]
            else:
                dominant_freqs[t, :len(top_freqs)] = top_freqs

    return dominant_freqs


def zero_crossings(data: ArrayLike) -> NDArray[np.int64]:
    #TODO
    """
    Calculate the indices of zero crossings in a 1D signal.
    
    Parameters
    ----------
    data : ArrayLike
        Input 1D audio signal.
        
    Returns
    -------
    NDArray[np.int64]
        Indices where zero crossings occur.
    """
    pass

def zero_crossing_rate(
        data: ArrayLike, 
        frame_length: int = 2048, 
        hop_length: int = 512
        ) -> np.float32:
    # TODO
    """
    Calculate the zero crossing rate of a 1D signal.
    
    Parameters  
    ----------
    data : ArrayLike
        Input 1D audio signal.
    frame_length : int, optional
        Length of each frame in samples. Default is 2048.
    hop_length : int, optional
        Number of samples to advance between frames. Default is 512.
    
    Returns     
    -------
    np.float32
        Zero crossing rate of the signal.
    """
    pass


# --------- Tokuda ----------
# https://www.ritsumei.ac.jp/~isao/NLM/

def _variance(x: NDArray[np.float32], dim: int) -> float:
    """Standard deviation of signal after skipping first `dim` points."""
    return float(np.std(x[dim:], ddof=0))


def _nearest_neighbors(
        length: int, 
        dim: int, 
        ss: NDArray[np.float32], 
        c: int, 
        exclusion: int
    ) -> Any:
    """Find nearest neighbors for index c based on embedding distance."""
    # Build embedding windows
    target = ss[c - dim + 1 : c + 1][None, :]  # shape (1, dim)
    windows = [ss[l - dim + 1 : l + 1] for l in range(dim - 1, length - 1) if abs(l - c) > exclusion]
    windows = np.array(windows)

    # Compute distances and sort
    distances = cdist(target, windows, metric="euclidean").ravel()
    sorted_idx = np.argsort(distances)

    # Return indices aligned to original l values
    valid_indices = [l for l in range(dim - 1, length - 1) if abs(l - c) > exclusion]
    return np.array(valid_indices)[sorted_idx]


def lpc_estimate(
        dim: int, 
        nnn: int, 
        ss: NDArray[np.float32], 
        nb: NDArray[np.float32]
    ) -> Any:
    """Estimate LPC coefficients using nearest neighbors (solves Ax = b)."""
    # Cross-correlation vector
    vt = np.array([sum(ss[nb[k] - i] * ss[nb[k] + 1] for k in range(nnn)) for i in range(dim)])

    # Covariance matrix
    A = np.array([[sum(ss[nb[k] - i] * ss[nb[k] - j] for k in range(nnn))
                   for j in range(dim)] for i in range(dim)])

    # Solve A * wd = vt
    try:
        wd = solve(A, vt, assume_a="sym")  # covariance is symmetric
    except np.linalg.LinAlgError:
        wd = np.zeros(dim)  # fallback: predict nothing
    return wd


def SNR(
        length: int, 
        dim: int, 
        nnn: int, 
        xx: NDArray[np.float32], 
        exclusion: int
    ) -> float:
    ss = xx.copy()
    rr = np.zeros_like(ss)

    # Predict each point using local linear model
    for i in range(dim - 1, length - 1):
        rr[i + 1] = ss[i + 1]
        nb = _nearest_neighbors(length, dim, ss, i, exclusion)
        wd = lpc_estimate(dim, nnn, ss, nb)
        rr[i + 1] -= np.dot(wd, ss[i - dim + 1 : i + 1][::-1])

    # Variance of original and residual signals
    vs = _variance(ss, dim)
    vr = _variance(rr, dim)
    return float(10.0 * np.log10(vs / vr))


def tokuda_nlm(
        dim: int, 
        data: NDArray[np.float32], 
        exclusion: int = 15
        ) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Compute the DVS plot values and the nonlinear measure (NLM).

    Parameters
    ----------
    dim : int
        Embedding dimension.
    data : array-like
        Time series data (1D).
    exclusion : int, default=15
        Minimum temporal separation for nearest neighbors.

    Returns
    -------
    results : list of tuple of (float, float)
        List of (percentage of neighbors, SNR) pairs.
    nlm_value : float
        Nonlinear measure in dB (difference between max SNR and full-data SNR).
    """
    xx = np.asarray(data, dtype=float)
    length = len(xx)
    if length <= 2 * exclusion + dim + 1:
        raise ValueError("Time series too short for given dim and exclusion.")

    # step size for sweeping nearest neighbor counts
    step = max(1, int(0.025 * length))

    results: List[Tuple[float, float]] = []
    max_snr: Optional[float] = None

    # Sweep number of neighbors from dim+1 up to max allowed
    for nnn in range(dim + 1, length - dim - 2 * exclusion, step):
        snr = SNR(length, dim, nnn, xx, exclusion)
        p_nnn = 100.0 * nnn / length  # percentage of neighbors
        results.append((p_nnn, snr))
        if max_snr is None or snr > max_snr:
            max_snr = snr

    # Fallback if loop produced no results
    if max_snr is None:
        return [], float("nan"), float("nan")

    # Compute SNR using all admissible neighbors (upper bound)
    nnn = length - dim - 2 * exclusion - 1
    snr_all = SNR(length, dim, nnn, xx, exclusion)

    # Nonlinear measure = difference between maximal SNR and full-data SNR
    nlm_value = max_snr - snr_all

    return results, nlm_value, snr_all

# ----------- Tokuda End --------------


def calculate_dominant_frequency_features(
        data: ArrayLike, 
        sr: int, 
        **kwargs: Any
    ) -> Dict[str, Union[float, NDArray[np.float64]]]:
        """
        Calculate dominant frequency features.
        """
        dominant_freqs = dominant_frequencies(data, sr, n_freqs=1, **kwargs)
        
        # exclude 0 values (no peak detected) from calculations
        dom_freqs_detected = dominant_freqs[dominant_freqs > 0]
        min_dom : float
        max_dom : float
        min_dom, max_dom = float(np.min(dom_freqs_detected)), float(np.max(dom_freqs_detected))
        range_dom = max_dom - min_dom
        cumulative_diff : float = np.sum(np.abs(np.diff(dom_freqs_detected)))
        mod_dom = cumulative_diff / range_dom if range_dom > 0 else 0

        return {
            "mean_dom": np.mean(dom_freqs_detected),
            "min_dom": min_dom,
            "max_dom": max_dom,
            "range_dom": range_dom,
            "mod_dom": mod_dom
        }

def spectrotemporal_features(
        data: ArrayLike, 
        sr: int,
        n_dominant_freqs : int = 1,
        **kwargs : Any
    ) -> dict[str, Union[float, np.floating, NDArray[np.float64]]]:
    """
    Extracts a set of spectrotemporal features from a signal.

    Args:
        data : ArrayLike
            The input signal as a 1D ArrayLike.
        sr : int 
            Sampling rate of the signal in Hz.
        n_dominant_frequencies : int
            Number of dominant frequencies to extract. Default is 1.

    Retuns:
        dict
        {"spectrotemporal_entropy": float,
        "dominant_frequencies": ArrayLike}
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    features = {
        "spectrotemporal_entropy": spectrotemporal_entropy(data, sr),
        "dominant_freqs": dominant_frequencies(data, sr, n_dominant_freqs, **kwargs),
    }

    dom_freq_feats = calculate_dominant_frequency_features(data, sr)

    return {**features, **dom_freq_feats}