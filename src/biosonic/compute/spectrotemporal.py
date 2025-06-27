from typing import Optional, Tuple, Union, Dict, Literal, Any
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import signal
from scipy.fft import fft, ifft, rfft
from scipy.fftpack import dct
import matplotlib.pyplot as plt

from .temporal import temporal_entropy
from .spectral import power_spectral_entropy
from .utils import check_signal_format, check_sr_format, window_signal
from ..filter import linear_filterbank, mel_filterbank, log_filterbank


def spectrogram(
    data: ArrayLike,
    sr: int,
    window_length: int = 512,
    window: Union[str, ArrayLike] = "hann",
    zero_padding: int = 0,
    overlap: float = 50,
    noisereduction: Optional[int] = None,
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
    zero_padding : int, optional
        Number of zeros to pad to each FFT window. Default is 0.
    overlap : float, optional
        Overlap between adjacent windows as a percentage (0–100). Default is 50.
    noisereduction : int or None, optional
        Noise reduction mode:
        - `None`: no noise reduction
        - `1`: subtract median across time
        - `2`: subtract median across frequency
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
    # TODO more scaling, dynamic range, dB transform, invert, etc 
    if window_length % 2 != 0:
        raise ValueError("'window_length' must be even")

    noverlap = int(window_length * overlap / 100)
    nfft = window_length + zero_padding

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
        nfft=nfft,
        padded=False,
        boundary=None
    )

    if complex_output:
        return Sx, t, f

    S_real = np.abs(Sx)

    # Noise reduction
    if noisereduction == 1:
        noise = np.median(S_real, axis=0)
        S_real = np.abs(S_real - noise[:, None])
    elif noisereduction == 2:
        noise = np.median(S_real, axis=1)
        S_real = np.abs(S_real - noise[None, :])

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
    data: ArrayLike,
    sr: int,
    window_length: int = 512,
    n_filters: int = 32,
    n_ceps: int = 16,
    pre_emphasis: float = 0.97,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    filterbank_type: Literal["mel", "linear", "log"] = "mel",
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
    **kwargs : dict
        Optional keyword arguments for the filter banks, e.g. corner frequency for mel.

    Returns
    -------
    np.ndarray
        Cepstral coefficient array of shape (n_ceps,).
    """
    def liftering(cc: np.ndarray, D: int = 22) -> np.ndarray:
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

    ceps = liftering(ceps)[:, :n_ceps]

    return np.asarray(ceps.T)[::-1]


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
    temporal_entropy : Computes the temporal_entropy of the data.
    power_spectral_entropy : Computes the temporal_entropy in the frequency domain.
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

    Parameters:
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
    
    Returns:
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