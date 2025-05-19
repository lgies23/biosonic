import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import fft, signal
from scipy.stats import gmean
from typing import Optional, Tuple, Union, Any, Dict
import warnings
from .utils import exclude_trailing_and_leading_zeros, check_signal_format, check_sr_format, cumulative_distribution_function


def spectrum(data: ArrayLike, 
             sr: Optional[int] = None,
             mode: Union[str, int, float] = 'amplitude') -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Computes the magnitude spectrum of a signal, allowing for amplitude, power,
    or arbitrary exponentiation of the magnitude.

    Args:
        data : ArrayLike
            The input time-domain signal as a 1D array-like. 
        sr: Optional Integer, default=None
            Sampling rate in Hz as an integer. If given, returns the frequency bins 
            of the magnitude spectrum. Defaults to None.
        mode : Union[str, int], default='amplitude'
            Specifies how to compute the spectrum:
            - 'amplitude': return the amplitude spectrum (|FFT|).
            - 'power': return the power spectrum (|FFT|^2).
            - int or float: raise the magnitude to the given power (e.g., 3 for |FFT|^3).

    Retuns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            A tuple (frequencies, spectrum), where:
            - frequencies: If sr is provided. 1D array of frequency bins corresponding to the spectrum.
            - spectrum: 1D array of the transformed frequency-domain representation (magnitude raised to the specified power).
    
    Raises:
        ValueError
            If `mode` is a string but not one of the supported options.
        TypeError
            If `mode` is not a string, int, or float.
    """
    data = check_signal_format(data)

    freqs = None

    if data.size == 0:
        warnings.warn("Input signal is empty; returning an empty spectrum.", RuntimeWarning)
        return freqs, np.array([], dtype=np.float64)
    
    if sr is not None:
        sr = check_sr_format(sr)
        freqs = fft.rfftfreq(len(data), d=1/sr)
    
    magnitude_spectrum = np.abs(fft.rfft(data))

    if isinstance(mode, str):
        mode = mode.lower()
        if mode == 'amplitude':
            return freqs, magnitude_spectrum
        elif mode == 'power':
            return freqs, magnitude_spectrum ** 2
        else:
            raise ValueError(f"Invalid string mode '{mode}'. Use 'amplitude', 'power', or an integer.")
    elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return freqs, magnitude_spectrum ** mode
    else:
        raise TypeError(f"'mode' must be a string, int or float, not {type(mode).__name__}.")


def spectrogram(
        data: ArrayLike, 
        sr: int, 
        window: Union[str, ArrayLike] = "hann", 
        window_length: int = 1024, 
        overlap: float = .5, 
        scaling: str = "magnitude",
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the spectrogram of a 1D signal using short-time Fourier transform (STFT).

    Args:
        data : ArrayLike
            Input signal as a 1D array-like.
        sr : int
            Sampling rate of the signal in Hz.
        window : Union[str, NDArray], optional
            Type of window to use (e.g., 'hann', 'hamming' of scipy.signal.windows) or a custom window array.
            Defaults to 'hann'.
        window_length : int, optional
            Length of the analysis window in samples.
            Defaults to 1024. Ignored if the window is given as a custom array,
        overlap : float, optional
            Fractional overlap between adjacent windows (0 < overlap < 1).
            Defaults to 0.5 (50% overlap).
        scaling : str, optional
            scaling of the spectrogram, either 'magnitude' or 'psd' for power spectral density. 
            Defaults to 'magnitude'.
        *args : tuple
            Additional positional arguments to pass to scipy's ShortTimeFFT class.
        **kwargs : dict
            Additional keyword arguments to pass to scipy's ShortTimeFFT class.

    Retuns:
        Sx : NDArray[np.float64]
            2D array of complex STFT values (frequency x time).
        t : NDArray[np.float64]
            Array of time values corresponding to the STFT columns.
        f : NDArray[np.float64]
            Array of frequency bins corresponding to the STFT rows.

    Notes:
        This function relies on scipy's `ShortTimeFFT` class and the submodule scipy.signal.windows.
    
    References:
        Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, 
        Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, 
        Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, 
        Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis 
        Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. 
        Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 
        (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 
        261-272. DOI: 10.1038/s41592-019-0686-2.
    """
    # todo more scaling, dynamic range, dB transform, invert, etc 
    data = check_signal_format(data)
    check_sr_format(sr)

    if 0 > overlap > 1 or not isinstance(overlap, float):
        raise ValueError("Window overlap must be a float between 0 and 1.")

    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("'window_length' must be a positive integer.")

    if isinstance(window, str):
        try:
            window = signal.windows.get_window(window, window_length)
        except ValueError as e:
            raise ValueError(f"Invalid window type: {window}") from e
    elif not isinstance(window, np.ndarray):
        raise TypeError("'window' must be either a string or a 1D NumPy array.")
    
    hop_length = int(window_length * (1 - overlap))

    STFT = signal.ShortTimeFFT(window, hop_length, sr, scale_to=scaling, *args, **kwargs)
    Sx = STFT.stft(data, *args, **kwargs)

    return Sx, STFT.t(len(data)), STFT.f


def spectral_quartiles(data: ArrayLike, sr: int) -> Tuple[float, float, float]:
    """
    Compute the 1st, 2nd (median), and 3rd quartiles of the power spectrum of a signal.

    Args:
        data : ArrayLike
            Input signal as a 1D array-like.
        sr : int
            Sampling rate (in Hz).

    Retuns:
        q1 : float
            Frequency at which the cumulative power spectrum reaches the 25% mark.
        q2 : float
            Frequency at which the cumulative power spectrum reaches the 50% mark (median frequency).
        q3 : float
            Frequency at which the cumulative power spectrum reaches the 75% mark.

    Raises:
        ValueError
            If the input signal is empty.
        ValueError
            If the input signal contains only zeros.
        ValueError
            If the spectrum output is inconsistent (e.g., mismatched lengths).

    Notes:
        This function calculates spectral quartiles using the cumulative distribution function (CDF)
        of the signal's power spectrum. Quartiles are determined by finding the frequencies
        at which the CDF crosses 25%, 50%, and 75%.

    See Also:
        spectrum : Computes the frequency and spectral envelope of a signal.
        cumulative_distribution_function : Computes the normalized cumulative sum of a spectrum.

    Examples:
        >>> import numpy as np
        >>> from mymodule import spectral_quartiles
        >>> sr = 1000
        >>> t = np.linspace(0, 1, sr, endpoint=False)
        >>> x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
        >>> spectral_quartiles(x, sr)
        (10.0, 20.0, 30.0)
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)
    if len(data) == 0:
        raise ValueError("Input is empty")
    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")
    
    frequencies, envelope = spectrum(data, sr=sr, mode="power")
    cdf = cumulative_distribution_function(envelope)

    if frequencies is None or len(frequencies) != len(envelope): 
        raise ValueError("Freuency bins don't match envelope") 
    
    return frequencies[np.searchsorted(cdf, 0.25)], frequencies[np.searchsorted(cdf, 0.5)], frequencies[np.searchsorted(cdf, 0.75)]


def flatness(data: ArrayLike) -> Union[float, np.floating[Any]]:
    """
    Compute the spectral flatness (also known as Wiener entropy) of a signal.

    Spectral flatness is a measure of how noise-like a signal is. A flatness close to 1 
    indicates a flat (white noise-like) spectrum, whereas a value close to 0 indicates 
    a peaky (tonal) spectrum.

    Args:
        data : ArrayLike
            Time-domain input signal (1D array-like).

    Retuns:
        float
            Spectral flatness value, defined as the ratio of the geometric mean to the arithmetic mean 
            of the signal's power spectrum (excluding leading/trailing zeros).

    Raises:
        ValueError
            If input signal is empty, contains only zeros, or the computed flatness is not a float (from np.mean and scipy's gmean).

    References:
        Sueur, J. (2018). *Sound Analysis and Synthesis with R*. Springer International Publishing, p. 299.
        https://doi.org/10.1007/978-3-319-77647-7
    """
    data = check_signal_format(data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, envelope = spectrum(data, mode="power")

    ps_wo_zeros = exclude_trailing_and_leading_zeros(envelope)
    if len(ps_wo_zeros) == 0:
        raise ValueError("Input signal contained only zero values")
    
    flatness_ = gmean(ps_wo_zeros) / np.mean(ps_wo_zeros)
    if not np.isscalar(flatness_) or not isinstance(flatness_, (float, np.floating)):
        raise ValueError(f"Received wrong data type for spectral flatness: {type(flatness_)}")
    return flatness_


def spectral_moments(
        data: ArrayLike, 
        sr: int
        ) -> Tuple[
            Union[float, np.floating[Any]],
            Union[float, np.floating[Any]],
            Union[float, np.floating[Any]],
            Union[float, np.floating[Any]]
            ]:
    """
    Args:
    data : ArrayLike
        1D array-like representing the input signal.
    sampling_rate : float 
        Sampling rate of the signal in Hz.

    Retuns:
        float or np.floating
            Spectral centroid in Hz
        float or np.floating
            Spectral bandwidth in Hz
        float or np.floating
            Spectral skewness
        float or np.floating
            Spectral kurtosis
            

    References:
        Klapuri A, Davy M. 2006 Signal processing methods for music transcription. 
        New York: Springer. p.136
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")

    freqs, ms = spectrum(data, sr=sr)
    centroid_ = np.average(freqs, weights=ms)
    bandwidth_ = np.sqrt(np.sum(ms * (freqs-centroid_)**2))
    if bandwidth_ == 0:
        warnings.warn("Bandwidth of signal is 0, returning NaN for skewness and kurtosis", RuntimeWarning)
        return centroid_, bandwidth_, np.nan, np.nan
    skewness_ = (np.sum(ms * (freqs-centroid_)**3))/(bandwidth_**3)
    kurtosis_ = (np.sum(ms * (freqs-centroid_)**4))/(bandwidth_**4)

    return centroid_, bandwidth_, skewness_, kurtosis_


def centroid(data: ArrayLike, sr: int)-> Union[float, np.floating[Any]]:
    r"""
    Compute the spectral centroid of a signal.

    The spectral centroid represents the "center of mass" of the power spectrum,
    giving a measure of where the energy of the spectrum is concentrated. It is 
    calculated as the first spectral moment, or the weighted average of the frequency components, using the 
    magnitude spectrum as weights: 
        
        .. math::
            C_f=\sum_{k \epsilon K_+}k X(k)
    
    Args:
        data : ArrayLike
            Input time-domain signal. Must be one-dimensional and convertible to a NumPy array.
        
        sr : int
            Sampling rate of the input signal in Hz.

    Retuns:
        float or np.floating
            Spectral centroid in Hz. A higher value indicates that the signal's energy 
            is biased toward higher frequencies.

    Raises:
        ValueError
            If the computed centroid is not a scalar floating-point value.

    Examples:
        >>> signal = np.random.randn(1024)
        >>> sr = 44100
        >>> centroid(signal, sr)
        7101.56

    References:
        Klapuri A, Davy M. 2006 Signal processing methods for music transcription. 
        New York: Springer. p.136
    """
    centroid_,_,_,_ = spectral_moments(data, sr)

    return centroid_


def bandwidth(data: ArrayLike, sr: int) -> Union[float, np.floating[Any]]:
    r"""
    Compute the mean spectral bandwidth (standard deviation or second spectral moment) of a signal. 
    It is calculated as

        .. math::
            S_f=S_f=\sqrt{\sum_{k \epsilon K_+}(k-C_f)^2 X(k)}
        
    Args:
        data : ArrayLike
            Input signal as a 1D array-like.
        sr : int
            Sampling rate of the signal, in Hz.

    Retuns:
        float or np.floating
            The standard deviation of the signal.
    
    Examples:
        >>> import numpy as np
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> bandwidth(signal, sr=1)
        1.4142135623730951

    References:
        Klapuri A, Davy M. 2006 Signal processing methods for music transcription. 
        New York: Springer. p.136
    """
    _,bandwidth_,_,_ = spectral_moments(data, sr)

    return bandwidth_


def skewness(data: ArrayLike, sr: int) -> Union[float, np.floating[Any]]:
    r"""
    Compute the spectral skewness (third spectral moment) of a signal. 
    The skewness describes the asymmetry of the spectrum 
    around the spectral centroid and is calculated as

        .. math::
            \gamma_1=\frac{\sum_{k \epsilon K_+}(k-C_f)^3 X(k)}{S_f^3}
        
    Args:
    data : ArrayLike
        Input signal as a 1D array-like.
    sr : int
        Sampling rate of the signal, in Hz.

    Retuns:
        float or np.floating
            The skewness of the signal.
    
    References:
        Klapuri A, Davy M. 2006 Signal processing methods for music transcription. 
        New York: Springer. p.136
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    _,_,skew,_ = spectral_moments(data, sr)
    return skew


def kurtosis(data: ArrayLike, sr: int) -> Union[float, np.floating[Any]]:
    r"""
    Compute the spectral kurtosis (fourth spectral moment) of a signal. 
    The skewness describes the 'peakedness' of the spectrum 
    and is calculated as

        .. math::
            \gamma_2=\frac{\sum_{k \epsilon K_+}(k-C_f)^4 X(k)}{S_f^4}
        
    Args:
    data : ArrayLike
        Input signal as a 1D array-like.
    sr : int
        Sampling rate of the signal, in Hz.

    Retuns:
        float or np.floating
            The kurtosis of the signal.
    
    References:
        Klapuri A, Davy M. 2006 Signal processing methods for music transcription. 
        New York: Springer. p.136
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    _,_,_,kurt = spectral_moments(data, sr)
    return kurt


def peak_frequency(data: ArrayLike, sr: int) -> Union[float, np.floating[Any]]:
    """
    Computes the peak frequency of a signal using the Fourier transform.

    The function applies the Fast Fourier Transform (FFT) to the input signal to obtain its frequency spectrum.
    It identifies the frequency corresponding to the maximum magnitude in the spectrum, which represents the dominant
    or peak frequency of the signal.

    Args:
    data : ArrayLike
        1D array-like representing the input signal.
    sampling_rate : float 
        Sampling rate of the signal in Hz.

    Retuns:
        float
            The peak frequency in Hz.

    Example:
        >>> import numpy as np
        >>> from biosonic.compute.temporal import peak_frequency
        >>> sampling_rate = 1000.0  # 1000 Hz
        >>> t = np.linspace(0, 1.0, int(sampling_rate), endpoint=False)
        >>> signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
        >>> freq = peak_frequency(signal, sampling_rate)
        >>> print(freq)
        50.0

    Notes:
        - The function assumes the input signal is real-valued and uniformly sampled.
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    if data.size == 0:
        raise ValueError("Input signal is empty; could not determine peak frequency.")

    freqs, ps = spectrum(data, sr=sr, mode="power")
    assert freqs is not None
    return float(freqs[np.argmax(ps)])


def dominant_frequencies(
        data: ArrayLike, 
        sr: int, 
        n_freqs: int = 3, 
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

    spec, _, freqs = spectrogram(data, sr, *args, **kwargs)
    spec_real = np.abs(spec)
    num_frames = spec_real.shape[1]
    if n_freqs == 1:
        dominant_freqs = np.full(num_frames, np.nan)
    else:
        dominant_freqs = np.full((num_frames, n_freqs), np.nan)

    peak_kwargs = peak_kwargs or {}

    for t in range(num_frames):
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


def spectral_features(data: ArrayLike, 
                      sr: int,
                      n_freqs: int=3,
                      ) -> Dict[str, Union[float, np.floating, NDArray[np.float64]]]:
    # TODO adjust for new functions
    """
    Extracts a set of spectral features from a signal.

    Args:
        data : ArrayLike
            The input signal as a 1D ArrayLike.
        sr : int 
            Sampling rate of the signal in Hz.

    Retuns:
        dict
        {"fq_q1": float,
        "fq_median": float,
        "fq_q3": float,
        "spectral_flatness": float,
        "centroid": float,
        "spectral_std": float,
        "peak_frequency": float,
        "dominant_frequencies": NDArray[np.float64]}
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    fq_q1_bin, fq_median_bin, fq_q3_bin = spectral_quartiles(data, sr)
    
    features = {
        "fq_q1": fq_q1_bin,
        "fq_median": fq_median_bin,
        "fq_q3": fq_q3_bin,
        "spectral_flatness": flatness(data),
        "centroid": centroid(data, sr),
        "spectral_std": bandwidth(data, sr),
        "peak_frequency": peak_frequency(data, sr),
        "dominant_frequencies": dominant_frequencies(data, sr, n_freqs=n_freqs)
    }

    return features