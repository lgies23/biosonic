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

    Parameters
    ----------
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

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple (frequencies, spectrum), where:
        - frequencies: If sr is provided. 1D array of frequency bins corresponding to the spectrum.
        - spectrum: 1D array of the transformed frequency-domain representation (magnitude raised to the specified power).
    
    Raises
    ------
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

    Parameters
    ----------
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

    Returns
    -------
    Sx : NDArray[np.float64]
        2D array of complex STFT values (frequency x time).
    t : NDArray[np.float64]
        Array of time values corresponding to the STFT columns.
    f : NDArray[np.float64]
        Array of frequency bins corresponding to the STFT rows.

    Notes
    -----
    This function relies on scipy's `ShortTimeFFT` class and the submodule scipy.signal.windows.
    
    References:
    -----
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

    Parameters
    ----------
    data : ArrayLike
        Input signal as a 1D array-like.
    sr : int
        Sampling rate (in Hz).

    Returns
    -------
    q1 : float
        Frequency at which the cumulative power spectrum reaches the 25% mark.
    q2 : float
        Frequency at which the cumulative power spectrum reaches the 50% mark (median frequency).
    q3 : float
        Frequency at which the cumulative power spectrum reaches the 75% mark.

    Raises
    ------
    ValueError
        If the input signal is empty.
    ValueError
        If the input signal contains only zeros.
    ValueError
        If the spectrum output is inconsistent (e.g., mismatched lengths).

    Notes
    -----
    This function calculates spectral quartiles using the cumulative distribution function (CDF)
    of the signal's power spectrum. Quartiles are determined by finding the frequencies
    at which the CDF crosses 25%, 50%, and 75%.

    See Also
    --------
    spectrum : Computes the frequency and spectral envelope of a signal.
    cumulative_distribution_function : Computes the normalized cumulative sum of a spectrum.

    Examples
    --------
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

    Parameters
    ----------
    data : ArrayLike
        Time-domain input signal (1D array-like).

    Returns
    -------
    float
        Spectral flatness value, defined as the ratio of the geometric mean to the arithmetic mean 
        of the signal's power spectrum (excluding leading/trailing zeros).

    Raises
    ------
    ValueError
        If input signal is empty, contains only zeros, or the computed flatness is not a float (from np.mean and scipy's gmean).

    References
    ----------
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


def centroid(data: ArrayLike, sr: int)-> Union[float, np.floating[Any]]:
    """
    Compute the spectral centroid of a signal.

    The spectral centroid represents the "center of mass" of the power spectrum,
    giving a measure of where the energy of the spectrum is concentrated. It is 
    calculated as the weighted average of the frequency components, using the 
    magnitude spectrum as weights.

    Parameters
    ----------
    data : ArrayLike
        Input time-domain signal. Must be one-dimensional and convertible to a NumPy array.
    
    sr : int
        Sampling rate of the input signal in Hz.

    Returns
    -------
    centroid : float or np.floating
        Spectral centroid in Hz. A higher value indicates that the signal's energy 
        is biased toward higher frequencies.

    Raises
    ------
    ValueError
        If the computed centroid is not a scalar floating-point value.

    Examples
    --------
    >>> signal = np.random.randn(1024)
    >>> sr = 44100
    >>> centroid(signal, sr)
    7101.56
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    freqs, ms = spectrum(data, sr=sr)
    centroid_ = np.average(freqs, weights=ms)

    if not np.isscalar(centroid_) or not isinstance(centroid_, (float, np.floating)):
        raise ValueError(f"Received wrong data type for spectral centroid: {type(centroid_)}")

    return centroid_


def bandwidth(data: ArrayLike, sr: int) -> Union[float, np.floating[Any]]:
    """
    Compute the mean spectral bandwidth (standard deviation) of a signal.

    Parameters
    ----------
    data : ArrayLike
        Input signal as a 1D array-like.
    sr : int
        Sampling rate of the signal, in Hz.

    Returns
    -------
    float
        The standard deviation of the signal.
    
    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> standard_deviation(signal, sr=1)
    1.4142135623730951
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    freqs, ms = spectrum(data, sr=sr)
    centroid_ = np.average(freqs, weights=ms)

    bandwidth_ = np.sqrt(np.sum(ms * (freqs-centroid_)**2))

    if not np.isscalar(bandwidth_) or not isinstance(bandwidth_, (float, np.floating)):
        raise ValueError(f"Received wrong data type for spectral bandwidth: {type(bandwidth_)}")

    return bandwidth_


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

    Returns:
        float: The peak frequency in Hz.

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


def dominant_frequencies(data: ArrayLike, sr: int, n_freqs: Optional[int] = 1, *args: Any, **kwargs: Any) -> NDArray[np.float64]:    
    # TODO docs, tests, n_freqs, parameters
    spec, _, freqs = spectrogram(data, sr, *args, **kwargs)
    
    dominant_freqs = []

    # Iterate over time frames
    for t in range(spec.shape[1]):
        # Get spectrum for time frame
        spectrum = spec[:, t]
        magnitude_range = float(np.max(spectrum)) - np.min(spectrum)
        peaks, _ = signal.find_peaks(spectrum, height=magnitude_range*0.05, distance=len(freqs)//50, prominence=magnitude_range*0.05) # 5% of mag range, 5% of freq bins, 5% of mag range

        # If no peaks, append nan
        if len(peaks) == 0:
            dominant_freqs.append(np.nan)
        else: 
            sorted_peaks = peaks[np.argsort(spectrum[peaks])][::-1][:n_freqs]
            dominant_freqs.append(freqs[sorted_peaks])

    return np.asarray(dominant_freqs)


def spectral_features(data: ArrayLike, 
                      sr: int,
                      n_freqs: Optional[int] = 1
                      ) -> Dict[str, Union[float, np.floating, NDArray[np.float64]]]:
    """
    Extracts a set of spectral features from a signal.

    Args:
        data (NDArray[np.float64]): Input signal.
        sr (int): Sampling rate of the signal in Hz.

    Returns:
        dict: {
            "fq_q1": float,
            "fq_median": float,
            "fq_q3": float,
            "spectral_flatness": float,
            "centroid": float,
            "spectral_std": float,
            "peak_frequency": float,
            "dominant_frequencies": NDArray[np.float64]
        }
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