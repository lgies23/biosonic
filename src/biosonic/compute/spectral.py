import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import fft, signal
from scipy.stats import gmean
from typing import Optional, Tuple, Union, Literal
import warnings
from .utils import exclude_trailing_and_leading_zeros, check_signal_format, check_sr_format, cumulative_distribution_function


def spectrum(data: ArrayLike, 
             mode: Union[str, int, float] = 'amplitude') -> NDArray[np.float64]:
    """
    Computes the magnitude spectrum of a signal, allowing for amplitude, power,
    or arbitrary exponentiation of the magnitude.

    Parameters
    ----------
    data : ArrayLike
        The input time-domain signal as a 1D array-like. 
    mode : Union[str, int], default='amplitude'
        Specifies how to compute the spectrum:
        - 'amplitude': return the amplitude spectrum (|FFT|).
        - 'power': return the power spectrum (|FFT|^2).
        - int or float: raise the magnitude to the given power (e.g., 3 for |FFT|^3).

    Returns
    -------
    NDArray[np.float64]
        The transformed frequency-domain representation (magnitude raised to the specified power).

    Raises
    ------
    ValueError
        If `mode` is a string but not one of the supported options.
    TypeError
        If `mode` is not a string, int, or float.
    """
    data = check_signal_format(data)

    if data.size == 0:
        warnings.warn("Input signal is empty; returning an empty spectrum.", RuntimeWarning)
        return np.array([], dtype=np.float64)
    
    magnitude_spectrum = np.abs(fft.fft(data))

    if isinstance(mode, str):
        mode = mode.lower()
        if mode == 'amplitude':
            return magnitude_spectrum
        elif mode == 'power':
            return magnitude_spectrum ** 2
        else:
            raise ValueError(f"Invalid string mode '{mode}'. Use 'amplitude', 'power', or an integer.")
    elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return magnitude_spectrum ** mode
    else:
        raise TypeError(f"'mode' must be a string, int or float, not {type(mode).__name__}.")


def spectrogram(
        data: ArrayLike, 
        sr: int, 
        window: Union[str, NDArray] = "hann", 
        window_length: int = 1024, 
        overlap: float = .5, 
        scaling: str = "magnitude",
        *args, 
        **kwargs
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

def transform_spectrogram_for_nn(
        spectrogram: ArrayLike,
        values_type: str = 'float32', 
        add_channel: bool = True,
        data_format: Literal['channels_last', 'channels_first'] = 'channels_first'
    ) -> ArrayLike:
    """
    Prepares a spectrogram for input into a neural network by normalizing, casting type,
    and optionally adding a channel dimension.

    Parameters:
        spectrogram : ArrayLike 
            The input spectrogram as a 2D array (frequency x time).
        values_type : str
            Data type to cast the spectrogram to (e.g., 'float32', 'float64'). Defaults to 'float32'
        add_channel : bool
            Whether to add a greyscale channel dimension.
        data_format : Literal['channels_last', 'channels_first']
            Specifies channel dimension placement when `add_channel` is True. 
            - 'channels_last' results in shape (H, W, 1) - e.g. for TensorFlow/Keras
            - 'channels_first' results in shape (1, H, W) - e.g. for PyTorch

    Returns:
        ArrayLike : 
            The transformed spectrogram, normalized to [0, 1], cast to the specified
            data type, and optionally with a channel dimension added.
    
    Example:
        >>> import numpy as np
        >>> spec = np.random.rand(128, 128) * 255  # Example spectrogram
        >>> processed = transform_spectrogram_for_nn(spec, values_type='float32',
        ...                                          add_channel=True, data_format='channels_last')
        >>> processed.shape
        (128, 128, 1)
        >>> processed.dtype
        dtype('float32')
    """
    # Todo Error handling, type checks
    if spectrogram.max() - spectrogram.min() == 0:
        warnings.warn(f"Spectrogram contains no information (values in range [{spectrogram.min()}, {spectrogram.max()}]).", RuntimeWarning)
    else:
        # min-max-scale to range [0, 1]
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

    # Convert to desired bit depth
    spectrogram = spectrogram.astype(values_type)
    
    # Add channel dimension (e.g., grayscale)
    if add_channel:
        if data_format == 'channels_last':
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # (H, W, 1)
        else:
            spectrogram = np.expand_dims(spectrogram, axis=0)   # (1, H, W)

    return spectrogram.astype(values_type)

def spectral_quartiles(data: ArrayLike) -> Tuple[float, float, float]:
    # TODO docs, tests
    data = check_signal_format(data)
    if len(data) == 0:
        raise ValueError("Input is empty")
    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")
    
    envelope = spectrum(data, mode="power")
    cdf = cumulative_distribution_function(envelope)

    return (np.searchsorted(cdf, 0.25), np.searchsorted(cdf, 0.5), np.searchsorted(cdf, 0.75))


def flatness(data: ArrayLike) -> float:
    # TODO docs, tests
    # Sueur, J. (2018). Sound Analysis and Synthesis with R (Springer International Publishing) https://doi.org/10.1007/978-3-319-77647-7, p. 299
    data = check_signal_format(data)
    ps_wo_zeros = exclude_trailing_and_leading_zeros(spectrum(data, mode="power"))
    
    return gmean(ps_wo_zeros) / np.mean(ps_wo_zeros)


def mean_frequency(data: ArrayLike, sr: int) -> float:
    # TODO docs, tests
    data = check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    return np.average(freqs, weights=ps)


def variance(data: ArrayLike, sr: int) -> float:
    # TODO docs, tests
    data = check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    mean_frequency = np.average(freqs, weights=ps)

    return np.sum(ps * (freqs - mean_frequency)**2) / np.sum(ps)


def standard_deviation(data: ArrayLike, sr: int) -> float:
    # TODO docs, tests
    
    return np.sqrt(variance(data, sr))


def peak_frequency(data: ArrayLike, sr: int) -> float:
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
        warnings.warn("Input signal is empty; returning NaN for peak frequency.", RuntimeWarning)
        # return empty array if empty signal
        return None

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    return freqs[np.argmax(ps)]


def dominant_frequencies(data: ArrayLike, sr: int, n_freqs: Optional[int] = 1, *args, **kwargs) -> NDArray[np.float64]:    
    # TODO docs, tests, n_freqs, parameters
    spec, _, freqs = spectrogram(data, sr, *args, **kwargs)
    
    dominant_freqs = []

    # Iterate over time frames
    for t in range(spec.shape[1]):
        # Get spectrum for time frame
        spectrum = spec[:, t]
        magnitude_range = np.max(spectrum)-np.min(spectrum)
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
                      n_freqs: Optional[int] = 1) -> dict:
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
            "mean_frequency": float,
            "spectral_variance": float,
            "spectral_std": float,
            "peak_frequency": float,
            "dominant_frequencies": NDArray[np.float64]
        }
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    fq_q1_bin, fq_median_bin, fq_q3_bin = spectral_quartiles(data)
    freqs = np.fft.fftfreq(len(data), d=1 / sr)

    features = {
        "fq_q1": freqs[fq_q1_bin],
        "fq_median": freqs[fq_median_bin],
        "fq_q3": freqs[fq_q3_bin],
        "spectral_flatness": flatness(data),
        "mean_frequency": mean_frequency(data, sr),
        "spectral_variance": variance(data, sr),
        "spectral_std": standard_deviation(data, sr),
        "peak_frequency": peak_frequency(data, sr),
        "dominant_frequencies": dominant_frequencies(data, sr, n_freqs=n_freqs)
    }

    return features