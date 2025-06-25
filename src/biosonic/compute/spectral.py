import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import fft, signal
from scipy.stats import gmean
from typing import Optional, Tuple, Union, Any, Dict, Literal
import warnings
from .utils import  (
    exclude_trailing_and_leading_zeros, 
    check_signal_format,  
    check_sr_format,  
    cumulative_distribution_function,  
    shannon_entropy
)


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


def quartiles(data: ArrayLike, sr: int) -> Tuple[float, float, float]:
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
    # normalize spectrum
    ms =  ms / np.sum(ms) 
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
    sr = check_sr_format(sr)

    if data.size == 0:
        raise ValueError("Input signal is empty; could not determine peak frequency.")

    freqs, ps = spectrum(data, sr=sr, mode="power")
    assert freqs is not None
    return float(freqs[np.argmax(ps)])


def power_spectral_entropy(
        data: ArrayLike, 
        sr: int, 
        unit: Literal["bits", "nat", "dits", "bans", "hartleys"] = "bits",
        *args : Any, 
        **kwargs : Any
        ) -> Tuple[float, float]:
    """
    Calculates the power spectral entropy as follows:
    1. Compute power spectral density (PSD)
    2. Normalize PSD (interpreted as a probability distribution)
    3. Calculate Shannon-Wiener entropy of normalized PSD

    Args:
        data : ArrayLike
            Input signal as a 1D ArrayLike
        sr : int
            Sampling rate in Hz.
        unit : str, optional
            Desired unit of the entropy, determines the logarithmic base used for calculatein. 
            Choose from "bits" (log2), "nat" (ln), or "dits"/"bans"/"hartleys" (log10).
            Defaults to "bits".

    Returns:
        float 
            Power spectral entropy.
    References:
        1. https://de.mathworks.com/help/signal/ref/spectralentropy.html accessed January 13th, 2025. 18:34 pm
        2. https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.entropy.html accessed May 20th 2025, 11:32 am
        3. Shannon C. E. 1948 A mathematical theory of communication. The Bell System Technical Journal XXVII.

    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    # _, psd = signal.welch(data, sr, nperseg=N_FFT, noverlap=N_FFT//HOP_OVERLAP) # would return psd - frequency spectrum squared and scaled by sum - 
    _, psd = spectrum(data, sr, mode="power")
    psd = exclude_trailing_and_leading_zeros(psd)

    psd_sum : float = np.sum(psd)
    psd_norm = psd / psd_sum
    # Ensure no zero values in normalized power distribution because H is undefined with p=0
    psd_norm = psd_norm[psd_norm > 0]
    return shannon_entropy(psd_norm, unit, *args, **kwargs)


def spectral_features(data: ArrayLike, 
                      sr: int,
                      ) -> Dict[str, Union[float, np.floating, NDArray[np.float64]]]:
    """
    Extracts a set of spectral features from a signal.

    Args:
        data : ArrayLike
            The input signal as a 1D ArrayLike.
        sr : int 
            Sampling rate of the signal in Hz.

    Retuns:
        dict
        {"mean_frequency" : float,
        "fq_q1": float,
        "fq_median": float,
        "fq_q3": float,
        "spectral_flatness": float,
        "centroid": float,
        "spectral_std": float,
        "peak_frequency": float}
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    fq_q1_bin, fq_median_bin, fq_q3_bin = quartiles(data, sr)
    freqs, ps = spectrum(data, sr, mode="power")
    
    features = {
        #"mean_frequency" : np.average(freqs, weights=ps), # because centroid is based on magnitude spectrum
        "fq_q1": fq_q1_bin,
        "fq_median": fq_median_bin,
        "fq_q3": fq_q3_bin,
        "spectral_flatness": flatness(data),
        "centroid": centroid(data, sr),
        "spectral_std": bandwidth(data, sr),
        "skew" : skewness(data, sr),
        "kurtosis" : kurtosis(data, sr),
        "peak_frequency": peak_frequency(data, sr),
    }

    return features