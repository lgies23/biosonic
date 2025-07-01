import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Optional, Tuple, Union, Dict, Literal, Any
import warnings
from .utils import (
    exclude_trailing_and_leading_zeros, 
    check_signal_format, 
    check_sr_format, 
    cumulative_distribution_function,
    shannon_entropy
)


def amplitude_envelope(
        data: ArrayLike, 
        kernel_size: Optional[int] = None,
        remove_trailing_zeros : bool = True,
        avoid_zero_values : bool = False
    ) -> NDArray[np.float64]:
    """
    Computes the amplitude envelope of a signal using the Hilbert transform.
    
    The function applies the Hilbert transform to calculate the analytic signal of the input data
    and then computes the envelope as the absolute value of the analytic signal. Optionally, 
    a smoothing kernel (Daniell kernel, moving average) can be applied to smooth the envelope.
    
    Args:
        data : ArrayLike
            A 1D array-like representing the input signal.
        kernel_size : Optional[int]
            The size of the moving average kernel to smooth the envelope. Must be an uneven number.
            If `None`, no smoothing is applied (default is `None`).
        remove_trailing_zeros : Optional[bool]
            If `True`, removes leading and trailing zeros of the envelope after applying Hilbert transform. 
            Defaults to `True`
        avoid_zero_values : Optional[bool]
            If `True`, replaces zero values with 1e-10 after applying Hilbert transform and removing leading and trailing zeros. 
            Defaults to `False`
    
    Returns:
        NDArray[np.float64]
            A 1D NumPy array of float64 values representing the amplitude envelope of the signal.
    
    Example:
        >>> import numpy as np
        >>> signal = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
        >>> envelope = amplitude_envelope(signal, kernel_size=3)
        >>> print(envelope)
        [0.  1.  2.  3.  2.  1.  0.]

    Notes:
        - The Hilbert transform is used to generate the analytic signal.
        - If `kernel_size` is provided, a moving average filter (Daniell kernel) is applied to smooth the envelope.
        - The envelope is processed to remove any leading or trailing zeros using the `exclude_trailing_and_leading_zeros` function.
    """
    if len(data) == 0:
        warnings.warn("Input signal is empty; returning an empty array.", RuntimeWarning)
        return np.array([], dtype=np.float64)
    
    data = check_signal_format(data)
    
    # compute the analytic signal using Hilbert transform
    analytic_signal = signal.hilbert(data)
    envelope = np.abs(analytic_signal)

    if kernel_size is not None:
        # apply Daniell kernel (moving average kernel) only if kernel_size is provided
        kernel = np.ones(kernel_size) / kernel_size

        # apply kernel to smooth envelope
        envelope = signal.convolve(envelope, kernel, mode='same')
    
    if remove_trailing_zeros:
        envelope = exclude_trailing_and_leading_zeros(envelope)

    if avoid_zero_values:
        # replace values close to zero to avoid log conflicts in following calculations
        envelope = np.where(envelope < 1e-10, 1e-10, envelope)

    return envelope


def duration(data: ArrayLike, sr: int, exclude_surrounding_silences: Optional[bool] = True) -> float:
    """
    Returns the duration in seconds of a signal.
    
    Args:
        data : ArrayLike
            The audio signal data.
        sr : int
            The sample rate of the audio signal (samples per second).
        exclude_surrounding_silences : bool, optional 
            If True, leading and trailing silences (zeros) in the audio 
            signal will be removed before calculating duration. 
            Defaults to False.

    Returns:
        float
            Duration of the (possibly trimmed) audio signal in seconds.
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    if exclude_surrounding_silences:
        data = exclude_trailing_and_leading_zeros(data)

    return len(data) / sr


def temporal_quartiles(data: ArrayLike, sr: int, kernel_size: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Computes the temporal quartiles (Q1, median, Q3) of the amplitude envelope.

    This function takes a signal and its sample rate, then normalizes the
    signal to obtain the amplitude envelope. It calculates the cumulative sum of the normalized
    envelope and uses the `searchsorted` function to find the time points corresponding to the
    25th, 50th, and 75th percentiles of the cumulative amplitude. These correspond to the temporal
    quartiles (Q1, median, Q3) of the signal.

    Args:
        data : ArrayLike
            The signal, stored as a 1D NumPy array of float64 values.
        sr : int
            The sample rate of the signal (Hz).
        kernel_size : Optional[int]
            Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        Tuple[float, float, float]
            A tuple containing the temporal quartiles (Q1, median, Q3),
            in seconds.

    Example:
        >>> signal = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
        >>> sr = 44100  # Sample rate of 44.1 kHz
        >>> temporal_quartiles(signal, sr)
        (0.03, 0.05, 0.07)

    Notes:
        - The `searchsorted` function is used to find the indices corresponding to the quartiles
          in the cumulative amplitude envelope. The result is scaled by the sample rate to return
          time values in seconds.
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    if len(data) == 0:
        raise ValueError("Input is empty")
    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")
    
    envelope = amplitude_envelope(data, kernel_size)
    
    cdf = cumulative_distribution_function(envelope)

    # temporal quartiles (Q1, median, Q3)
    t_q1 = float(np.searchsorted(cdf, 0.25) / sr)
    t_median = float(np.searchsorted(cdf, 0.5) / sr)
    t_q3 = float(np.searchsorted(cdf, 0.75) / sr)

    return (t_q1, t_median, t_q3)


def temporal_sd(data: ArrayLike, sr: int, kernel_size: Optional[int] = None) -> float:
    """
    Computes the temporal standard deviation of the amplitude envelope.

    Args:
        data : ArrayLike
            The signal, stored as a 1D NumPy array of float64 values.
        sr : int
            The sample rate of the signal (Hz).
        kernel_size : Optional[int] 
            Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float
            The standard deviation of the amplitude envelope
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    return float(np.std(amplitude_envelope(data, kernel_size)))


def skewness(data: ArrayLike, sr: int, kernel_size: Optional[int] = None) -> Optional[float]:
    """
    Computes the temporal skew of the signal.

    Args:
        data : ArrayLike 
            The signal, stored as a 1D NumPy array of float64 values.
        sr : int
            The sample rate of the signal (Hz).
        kernel_size : Optional[int]
            Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float
            The temporal skew
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    skew_ = skew(amplitude_envelope(data, kernel_size))
    if skew_ == None:
        warnings.warn("All values are equal, returning None for skew.")

    return float(skew_)


def temporal_kurtosis(data: ArrayLike, sr: int, kernel_size: Optional[int] = None) -> Optional[float]:
    """
    Computes the temporal kurtosis of the signal.

    Args:
        data : ArrayLike
            The signal, stored as a 1D NumPy array of float64 values.
        sr : int 
            The sample rate of the signal (Hz).
        kernel_size : Optional[int]
            Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float
            The temporal kurtosis
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)
    
    kurtosis_ = kurtosis(amplitude_envelope(data, kernel_size))

    if kurtosis_ == None:
        warnings.warn("All values are equal, returning None for kurtosis.")

    return float(kurtosis_)


def temporal_entropy(
        data: ArrayLike,  
        unit: Literal["bits", "nat", "dits", "bans", "hartleys"] = "bits",
        *args : Any, 
        **kwargs : Any
    ) -> Tuple[float, float]:
    """
    Calculates the entropy of the amplitude envelope as follows:
    1. Compute the amplitude envelope.
    2. Calculate a discrete probability distribution from the histogram of the envelope.
    3. Calculate Shannon-Wiener entropy of the probability distribution.

    Args:
        data : ArrayLike
            Input signal as a 1D ArrayLike.
        unit : str, optional
            Desired unit of the entropy, determines the logarithmic base used for calculatein. 
            Choose from "bits" (log2), "nat" (ln), or "dits"/"bans"/"hartleys" (log10).
            Defaults to "bits".

    Returns:
        float 
            Temporal entropy.
    References:
        1. https://de.mathworks.com/help/signal/ref/spectralentropy.html accessed January 13th, 2025. 18:34 pm
        2. https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.entropy.html accessed May 20th 2025, 11:32 am
        3. Shannon C. E. 1948 A mathematical theory of communication. The Bell System Technical Journal XXVII.

    Notes:
        Rounds
    """
    envelope = amplitude_envelope(data)
    hist, _ = np.histogram(envelope, bins="auto", density=False)
    hist = hist[hist > 0]
    norm_counts = hist / np.sum(hist)

    return shannon_entropy(norm_counts, unit, *args, **kwargs)


def temporal_features(
        data: NDArray[np.float64], 
        sr: int, 
        kernel_size: Optional[int] = None,
        ) -> Dict[str, Union[float, NDArray[np.float64]]]:
    """
    Extracts a set of temporal features from the amplitude envelope of a signal.

    Returns
    -------
        dict
        {
        "t_q1": float,
        "t_median": float,
        "t_q3": float,
        "temporal_sd": float,
        "temporal_skew": float,
        "temporal_kurtosis": float,
        "amplitude_envelope": NDArray[np.float64],
        "duration": float
        }
    """
    envelope = amplitude_envelope(data)
    times = np.linspace(0, len(data)/sr, len(data))
    t_q1, t_median, t_q3 = temporal_quartiles(data, sr, kernel_size)
    
    features = {
        "t_q1": t_q1,
        "t_median": t_median,
        "t_q3": t_q3,
        "temporal_centroid": np.average(times, weights=envelope/np.mean(envelope)),
        "temporal_sd": temporal_sd(data, sr, kernel_size),
        "temporal_skew": skewness(data, sr, kernel_size),
        "temporal_kurtosis": temporal_kurtosis(data, sr, kernel_size),
        "amplitude_envelope": envelope,
        "duration": duration(data, sr, exclude_surrounding_silences=True)
    }

    return features
