import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.stats import kurtosis, skew
from typing import Optional, Tuple
import warnings
from .utils import exclude_trailing_and_leading_zeros, check_signal_format, check_sr_format, cumulative_distribution_function

def amplitude_envelope(data: NDArray[np.float64], kernel_size: Optional[int] = None) -> NDArray[np.float64]:
    """
    Computes the amplitude envelope of a signal using the Hilbert transform.
    
    The function applies the Hilbert transform to calculate the analytic signal of the input data
    and then computes the envelope as the absolute value of the analytic signal. Optionally, 
    a smoothing kernel (Daniell kernel, moving average) can be applied to smooth the envelope.
    
    Args:
        data (NDArray[np.float64]): A 1D NumPy array of float64 values representing the input signal.
        kernel_size (Optional[int]): The size of the moving average kernel to smooth the envelope. Must be an uneven number.
                                      If `None`, no smoothing is applied (default is `None`).
    
    Returns:
        NDArray[np.float64]: A 1D NumPy array of float64 values representing the amplitude envelope of the signal.
    
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
    
    # TODO remove before calculating analytic signal? -> check
    envelope = exclude_trailing_and_leading_zeros(envelope)

    # replace values close to zero to avoid log conflicts in following calculations
    envelope = np.where(envelope < 1e-10, 1e-10, envelope)

    return envelope


def duration(data: NDArray[np.float64], sr: int, exclude_surrounding_silences: Optional[bool] = True) -> float:
    """
    Returns the duration in seconds of a signal.
    
    Args:
        data (NDArray[np.float64]): The audio signal data.
        sr (int): The sample rate of the audio signal (samples per second).
        exclude_surrounding_silences (bool, optional): 
            If True, leading and trailing silences (zeros) in the audio 
            signal will be removed before calculating duration. 
            Defaults to False.

    Returns:
        float: Duration of the (possibly trimmed) audio signal in seconds.
    """
    data = check_signal_format(data)
    check_sr_format(sr)

    if exclude_surrounding_silences:
        data = exclude_trailing_and_leading_zeros(data)

    return len(data) / sr


def temporal_quartiles(data: NDArray[np.float64], sr: int, kernel_size: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Computes the temporal quartiles (Q1, median, Q3) of the amplitude envelope.

    This function takes a signal and its sample rate, then normalizes the
    signal to obtain the amplitude envelope. It calculates the cumulative sum of the normalized
    envelope and uses the `searchsorted` function to find the time points corresponding to the
    25th, 50th, and 75th percentiles of the cumulative amplitude. These correspond to the temporal
    quartiles (Q1, median, Q3) of the signal.

    Args:
        data (NDArray[np.float64]): The signal, stored as a 1D NumPy array of float64 values.
        sr (int): The sample rate of the signal (Hz).
        kernel_size (Optional[int]): Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        Tuple[float, float, float]: A tuple containing the temporal quartiles (Q1, median, Q3),
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
    t_q1 = np.searchsorted(cdf, 0.25) / sr
    t_median = np.searchsorted(cdf, 0.5) / sr
    t_q3 = np.searchsorted(cdf, 0.75) / sr

    return (t_q1, t_median, t_q3)


def temporal_sd(data: NDArray[np.float64], sr: int, kernel_size: Optional[int] = None) -> float:
    """
    Computes the temporal standard deviation of the amplitude envelope.

    Args:
        data (NDArray[np.float64]): The signal, stored as a 1D NumPy array of float64 values.
        sr (int): The sample rate of the signal (Hz).
        kernel_size (Optional[int]): Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float: The temporal standard deviation in seconds
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    return np.std(amplitude_envelope(data, kernel_size))


def temporal_skew(data: NDArray[np.float64], sr: int, kernel_size: Optional[int] = None) -> float:
    """
    Computes the temporal skew of the signal.

    Args:
        data (NDArray[np.float64]): The signal, stored as a 1D NumPy array of float64 values.
        sr (int): The sample rate of the signal (Hz).
        kernel_size (Optional[int]): Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float: The temporal skew
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    return skew(amplitude_envelope(data, kernel_size))


def temporal_kurtosis(data: NDArray[np.float64], sr: int, kernel_size: Optional[int] = None) -> float:
    """
    Computes the temporal kurtosis of the signal.

    Args:
        data (NDArray[np.float64]): The signal, stored as a 1D NumPy array of float64 values.
        sr (int): The sample rate of the signal (Hz).
        kernel_size (Optional[int]): Optional smoothing kernel applied to the amplitude envelope.

    Returns:
        float: The temporal kurtosis
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    
    return kurtosis(amplitude_envelope(data, kernel_size))


def temporal_features(data: NDArray[np.float64], sr: int, kernel_size: Optional[int] = None) -> dict:
    """
    Extracts a set of temporal features from the amplitude envelope of a signal.

    Returns:
        dict: {
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
    t_q1, t_median, t_q3 = temporal_quartiles(data, sr, kernel_size)
    
    features = {
        "t_q1": t_q1,
        "t_median": t_median,
        "t_q3": t_q3,
        "temporal_sd": temporal_sd(data, sr, kernel_size),
        "temporal_skew": temporal_skew(data, sr, kernel_size),
        "temporal_kurtosis": temporal_kurtosis(data, sr, kernel_size),
        "amplitude_envelope": amplitude_envelope(data, kernel_size),
        "duration": duration(data, sr, exclude_surrounding_silences=True)
    }

    return features
