import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Optional, Tuple
from utils import exclude_trailing_and_leading_zeros


def amplitude_envelope(data: NDArray[np.float64], kernel_size: Optional[int] = 101) -> NDArray[np.float64]:
    """
    Computes the amplitude envelope of a 1D signal using the Hilbert transform.
    
    The function applies the Hilbert transform to calculate the analytic signal of the input data
    and then computes the envelope as the absolute value of the analytic signal. Optionally, 
    a smoothing kernel (Daniell kernel, moving average) can be applied to smooth the envelope.
    
    Args:
        data (NDArray[np.float64]): A 1D NumPy array of float64 values representing the input signal.
        kernel_size (Optional[int]): The size of the moving average kernel to smooth the envelope.
                                      If `None`, no smoothing is applied (default is `101`).
    
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
    
    # Compute the analytic signal using Hilbert transform
    analytic_signal = signal.hilbert(data)
    envelope = np.abs(analytic_signal)

    if kernel_size is not None:
        # Apply Daniell kernel (moving average kernel) only if kernel_size is provided
        kernel = np.ones(kernel_size) / kernel_size

        # Apply kernel to smooth envelope
        envelope = signal.convolve(envelope, kernel, mode='same')
        envelope = exclude_trailing_and_leading_zeros(envelope)
    
    return envelope

def duration_s(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, checks
    return len(data) / sr

def temporal_quartiles(data: NDArray[np.float64], sr: int) -> Tuple[float, float, float]:
    """
    Computes the temporal quartiles (Q1, median, Q3) of the amplitude envelope.

    This function takes a 1D array representing a signal and its sample rate, then normalizes the
    signal to obtain the amplitude envelope. It calculates the cumulative sum of the normalized
    envelope and uses the `searchsorted` function to find the time points corresponding to the
    25th, 50th, and 75th percentiles of the cumulative amplitude. These correspond to the temporal
    quartiles (Q1, median, Q3) of the signal.

    Args:
        data (NDArray[np.float64]): A 1D NumPy array of float64 values representing the signal.
        sr (int): The sample rate of the signal (Hz).

    Returns:
        Tuple[float, float, float]: A tuple containing the temporal quartiles (Q1, median, Q3),
                                     in seconds.

    Example:
        >>> signal = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
        >>> sr = 100  # Sample rate of 100 Hz
        >>> temporal_quartiles(signal, sr)
        (0.03, 0.05, 0.07)

    Notes:
        - The `searchsorted` function is used to find the indices corresponding to the quartiles
          in the cumulative amplitude envelope. The result is scaled by the sample rate to return
          time values in seconds.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if data.ndim != 1:
        raise ValueError("data must be a 1D array.")
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError("data must be of type np.float64.")
    
    if not isinstance(sr, int):
        raise TypeError("sr must be of type integer.")
    if sr <= 0:
        raise ValueError("sr must be positive.")
    
    # Normalize envelope
    normalized_env = data / np.sum(data)
    
    # Cumulative sum of normalized envelope
    cumulative_amplitude_envelope = np.cumsum(normalized_env)

    # Temporal quartiles (Q1, median, Q3)
    t_q1 = np.searchsorted(cumulative_amplitude_envelope, 0.25) / sr
    t_median = np.searchsorted(cumulative_amplitude_envelope, 0.5) / sr
    t_q3 = np.searchsorted(cumulative_amplitude_envelope, 0.75) / sr

    return (t_q1, t_median, t_q3)