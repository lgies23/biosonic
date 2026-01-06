import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.stats import kurtosis, skew

from .utils import (
    check_signal_format,
    check_sr_format,
    cumulative_distribution_function,
    rms,
    shannon_entropy
)


def amplitude_envelope(
        data: ArrayLike,
        n_out: Optional[int] = None,
        avoid_zero_values: bool = False,
        silence_threshold: Optional[float] = None,
        return_indices: bool = False,
        sr: Optional[int] = None,
        **rms_kwargs: Any
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], Tuple[int, int]]]:
    """
    Computes the amplitude envelope of a signal using scipy.signal.envelope.
    Optionally, silences can be removed based on percentile thresholds.

    Args:
        data : ArrayLike
            A 1D array-like representing the input signal.
        n_out : Optional[int]
            The number of points in the output envelope. If `None`, the output length is the same as the input length.
        avoid_zero_values : Optional[bool]
            If `True`, replaces zero values with 1e-10 after applying Hilbert transform and removing leading and trailing zeros.
            Defaults to `False`
        silence_threshold : Optional[float]
            If None, leading and trailing silences (zeros) in the audio signal will be removed before calculating duration.
            If a float between 0 and 50, trims the given percentile from both ends based on amplitude envelope (e.g., 1.0 trims 1 percentile from each end).
            Defaults to None.
        return_indices : Optional[bool]
            If `True`, returns a tuple containing the amplitude envelope and a tuple of start and end sample indices used for trimming silences.
            Defaults to `False`.
        **rms_kwargs :
            Additional keyword arguments passed to the `rms` function for silence trimming.

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
        - This function relies on `scipy.signal.envelope` to compute the amplitude envelope.
        - The optional `silence_threshold` parameter allows for trimming of leading and trailing silences based on a
        relative RMS threshold as in librosa and soundgen (see references).

    References:
        1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.envelope.html accessed January 5th, 2026. 2:49 pm

    """
    if len(data) == 0:
        warnings.warn("Input signal is empty; returning an empty array.", RuntimeWarning)
        if return_indices:
            return np.array([], dtype=np.float64), (0, 0)
        return np.array([], dtype=np.float64)

    if np.all(data == 0):
        warnings.warn("Input signal contains only zeros; returning empty array.", RuntimeWarning)
        if return_indices:
            return np.array([], dtype=np.float64), (0, 0)
        return np.array([], dtype=np.float64)

    start_sample, end_sample = 0, len(data)

    # Relative RMS-based silence trimming as in librosa and soundgen
    if silence_threshold is not None and 0 < silence_threshold < 1:
        assert sr is not None, "Sample rate must be provided when using silence_threshold."
        rms_array, times_s = rms(data, sr=sr, **rms_kwargs)
        rms_norm = rms_array / np.max(rms_array)
        above = np.where(rms_norm >= silence_threshold)[0]

        if len(above) == 0:
            warnings.warn(
                "No signal above silence threshold; returning empty array.",
                RuntimeWarning
            )
            data = np.array([], dtype=np.float64)
            start_sample, end_sample = 0, 0
        else:
            start_sample = int(times_s[above[0]] * sr)
            end_sample = int(times_s[above[-1]] * sr)
            data = data[start_sample:end_sample]

    data = check_signal_format(data)
    envelope = signal.envelope(data, n_out=n_out)[0]

    if avoid_zero_values:
        envelope = np.where(envelope < 1e-10, 1e-10, envelope)

    if return_indices:
        return envelope, (start_sample, end_sample)
    return envelope


def duration(data: ArrayLike, sr: int, **envelope_kwargs: Any) -> float:
    """
    Returns the duration in seconds of a signal, optionally trimming leading and trailing silences based on percentiles.

    Args:
        data : ArrayLike
            The audio signal data.
        sr : int
            The sample rate of the audio signal (samples per second).
        percentile_silence : Optional[float]
            If None, leading and trailing silences (zeros) in the audio signal will be removed before calculating duration.
            If a float between 0 and 50, trims the given percentile from both ends based on amplitude envelope (e.g., 1.0 trims 1 percentile from each end).
            Defaults to None.
        **rms_kwargs :
            Additional keyword arguments passed to the `rms` function for silence trimming.

    Returns:
        float
            Duration of the (possibly trimmed) audio signal in seconds.
    """
    data = check_signal_format(data)
    check_sr_format(sr)
    envelope = amplitude_envelope(data, sr=sr, **envelope_kwargs)

    return len(envelope) / sr


def temporal_quartiles(
        data: ArrayLike,
        sr: int,
        **envelope_kwargs: Any
        ) -> Tuple[float, float, float]:
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
    sr = check_sr_format(sr)

    if len(data) == 0:
        raise ValueError("Input is empty")
    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")

    envelope = amplitude_envelope(data, sr=sr, **envelope_kwargs)

    cdf = cumulative_distribution_function(envelope)

    # temporal quartiles (Q1, median, Q3)
    t_q1 = float(np.searchsorted(cdf, 0.25) / sr)
    t_median = float(np.searchsorted(cdf, 0.5) / sr)
    t_q3 = float(np.searchsorted(cdf, 0.75) / sr)

    return (t_q1, t_median, t_q3)


def temporal_sd(data: ArrayLike, sr: int, **envelope_kwargs: Any) -> float:
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
    sr = check_sr_format(sr)

    return float(np.std(amplitude_envelope(data, sr=sr, **envelope_kwargs)))


def skewness(
        data: ArrayLike,
        sr: Optional[int] = None,
        **envelope_kwargs: Any
    ) -> Optional[float]:
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

    skew_ = skew(amplitude_envelope(data, sr=sr, **envelope_kwargs))
    if skew_ is None:
        warnings.warn("All values are equal, returning None for skew.")

    return float(skew_)


def temporal_kurtosis(
        data: ArrayLike,
        sr: Optional[int] = None,
        **envelope_kwargs: Any
        ) -> Optional[float]:
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

    kurtosis_ = kurtosis(amplitude_envelope(data, sr=sr, **envelope_kwargs))

    if kurtosis_ is None:
        warnings.warn("All values are equal, returning None for kurtosis.")

    return float(kurtosis_)


def temporal_entropy(
        data: ArrayLike,
        unit: Literal["bits", "nat", "dits", "bans", "hartleys"] = "bits",
        norm: bool = False,
        sr: Optional[int] = None,
        **envelope_kwargs: Any
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
        norm : bool, optional
            If True, normalizes the entropy by dividing it by the log of the number of bins.
            Defaults to False.
        **envelope_kwargs :
            Additional keyword arguments passed to the `amplitude_envelope` function.

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
    envelope = amplitude_envelope(data, sr=sr, **envelope_kwargs)
    hist, _ = np.histogram(envelope, bins="auto", density=False)
    hist = hist[hist > 0]
    norm_counts = hist / np.sum(hist)

    return shannon_entropy(norm_counts, unit, norm=norm)


def temporal_features(
    data: NDArray[np.float32],
    sr: int,
    return_trim_indices: bool = False,
    **envelope_kwargs: Any
    ) -> Dict[str, Union[float, NDArray[np.float32], Tuple[int, int]]]:
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
        "duration": float,
        "temporal_entropy": float
        }
    """
    # Get envelope and optionally trim indices
    if envelope_kwargs.get("silence_threshold", None) is not None or return_trim_indices:
        envelope, (start_sample, end_sample) = amplitude_envelope(data, sr=sr, return_indices=True, **envelope_kwargs)
    else:
        envelope = amplitude_envelope(data, sr=sr, **envelope_kwargs)
        start_sample, end_sample = 0, len(data)

    # times for the trimmed envelope in the context of the original signal
    times = np.linspace(start_sample / sr, end_sample / sr, len(envelope), endpoint=False)
    t_q1, t_median, t_q3 = temporal_quartiles(data[start_sample:end_sample], sr, **envelope_kwargs)

    features = {
        "t_q1": t_q1,
        "t_median": t_median,
        "t_q3": t_q3,
        "temporal_centroid": np.average(times, weights=envelope/np.mean(envelope)) if len(envelope) > 0 else np.nan,
        "temporal_sd": temporal_sd(data, sr, **envelope_kwargs),
        "temporal_skew": skewness(data, sr, **envelope_kwargs),
        "temporal_kurtosis": temporal_kurtosis(data, sr, **envelope_kwargs),
        "amplitude_envelope": envelope,
        "duration": duration(data, sr, **envelope_kwargs),
        "temporal_entropy": temporal_entropy(data, sr=sr, **envelope_kwargs)[0],
        "trim_indices": (start_sample, end_sample),
        "trim_times": (start_sample / sr, end_sample / sr)
    }

    if not return_trim_indices:
        features.pop("trim_indices")
        features.pop("trim_times")

    return features
