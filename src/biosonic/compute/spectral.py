import numpy as np
from numpy.typing import NDArray
from scipy import fft, signal
from scipy.stats import gmean
from typing import Optional, Tuple
from typing import Union
import warnings
from .utils import exclude_trailing_and_leading_zeros, check_signal_format, check_sr_format, cumulative_distribution_function


def spectrum(data: NDArray[np.float64], 
             mode: Union[str, int, float] = 'amplitude') -> NDArray[np.float64]:
    """
    Computes the magnitude spectrum of a signal, allowing for amplitude, power,
    or arbitrary exponentiation of the magnitude.

    Parameters
    ----------
    data : NDArray[np.float64]
        The input time-domain signal as a 1D NumPy array of floats.
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


def spectrogram(data: NDArray[np.float64], sr: int, *args, **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    data = check_signal_format(data)
    check_sr_format(sr)

    return signal.spectrogram(data, sr, *args, **kwargs)


def spectral_quartiles(data: NDArray[np.float64]) -> Tuple[float, float, float]:
    # TODO docs, tests
    data = check_signal_format(data)
    if len(data) == 0:
        raise ValueError("Input is empty")
    if np.all(data == 0):
        raise ValueError("Signal contains no nonzero values")
    
    envelope = spectrum(data, mode="power")
    cdf = cumulative_distribution_function(envelope)

    return (np.searchsorted(cdf, 0.25), np.searchsorted(cdf, 0.5), np.searchsorted(cdf, 0.75))

def flatness(data: NDArray[np.float64]) -> float:
    # TODO docs, tests
    # Sueur, J. (2018). Sound Analysis and Synthesis with R (Springer International Publishing) https://doi.org/10.1007/978-3-319-77647-7, p. 299
    data = check_signal_format(data)
    ps_wo_zeros = exclude_trailing_and_leading_zeros(spectrum(data, mode="power"))
    
    return gmean(ps_wo_zeros) / np.mean(ps_wo_zeros)


def mean_frequency(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    data = check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    return np.average(freqs, weights=ps)

def variance(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    data = check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    mean_frequency = np.average(freqs, weights=ps)

    return np.sum(ps * (freqs - mean_frequency)**2) / np.sum(ps)


def standard_deviation(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    
    return np.sqrt(variance(data, sr))


def peak_frequency(data: NDArray[np.float64], sr: int) -> float:
    """
    Computes the peak frequency of a signal using the Fourier transform.

    The function applies the Fast Fourier Transform (FFT) to the input signal to obtain its frequency spectrum.
    It identifies the frequency corresponding to the maximum magnitude in the spectrum, which represents the dominant
    or peak frequency of the signal.

    Args:
        data (NDArray[np.float64]): 1D NumPy array representing the input signal.
        sampling_rate (float): Sampling rate of the signal in Hz.

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


def dominant_frequencies(data: NDArray[np.float64], sr: int, n_freqs: Optional[int] = 1, *args, **kwargs) -> NDArray[np.float64]:    
    # TODO docs, tests, n_freqs, parameters
    freqs, _, spec = spectrogram(data, sr, *args, **kwargs)
    
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

def spectral_features(data: NDArray[np.float64], 
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