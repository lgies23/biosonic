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
    check_signal_format(data)

    if data.size == 0:
        warnings.warn("Input signal is empty; returning an empty spectrum.", RuntimeWarning)
        # return empty array if empty signal
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


def spectral_quartiles(data: NDArray[np.float64]) -> Tuple[float, float, float]:
    # TODO docs, tests
    check_signal_format(data)
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
    check_signal_format(data)
    ps_wo_zeros = exclude_trailing_and_leading_zeros(spectrum(data, mode="power"))
    
    return gmean(ps_wo_zeros) / np.mean(ps_wo_zeros)


def mean_frequency(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    return np.average(freqs, weights=ps)

def variance(data: NDArray[np.float64], sr: int) -> float:
    check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    mean_frequency = np.average(freqs, weights=ps)

    return np.sum(ps * (freqs - mean_frequency)**2) / np.sum(ps)

def standard_deviation(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    
    return np.sqrt(variance(data, sr))

def peak_frequency(data: NDArray[np.float64], sr: int) -> float:
    # TODO docs, tests
    check_signal_format(data)
    check_sr_format(sr)

    ps = spectrum(data, mode="power")
    freqs = np.fft.fftfreq(len(data), d=1/sr)

    return freqs[np.argmax(ps)] / 1000


def spectrogram(data: NDArray[np.float64], sr: int, *args, **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    check_signal_format(data)
    check_sr_format(sr)

    return signal.spectrogram(data, sr, *args, **kwargs)

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
            dominant_freqs.append(sorted_peaks)

    return np.asarray(dominant_freqs)