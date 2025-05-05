import numpy as np
from numpy.typing import NDArray
from scipy import fft
from typing import Optional, Tuple
from typing import Union
import warnings
from .utils import exclude_trailing_and_leading_zeros, check_signal_format, check_sr_format


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