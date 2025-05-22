import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Any

from biosonic.compute.utils import hz_to_mel, mel_to_hz


def check_filterbank_parameters(
        n_filters : int, 
        n_fft : int, 
        sr : int, 
        fmin : float, 
        fmax : float,
        ) -> None:
    
    if fmax > sr / 2:
        raise ValueError(f"fmax must be <= Nyquist frequency (sr/2 = {sr/2}), but got fmax={fmax}")

    if fmin < 0 or fmin >= fmax:
        raise ValueError(f"fmin must be >= 0 and < fmax, but got fmin={fmin}, fmax={fmax}")

    # Ensure n_fft provides enough resolution to resolve unique bins for each filter
    min_bins_required = n_filters + 2
    max_bin_index = int(np.floor((n_fft + 1) * fmax / sr))
    if max_bin_index < min_bins_required:
        raise ValueError(
            f"n_fft={n_fft} doesn't provide enough resolution for {n_filters} filters up to fmax={fmax}Hz "
            f"(only {max_bin_index} frequency bins available). Increase n_fft or reduce fmax/n_filters."
        )
    
def filterbank(
        n_filters : int, 
        n_fft : int, 
        bin_freqs : ArrayLike
    ) -> ArrayLike:

    filterbank = np.zeros([n_filters, n_fft // 2 + 1])
    for i in range(1, n_filters + 1):
        left = bin_freqs[i - 1]
        center = bin_freqs[i]
        right = bin_freqs[i + 1]

        # Triangular filter
        filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    
    return filterbank


def mel_filterbank(
        n_filters : int, 
        n_fft : int, 
        sr : int, 
        fmin : float = 0.0, 
        fmax : Optional[float] = None,
        corner_frequency : Optional[float] = None,
        **kwargs : Any
    ) -> ArrayLike:

    if fmax is None:
        fmax = sr / 2

    check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    # boundaries
    mel_min = hz_to_mel(fmin, corner_frequency=corner_frequency, **kwargs)
    mel_max = hz_to_mel(fmax, corner_frequency=corner_frequency, **kwargs)

    # evenly spaced in Mel scale, then convert to Hz
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points, corner_frequency=corner_frequency, **kwargs)

    # FFT bin numbers
    bin_freqs = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    return filterbank(n_filters, n_fft, bin_freqs)


def linear_filterbank(
        n_filters: int, 
        n_fft: int, 
        sr: int, 
        fmin : float = 0.0, 
        fmax : Optional[float] = None
    ) -> ArrayLike:
    if fmax is None:
        fmax = sr / 2

    check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    hz_points = np.linspace(fmin, fmax, n_filters + 2)
    bin_freqs = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    return filterbank(n_filters, n_fft, bin_freqs)


def log_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    base: float = 2.0,
) -> ArrayLike:
    """
    Create a logarithmically spaced triangular filterbank.

    Parameters
    ----------
    n_filters : int
        Number of triangular filters.
    n_fft : int
        FFT size (defines frequency resolution).
    sr : int
        Sampling rate of the signal in Hz.
    fmin : float
        Minimum frequency in Hz. Must be > 0.
    fmax : float, optional
        Maximum frequency in Hz. Defaults to Nyquist (sr/2).
    base : float
        Base of the logarithmic spacing. Typically 2 (for octaves).

    Returns
    -------
    filterbank : np.ndarray
        Array of shape (n_filters, n_fft//2 + 1), each row a filter.
    """
    if fmax is None:
        fmax = sr / 2

    check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    # compute log-spaced center frequencies
    log_min = np.log(fmin+1e-30) / np.log(base)
    log_max = np.log(fmax) / np.log(base)
    log_centers = np.linspace(log_min, log_max, n_filters + 2)
    hz_points = base ** log_centers

    # convert to FFT bin indices
    bin_freqs = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    return filterbank(n_filters, n_fft, bin_freqs)
