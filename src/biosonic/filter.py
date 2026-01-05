from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt

from biosonic.compute.utils import hz_to_mel, mel_to_hz


def _check_filterbank_parameters(
        n_filters: int,
        n_fft: int,
        sr: int,
        fmin: float,
        fmax: float,
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


def _filterbank(
        n_filters: int,
        n_fft: int,
        center_freqs: ArrayLike,
        fft_freqs: ArrayLike
    ) -> ArrayLike:
    """
    Construct a triangular filter bank for spectral analysis.

    This function creates a bank of `n_filters` triangular filters to be applied
    to the magnitude spectrum of an audio signal, commonly used in feature
    extraction methods such as Mel-frequency cepstral coefficients (MFCCs).
    Each filter is shaped like a triangle and spans a range of FFT bins
    determined by the `bin_indices` array.

    Parameters
    ----------
    n_filters : int
        Number of filters in the filter bank.

    n_fft : int
        FFT size used to compute the frequency bins. Determines the number of
        frequency bins available.

    bin_indices : ArrayLike
        Array of bin indices (length = n_filters + 2) that specify the start,
        center, and end points of each triangular filter in the filter bank.

    Returns
    -------
    filterbank : ArrayLike
        A 2D array of shape `(n_filters, n_fft // 2 + 1)` containing the filter
        bank matrix. Each row represents a single triangular filter applied
        across the FFT bins.

    Notes
    -----
    The filters are normalized according to the slaney method (directly taken from `librosa`, see references) to
    ensure energy preservation.

    References
    ----------
    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg,
    and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings
    of the 14th python in science conference, pp. 18-25. 2015.
    """
    filterbank = np.zeros((n_filters, int(n_fft // 2 + 1)), dtype=np.float32)

    fdiff = np.diff(center_freqs)
    ramps = np.subtract.outer(center_freqs, fft_freqs)

    for i in range(n_filters):
        # lower and upper slopes
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # intersect with each other and zero
        filterbank[i] = np.maximum(0, np.minimum(lower, upper))

    # normalize each filter to have approx constant energy
    enorm = 2.0 / (center_freqs[2:n_filters+2] - center_freqs[:n_filters])
    filterbank *= enorm[:, np.newaxis]

    # check if any filter is empty
    if np.any(filterbank.max(axis=1) == 0):
        print(
            "Empty filters detected in mel filterbank. "
            "Consider reducing number of bands or increasing n_fft."
        )

    return filterbank


def mel_filterbank(
        n_filters: int,
        n_fft: int,
        sr: int,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        **kwargs: Any
    ) -> ArrayLike:
    """
    Create a mel spaced triangular filterbank.

    Parameters
    ----------
    n_filters : int
        Number of triangular filters.
    window_length : int
        Window size in samples.
    sr : int
        Sampling rate of the signal in Hz.
    fmin : float
        Minimum frequency in Hz. Must be > 0.
    fmax : float, optional
        Maximum frequency in Hz. Defaults to Nyquist (sr/2).

    Returns
    -------
    filterbank : np.ndarray
        Array of shape (n_filters, n_fft//2 + 1), each row a filter.
    """
    if fmax is None:
        fmax = float(sr) / 2

    _check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    # fft bin center frequencies
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # boundaries
    mel_min = hz_to_mel(fmin, **kwargs)
    mel_max = hz_to_mel(fmax, **kwargs)

    # mel filter center frequencies
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points, **kwargs)
    return _filterbank(n_filters, n_fft, hz_points, fft_freqs), mel_points


def linear_filterbank(
        n_filters: int,
        n_fft: int,
        sr: int,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ) -> ArrayLike:
    """
    Create a linearly spaced triangular filterbank.

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

    Returns
    -------
    filterbank : np.ndarray
        Array of shape (n_filters, n_fft//2 + 1), each row a filter.
    """
    if fmax is None:
        fmax = sr / 2

    _check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    # fft bin center frequencies
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    hz_points = np.linspace(fmin, fmax, n_filters + 2)

    return _filterbank(n_filters, n_fft, hz_points, fft_freqs), hz_points


def log_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 0.0,
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

    _check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    if fmin <= 0:
        raise ValueError("fmin must be greater than 0 for log-scaled filterbanks.")

    # fft bin center frequencies
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # compute log-spaced center frequencies
    log_min = np.log(fmin) / np.log(base)
    log_max = np.log(fmax) / np.log(base)
    log_points = np.linspace(log_min, log_max, n_filters+2, dtype=np.float32)
    hz_points = base ** log_points

    # convert to FFT bin indices
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    return _filterbank(n_filters, n_fft, hz_points, fft_freqs), log_points


# TODO weighted filter (seewave), rolloff like in audacity f-filter (tuneR)
def filter(
        data: ArrayLike,
        sr: int,
        f_cutoff: Union[int, Tuple[int, int]],
        type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        order: int = 2,
) -> ArrayLike:
    """
    Apply a zero-phase Butterworth filter to a 1D signal using SciPy.

    This function is a wrapper around `scipy.signal.butter` and `scipy.signal.filtfilt`.
    It designs a digital Butterworth filter of the specified type and order,
    then applies it using forward-backward filtering for zero phase distortion.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal to filter.
    sr : int
        Sampling rate of the signal in Hz.
    type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        Type of filter to apply. Defaults to 'lowpass'.
    f_cutoff : float or tuple of float
        Cutoff frequency/frequencies in Hz:
        - Single int for 'lowpass' or 'highpass'
        - Tuple of two floats for 'bandpass' or 'bandstop'
        Values must be within (0, Nyquist), where Nyquist = sr / 2.
    order : int
        Filter order. Higher values result in a steeper frequency cutoff,
        but can introduce more edge artifacts and potential instability.
        Defaults to 2, resulting in a slope of 40 dB per decade (i.e. ten-fold change in frequency).

    Returns
    -------
    filtered_signal : np.ndarray
        The filtered signal, same shape as the input.

    Notes
    -----
    - This uses `scipy.signal.butter` and `scipy.signal.filtfilt` to apply the filter forward and backward,
      ensuring zero-phase distortion.

    References
    ----------
    Virtanen P et al. 2020 SciPy 1.0: fundamental algorithms for scientific computing in Python.
    Nat Methods 17, 261–272. (doi:10.1038/s41592-019-0686-2)
    """

    if type in ['bandpass', 'bandstop']:
        if not isinstance(f_cutoff, (list, tuple)) or len(f_cutoff) != 2:
            raise ValueError("f_cutoff must be a tuple/list of two values for bandpass/bandstop filters.")
    else:
        if isinstance(f_cutoff, (list, tuple)):
            raise ValueError("f_cutoff must be a scalar for lowpass/highpass filters.")

    b, a = butter(order, f_cutoff, btype=type, analog=False, fs=sr)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal
