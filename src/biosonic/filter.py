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
    hz_points = np.asarray(mel_to_hz(mel_points, corner_frequency=corner_frequency, **kwargs))

    # FFT bin numbers
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    f_centers = np.sqrt(hz_points[:-2] * hz_points[2:])

    return filterbank(n_filters, n_fft, bin_indices), f_centers


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
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # geometric mean of adjacent edges skipping first and last to get centers
    f_centers = np.sqrt(hz_points[:-2] * hz_points[2:])

    return filterbank(n_filters, n_fft, bin_indices), f_centers


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

    check_filterbank_parameters(n_filters, n_fft, sr, fmin, fmax)

    # compute log-spaced center frequencies
    log_min = np.log(fmin) / np.log(base) if fmin > 0 else 0
    log_max = np.log(fmax) / np.log(base)
    log_centers = np.linspace(log_min, log_max, n_filters + 2)
    hz_points = base ** log_centers

    # convert to FFT bin indices
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    f_centers = hz_points[1:-1]

    return filterbank(n_filters, n_fft, bin_indices), f_centers


# TODO lowpass, highpass, bandpass, bandstop, weighted filter (seewave), rolloff like in audacity f-filter (tuneR)

    # # Initialize the weights
    # n_mels = int(n_mels)
    # weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # # Center freqs of each FFT bin
    # fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # # 'Center freqs' of mel bands - uniformly spaced between limits
    # mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    # fdiff = np.diff(mel_f)
    # ramps = np.subtract.outer(mel_f, fftfreqs)

    # for i in range(n_mels):
    #     # lower and upper slopes for all bins
    #     lower = -ramps[i] / fdiff[i]
    #     upper = ramps[i + 2] / fdiff[i + 1]

    #     # .. then intersect them with each other and zero
    #     weights[i] = np.maximum(0, np.minimum(lower, upper))

    # if isinstance(norm, str):
    #     if norm == "slaney":
    #         # Slaney-style mel is scaled to be approx constant energy per channel
    #         enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    #         weights *= enorm[:, np.newaxis]
    #     else:
    #         raise ParameterError(f"Unsupported norm={norm}")
    # else:
    #     weights = util.normalize(weights, norm=norm, axis=-1)

    # # Only check weights if f_mel[0] is positive
    # if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    #     # This means we have an empty channel somewhere
    #     warnings.warn(
    #         "Empty filters detected in mel frequency basis. "
    #         "Some channels will produce empty responses. "
    #         "Try increasing your sampling rate (and fmax) or "
    #         "reducing n_mels.",
    #         stacklevel=2,
    #     )

    # return weights
