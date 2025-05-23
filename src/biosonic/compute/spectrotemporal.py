from typing import Optional, Tuple, Union, Dict, Literal, Any
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import signal
from scipy.fft import fft, ifft

from .temporal import temporal_entropy
from .spectral import power_spectral_entropy
from .utils import check_signal_format, check_sr_format
from ..filter import linear_filterbank, mel_filterbank, log_filterbank


def spectrogram(
        data: ArrayLike, 
        sr: int, 
        window: Union[str, ArrayLike] = "hann", 
        window_length: int = 1024, 
        overlap: float = .5, 
        scaling: str = "magnitude",
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the spectrogram of a 1D signal using short-time Fourier transform (STFT).

    Args:
        data : ArrayLike
            Input signal as a 1D array-like.
        sr : int
            Sampling rate of the signal in Hz.
        window : Union[str, NDArray], optional
            Type of window to use (e.g., 'hann', 'hamming' of scipy.signal.windows) or a custom window array.
            Defaults to 'hann'.
        window_length : int, optional
            Length of the analysis window in samples.
            Defaults to 1024. Ignored if the window is given as a custom array,
        overlap : float, optional
            Fractional overlap between adjacent windows (0 < overlap < 1).
            Defaults to 0.5 (50% overlap).
        scaling : str, optional
            scaling of the spectrogram, either 'magnitude' or 'psd' for power spectral density. 
            Defaults to 'magnitude'.
        *args : tuple
            Additional positional arguments to pass to scipy's ShortTimeFFT class.
        **kwargs : dict
            Additional keyword arguments to pass to scipy's ShortTimeFFT class.

    Retuns:
        Sx : NDArray[np.float64]
            2D array of complex STFT values (frequency x time).
        t : NDArray[np.float64]
            Array of time values corresponding to the STFT columns.
        f : NDArray[np.float64]
            Array of frequency bins corresponding to the STFT rows.

    Notes:
        This function relies on scipy's `ShortTimeFFT` class and the submodule scipy.signal.windows.
    
    References:
        Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, 
        Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, 
        Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, 
        Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis 
        Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. 
        Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 
        (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 
        261-272. DOI: 10.1038/s41592-019-0686-2.
    """
    # todo more scaling, dynamic range, dB transform, invert, etc 
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    # pad_amt = 0 
    # if pad:
    #     pad_amt = window_length // 2
    #     data = np.pad(data, pad_amt, mode='reflect')

    if 0 > overlap > 1 or not isinstance(overlap, float):
        raise ValueError("Window overlap must be a float between 0 and 1.")

    if not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("'window_length' must be a positive integer.")

    if isinstance(window, str):
        try:
            window = signal.windows.get_window(window, window_length)
        except ValueError as e:
            raise ValueError(f"Invalid window type: {window}") from e
    elif not isinstance(window, np.ndarray):
        raise TypeError("'window' must be either a string or a 1D NumPy array.")
    
    hop_length = int(window_length * (1 - overlap))

    STFT = signal.ShortTimeFFT(window, hop_length, sr, scale_to=scaling, *args, **kwargs)
    Sx = STFT.stft(data, *args, **kwargs)
    
    # # adjust time vector for padding offset
    t = STFT.t(len(data))

    # trim frames whose center is outside the unpadded signal range
    duration = len(data) / sr  # real signal duration
    valid_mask = (t >= 0) & (t <= duration)
    Sx = Sx[:, valid_mask]
    t = t[valid_mask]

    return Sx, t, STFT.f


def cepstrum(
        data: ArrayLike,  
        sr: int,
        mode : Literal ["amplitude", "power"] = "power",
    ) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the cepstrum of a real-valued time-domain signal.

    The cepstrum is computed by taking the inverse Fourier transform 
    of the logarithm of the magnitude spectrum. Depending on the `mode`, 
    this function returns either the amplitude or power cepstrum.

    Parameters
    ----------
    data : ArrayLike
        Input real-valued time-domain signal (1D array).
    sr : int
        Sampling rate of the signal in Hz.
    mode : {"amplitude", "power"}, optional
        Type of cepstrum to compute:
        - "amplitude" : Returns the absolute value of the inverse FFT of the log-magnitude spectrum.
        - "power"     : Returns the squared magnitude of the inverse FFT of the log-power spectrum.
        Default is "power".

    Returns
    -------
    cepstrum : ArrayLike
        The computed cepstrum (amplitude or power based on the selected mode).
    quefrencies : ArrayLike
        Array of quefrency values (in seconds), corresponding to each element in the cepstrum.

    References
    ----------
    Childers DG, Skinner DP, Kemerait RC. 1977 
    The cepstrum: A guide to processing. Proc. IEEE 65, 1428–1443. 
    (doi:10.1109/PROC.1977.10747)
    """
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    if np.all(data == data[0]):
        raise ValueError("Cannot compute cepstrum of flat signal.")

    cepstrum_ = np.asarray([])

    if mode == "power":
        cepstrum_ = np.abs(ifft(np.log(np.abs(fft(data))**2)))**2

    elif mode == "amplitude":
        cepstrum_ = np.abs(ifft(np.log(np.abs(fft(data)))))
    
    else:
        raise ValueError(f"Invalid mode for cepstrum calculation: {mode}")

    quefrencies = np.arange(len(data)) / sr

    return cepstrum_, quefrencies


def cepstral_coefficients(
    signal: ArrayLike,
    sr: int,
    n_fft: int = 512,
    n_filters: int = 32,
    n_ceps: int = 16,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    filterbank_type: Literal["mel", "linear", "log"] = "mel",
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    **kwargs : Any
) -> ArrayLike:
    """
    Compute cepstral coefficients from a signal using the specified filter bank.

    Parameters
    ----------
    signal : ArrayLike
        Input time-domain signal.
    sr : int
        Sampling rate in Hz.
    n_fft : int
        FFT size.
    n_filters : int
        Number of filters in the filter bank.
    n_ceps : int
        Number of cepstral coefficients to return.
    fmin : float
        Minimum frequency for the filter bank.
    fmax : Optional[float]
        Maximum frequency for the filter bank. Defaults to Nyquist (sr/2).
    filterbank_type : {'mel', 'linear', 'log'}
        Type of filter bank to apply before DCT.
    dct_type : int
        Type of Discrete Cosine Transform. Typically 2 for MFCCs.
    norm : Optional[str]
        Norm parameter for `scipy.fftpack.dct`. Use 'ortho' for MFCCs.
    corner_frequency : Optional[float]
        Corner frequency for mel scaling if other than 1000 Hz is desired, given in Hz. If provided, calculations are based on Fant (1970).
    after : Literal
        Choice of Mel scale formula to use.
        - 'fant': Classic formula: `f = b * (exp(m/a) - 1)` with `a=b=1000`
        - 'oshaughnessy' or 'beranek': Common formula: `a=2595`, `b=700`
        - 'umesh': Rational model: `f = b * m / (1 - a * m)`
        - 'koenig': (Not yet implemented)
    base : float
        Base of the logarithmic spacing. Typically 2 (for octaves).

    Returns
    -------
    np.ndarray
        Cepstral coefficient array of shape (n_ceps,).
    """
    from scipy.fftpack import dct
    from scipy.signal import get_window

    # power spectrum
    windowed = signal * get_window("hann", len(signal), fftbins=True)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

    # filter bank selection
    if fmax is None:
        fmax = sr / 2

    if filterbank_type == "mel":
        fbanks = mel_filterbank(n_filters, n_fft, sr, fmin, fmax, **kwargs)
    elif filterbank_type == "linear":
        fbanks = linear_filterbank(n_filters, n_fft, sr, fmin, fmax)
    elif filterbank_type == "log":
        fbanks = log_filterbank(n_filters, n_fft, sr, fmin, fmax, **kwargs)
    else:
        raise ValueError(f"Unknown filterbank_type: {filterbank_type}")

    # filterbank energies
    energies = np.dot(fbanks, spectrum)
    log_energies = np.log(energies + 1e-30)  # epsilon for log stability

    # DCT to cepstral domain
    ceps = dct(log_energies, type=dct_type, norm=norm)[:n_ceps]

    return np.asarray(ceps)


def spectrotemporal_entropy(
        data: ArrayLike,  
        sr: int, 
        *args : Any, 
        **kwargs : Any
    ) -> float:
    """
    Compute the product of temporal_entropy and power spectral entropy of the input data.

    This function computes two information-theoretic measures — temporal temporal_entropy and
    power spectral entropy — using the same unit and multiplies their values 
    to produce a combined spectrotemporal complexity measure.

    Parameters
    ----------
    data : ArrayLike
        Input signal as a 1D ArrayLike.
    sr : int
        Sampling rate in Hz.
    unit : {"bits", "nat", "dits", "bans", "hartleys"}, optional
        The logarithmic base to use for temporal_entropy calculations.
        Default is "bits".
    *args : Any
        Additional positional arguments passed to `temporal_entropy` and `power_spectral_entropy`.
    **kwargs : Any
        Additional keyword arguments passed to `temporal_entropy` and `power_spectral_entropy`.

    Returns
    -------
    float
        The temporal_entropy of the input data.

    See Also
    --------
    temporal_entropy : Computes the temporal_entropy of the data.
    power_spectral_entropy : Computes the temporal_entropy in the frequency domain.
    """
    H_t, _ = temporal_entropy(data, *args, **kwargs)
    H_f, _ = power_spectral_entropy(data, sr, *args, **kwargs)
    return H_t * H_f


def dominant_frequencies(
        data: ArrayLike, 
        sr: int, 
        n_freqs: int = 1, 
        min_height: float = 0.05,
        min_distance: float = 0.05,
        min_prominence: float = 0.05,
        peak_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any, 
        **kwargs: Any
    ) -> NDArray[np.float64]: 
    """
    Extracts the dominant frequency or frequencies from each time frame of a spectrogram 
    based on the scipy.signal function find_peaks.

    Parameters:
    data : ArrayLike
        Input 1D audio signal.
    sr : int
        Sample rate of the input signal in Hz.
    n_freqs : Optional[int], default=3
        Number of dominant frequencies to extract per time frame.
        If 1, a 1D array is returned. If >1, a 2D array of shape (time_frames, n_freqs) is returned.
    min_height : Optional[float], default=0.05
        Minimum normalized height of a peak (as a fraction of the spectral magnitude range).
        Must be between 0 and 1.
    min_distance : Optional[float], default=0.05
        Minimum normalized distance between peaks (as a fraction of the total number of frequency bins).
        Must be between 0 and 1.
    min_prominence : Optional[float], default=0.05
        Minimum normalized prominence of a peak (as a fraction of the spectral magnitude range).
        Must be between 0 and 1.
    peak_kwargs : Optional[dict], default=None
        Additional keyword arguments passed directly to `scipy.signal.find_peaks`.
        These override any values calculated from `min_height`, `min_distance`, and `min_prominence`.
    *args, **kwargs :
        Additional arguments passed to the scipy.signal ShortTimeFFT class.
    
    Returns:
    NDArray[np.float64]
        - If n_freqs == 1:
            1D array of shape (time_frames,) containing the dominant frequency per frame.
        - If n_freqs > 1:
            2D array of shape (time_frames, n_freqs) containing the top `n_freqs` dominant
            frequencies per frame. NaNs are used to pad frames with fewer than `n_freqs` detected peaks.
    """   
    if not (0.0 <= min_height <= 1.0):
        raise ValueError("min_height must be between 0 and 1")
    
    if not (0.0 <= min_distance <= 1.0):
        raise ValueError("min_distance must be between 0 and 1")
    
    if not (0.0 <= min_prominence <= 1.0):
        raise ValueError("min_prominence must be between 0 and 1")

    spec, _, freqs = spectrogram(data, sr, *args, **kwargs)
    spec_real = np.abs(spec)
    num_frames = spec_real.shape[1]
    if n_freqs == 1:
        dominant_freqs = np.full(num_frames, np.nan)
    else:
        dominant_freqs = np.full((num_frames, n_freqs), np.nan)

    peak_kwargs = peak_kwargs or {}

    for t in range(num_frames):
        spectrum = spec_real[:, t]
        magnitude_range = float(np.max(spectrum)) - np.min(spectrum)

        default_peak_params = {
            "height" : magnitude_range*min_height, 
            "distance" : max(1, len(freqs)//(1/min_distance)), 
            "prominence" : magnitude_range*min_prominence
        }

        peak_params = {**default_peak_params, **peak_kwargs}
        peaks, _ = signal.find_peaks(spectrum, **peak_params)

        if len(peaks) > 0:
            sorted_peaks = peaks[np.argsort(spectrum[peaks])[::-1]]
            top_peaks = sorted_peaks[:n_freqs]
            top_freqs = freqs[top_peaks]

            if n_freqs == 1:
                dominant_freqs[t] = top_freqs[0]
            else:
                dominant_freqs[t, :len(top_freqs)] = top_freqs

    return dominant_freqs