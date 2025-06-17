from numpy.typing import NDArray, ArrayLike
import numpy as np
import logging
from typing import Optional, Literal, Union, Tuple, Any
import warnings
from scipy.signal import windows

def check_sr_format(sr: Union[int, float]) -> int:
    try:
        sr = int(sr)
    except Exception as e:
        raise TypeError(f"Sample rate not transformable to integer: {e}")
    if sr <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    return sr


def check_signal_format(data: ArrayLike) -> NDArray[np.float32]:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Signal must be a 1D array.")
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError("Signal must be an array of type float.")
    # if np.max(data) > 1:
        # scale to -1 to 1
        # TODO
    return data


def exclude_trailing_and_leading_zeros(envelope: ArrayLike) -> NDArray[np.float32]:
    """
    Removes leading and trailing zeros from a NumPy array.

    This function identifies and excludes the leading and trailing
    zeros in the input array `envelope`. It finds the first and last non-zero
    elements, and returns a new array that starts from the first non-zero value
    and ends at the last non-zero value.

    Args:
        envelope : ArrayLike
            A 1D NumPy array containing numerical values, 
            potentially with leading and/or trailing zeros.

    Returns:
        np.ndarray
            A 1D NumPy array with leading and trailing zeros excluded. 
            The returned array will only contain values between the first 
            and last non-zero elements from the original array.

    Example:
        >>> import numpy as np
        >>> arr = np.array([0, 0, 1, 2, 3, 0, 0])
        >>> exclude_trailing_and_leading_zeros(arr)
        array([1, 2, 3])

    Notes:
        - If the array consists entirely of zeros, an empty array will be returned.
        - The function assumes that the input array is 1D.

    """
    if envelope.ndim != 1:
        raise ValueError("Input array must be 1D.")
    
    if np.all(envelope == 0):  # check if all zeros and return empty array
        return np.array([])

    try:
        # Exclude trailing zeros (identify last non-zero element)
        non_zero_end = np.argmax(envelope[::-1] > 0)
        envelope = envelope[:len(envelope) - non_zero_end]
        
        # Exclude leading zeros (identify first non-zero value)
        non_zero_start = np.argmax(envelope > 0)
        envelope = envelope[non_zero_start:]
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    
    return envelope


def probability_mass_function(envelope: NDArray[np.float32]) -> NDArray[np.float32]:
    return envelope / np.sum(envelope)


def cumulative_distribution_function(envelope: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.cumsum(probability_mass_function(envelope))


def extract_all_features(
    data: ArrayLike, 
    sr: int, 
    kernel_size: Optional[int] = None,
    n_dominant_freqs: int = 1
) -> dict[str, Union[float, NDArray[np.float64]]]:
    """
    Extracts a comprehensive set of temporal and spectral features from a signal.

    Args:
        data (ArrayLike): Input 1D signal.
        sr (int): Sampling rate in Hz.
        kernel_size (Optional[int]): Size of smoothing kernel for amplitude envelope.
        n_dominant_freqs (int): Number of dominant frequencies to extract per frame.

    Returns:
        dict: Dictionary of extracted features.
    """
    from .spectral import spectral_features
    from .temporal import temporal_features
    
    data = check_signal_format(data)
    sr = check_sr_format(sr)

    temporal_feats = temporal_features(data, sr, kernel_size)
    spectral_feats = spectral_features(data, sr)

    return {**temporal_feats, **spectral_feats}


def transform_spectrogram_for_nn(
        spectrogram: ArrayLike,
        values_type: str = 'float32', 
        add_channel: bool = True,
        data_format: Literal['channels_last', 'channels_first'] = 'channels_first'
    ) -> ArrayLike:
    """
    Prepares a spectrogram for input into a neural network by normalizing, casting type,
    and optionally adding a channel dimension.

    Parameters:
        spectrogram : ArrayLike 
            The input spectrogram as a 2D array (frequency x time).
        values_type : str
            Data type to cast the spectrogram to (e.g., 'float32', 'float64'). Defaults to 'float32'
        add_channel : bool
            Whether to add a greyscale channel dimension.
        data_format : Literal['channels_last', 'channels_first']
            Specifies channel dimension placement when `add_channel` is True. 
            - 'channels_last' results in shape (H, W, 1) - e.g. for TensorFlow/Keras
            - 'channels_first' results in shape (1, H, W) - e.g. for PyTorch

    Returns:
        ArrayLike : 
            The transformed spectrogram, normalized to [0, 1], cast to the specified
            data type, and optionally with a channel dimension added.
    
    Example:
        >>> import numpy as np
        >>> spec = np.random.rand(128, 128) * 255  # Example spectrogram
        >>> processed = transform_spectrogram_for_nn(spec, values_type='float32',
        ...                                          add_channel=True, data_format='channels_last')
        >>> processed.shape
        (128, 128, 1)
        >>> processed.dtype
        dtype('float32')
    """
    # TODO Error handling, type checks
    if spectrogram.max() - spectrogram.min() == 0:
        warnings.warn(f"Spectrogram contains no information (values in range [{spectrogram.min()}, {spectrogram.max()}]).", RuntimeWarning)
    else:
        # min-max-scale to range [0, 1]
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

    # convert to desired bit depth
    spectrogram = spectrogram.astype(values_type)
    
    # add channel dimension (e.g., grayscale)
    if add_channel:
        if data_format == 'channels_last':
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # (H, W, 1)
        else:
            spectrogram = np.expand_dims(spectrogram, axis=0)   # (1, H, W)

    return spectrogram.astype(values_type)


def shannon_entropy(
        prob_dist : ArrayLike, 
        unit : Literal["bits", "nat", "dits", "bans", "hartleys"] = "bits",
        norm : bool = True
    ) -> Tuple[float, float]:

    if unit == "bits":
        log_ : Any = np.log2
    elif unit == "nat":
        log_ = np.log
        #H = np.negative(np.sum(prob_dist * np.log(prob_dist)))
    elif unit in ["dits", "bans", "hartleys"]:
        log_ = np.log10
        #H = np.negative(np.sum(prob_dist * np.log10(prob_dist)))
    else:
        raise ValueError(f'Invalid unit for power spectral entropy: {unit} Must be in ["bits", "nat", "dits", "bans", "hartleys"]')

    if np.all(prob_dist == prob_dist[0]):
        return 0.0, 1 if norm else log_(len(prob_dist))

    if norm:
        H = np.negative(np.sum(prob_dist * (log_(prob_dist)/log_(len(prob_dist)))))
        max = 1
    else:
        H = np.negative(np.sum(prob_dist * log_(prob_dist)))
        max = log_(len(prob_dist))
    
    return float(H), float(max)


def hz_to_mel(
        f: Union[ArrayLike, float],
        a: Optional[float] = None,
        b: Optional[float] = None,
        corner_frequency: Optional[float] = None,
        after : Literal["fant", "koenig", "oshaughnessy", "umesh"] = "oshaughnessy"
    ) -> Union[ArrayLike, float]:
    """
    Converts a frequency or array of frequencies in Hertz to the Mel scale
    using one of several proposed formulas.

    Parameters:
        f : Union[ArrayLike, np.floating])
            Frequency or array of frequencies in Hz.
        a : Optional[float])
            Scaling factor for the selected formula. Default depends on `after`.
        b : Optional[float]
            Denominator or offset parameter. Default depends on `after`.
        corner_frequency : Optional[float]
            If corner frequency other than 1000 Hz is desired, given in Hz. If provided, calculations are based on Fant (1970).
        after (Literal): Choice of Mel scale formula to use.
            - 'fant': Classic formula with `a=b=1000`: `F_m = a * np.log(1 + f / b)`
            - 'koenig': (Not yet implemented)
            - 'oshaughnessy' or 'beranek': Commonly used formula as fant, but with `a=2595`, `b=700`
            - 'umesh': Formula using rational function with `a=0.0004`, `b=0.603`.

    Returns:
        np.ndarray: Frequencies mapped to the Mel scale.


    References:
        1. Fant G. 1970 Acoustic Theory of Speech Production. The Hague: Mouton & Co. 
        2. Umesh S, Cohen L, Nelson D. 1999 Fitting the Mel scale. In 1999 IEEE International 
           Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No.99CH36258), 
           pp. 217–220 vol.1. Phoenix, AZ, USA: IEEE. (doi:10.1109/ICASSP.1999.758101)
        3. D. O'Shaughnessy, ”Speech Communication - Human and Machine” Addison- Wesley, New York, 1987. As cited in [2]

    """
    f_arr = np.asarray(f, dtype=np.float64)

    if corner_frequency is not None:
        a, b = corner_frequency, corner_frequency
        after = "fant"

    if after == "fant":
        a = 1000.0 if a is None else a
        b = 1000.0 if b is None else b
        F_m = a * np.log(1 + f_arr / b)
    elif after in ["oshaughnessy", "beranek"]:
        a = 2595.0 if a is None else a
        b = 700.0 if b is None else b
        F_m = a * np.log(1 + f_arr / b)
    elif after == "umesh":
        a = 0.0004 if a is None else a
        b = 0.603 if b is None else b
        F_m = f_arr / (a * f_arr + b)
    elif after == "koenig":
        raise NotImplementedError("The 'koenig' Mel scale formula is not yet implemented.")
    else:
        raise ValueError(f"Unknown Mel scale method: '{after}'")

    return F_m


def mel_to_hz(
    m: Union[ArrayLike, float],
    a: Optional[float] = None,
    b: Optional[float] = None,
    corner_frequency: Optional[float] = None,
    after: Literal["fant", "koenig", "oshaughnessy", "umesh"] = "oshaughnessy"
) -> Union[ArrayLike, float]:
    # TODO Slaney
    """
    Converts a Mel scale value or array of values to frequency in Hertz,
    using one of several inverse Mel formulas.

    Parameters
    ----------
    m : Union[ArrayLike, float]
        Mel frequency or array of Mel frequencies.
    a : Optional[float]
        Scaling factor for the selected formula. Default depends on `after`.
    b : Optional[float]
        Denominator or offset parameter. Default depends on `after`.
    corner_frequency : Optional[float]
        If a corner frequency other than 1000 Hz is desired, given in Hz. If provided, calculations are based on Fant (1970).
    after : Literal
        Choice of Mel scale formula to use.
        - 'fant': Classic formula: `f = b * (exp(m/a) - 1)` with `a=b=1000`
        - 'oshaughnessy' or 'beranek': Common formula: `a=2595`, `b=700`
        - 'umesh': Rational model: `f = b * m / (1 - a * m)`
        - 'koenig': (Not yet implemented)

    Returns
    -------
    Union[ArrayLike, float]
        Frequency or array of frequencies in Hertz.

    References
    ----------
    1. Fant G. 1970 Acoustic Theory of Speech Production.
    2. Umesh S, Cohen L, Nelson D. 1999 Fitting the Mel scale. ICASSP.
    3. D. O'Shaughnessy, "Speech Communication - Human and Machine", 1987.
    """
    m_arr = np.asarray(m, dtype=np.float64)

    if corner_frequency is not None:
        a = corner_frequency
        b = corner_frequency
        after = "fant"

    if after == "fant":
        a = 1000.0 if a is None else a
        b = 1000.0 if b is None else b
        f_hz = b * (np.exp(m_arr / a) - 1)
    elif after in ["oshaughnessy", "beranek"]:
        a = 2595.0 if a is None else a
        b = 700.0 if b is None else b
        f_hz = b * (np.exp(m_arr / a) - 1)
    elif after == "umesh":
        a = 0.0004 if a is None else a
        b = 0.603 if b is None else b
        denominator = 1 - a * m_arr
        if np.any(denominator == 0):
            raise ZeroDivisionError("Denominator in Umesh model became zero.")
        f_hz = b * m_arr / denominator
    elif after == "koenig":
        raise NotImplementedError("The 'koenig' Mel scale formula is not yet implemented.")
    else:
        raise ValueError(f"Unknown Mel scale method: '{after}'")

    return f_hz


def frame_signal(
        data: ArrayLike,
        sr : int,
        window_length : int = 512, 
        timestep : float = 0.01,
        normalize : bool = False
    ) -> ArrayLike:
    samples_step = timestep * sr
    data = np.pad(data, int(window_length / 2), mode='reflect')

    frame_num = int((len(data) - window_length) / samples_step) + 1
    frames = np.zeros((frame_num, window_length))
    
    for n in range(frame_num):
        start = int(n * samples_step)
        frames[n] = data[start : start + window_length]
    
    if normalize:
        frames = [frame / np.mean(frame) if frame.any() else frame for frame in frames] # skip normalization for frames with all 0
        
    return frames


def window_signal(
        data: ArrayLike,
        sr : int, 
        window_length: int = 512,
        window: Union[str, ArrayLike] = "hann",
        timestep : float = 0.01,
        normalize : bool = False
) -> ArrayLike:
    
    if isinstance(window, str):
        try:
            window = windows.get_window(window, window_length)
        except ValueError as e:
            raise ValueError(f"Invalid window type: {window}") from e
    else:
        window = np.asarray(window) 
        if not isinstance(window, np.ndarray):
            raise TypeError("'window' must be either a string or a 1D NumPy array.")

    frames = frame_signal(data, sr, window_length, timestep, normalize)

    return frames * window