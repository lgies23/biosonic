from numpy.typing import NDArray, ArrayLike
import numpy as np
import logging
from typing import Optional, Literal, Union
import warnings


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
    return data


def exclude_trailing_and_leading_zeros(envelope: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Removes leading and trailing zeros from a NumPy array.

    This function identifies and excludes the leading and trailing
    zeros in the input array `envelope`. It finds the first and last non-zero
    elements, and returns a new array that starts from the first non-zero value
    and ends at the last non-zero value.

    Args:
        envelope (np.ndarray): A 1D NumPy array containing numerical values, 
                                potentially with leading and/or trailing zeros.

    Returns:
        np.ndarray: A 1D NumPy array with leading and trailing zeros excluded. 
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
    data: NDArray[np.float32], 
    sr: int, 
    kernel_size: Optional[int] = None,
    n_dominant_freqs: int = 1
) -> dict[str, Union[float, NDArray[np.float64]]]:
    """
    Extracts a comprehensive set of temporal and spectral features from a signal.

    Args:
        data (NDArray[np.float32]): Input 1D signal.
        sr (int): Sampling rate in Hz.
        kernel_size (Optional[int]): Size of smoothing kernel for amplitude envelope.
        n_dominant_freqs (int): Number of dominant frequencies to extract per frame.

    Returns:
        dict: Dictionary of extracted features.
    """
    from .spectral import spectral_features
    from .temporal import temporal_features
    
    check_signal_format(data)
    check_sr_format(sr)

    temporal_feats = temporal_features(data, sr, kernel_size)
    spectral_feats = spectral_features(data, sr, n_freqs=n_dominant_freqs)

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