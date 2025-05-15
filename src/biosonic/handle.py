from scipy.io import wavfile
from numpy.typing import NDArray, ArrayLike
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Literal, Union

QuantizationStr = Literal["int8", "int16", "int32", "float32", "float64"]

@dataclass
class Signal:
    """
    A dataclass representing an audio signal.

    Attributes:
        data : NDArray
            Audio samples as a NumPy array. 1D for mono, otherwise a 2D array with shape (n_samples, n_channels)
        n_channels : int 
            Number of audio channels (1 for mono, 2 for stereo, etc.).
        sr : int
            Sample rate in Hz.
        quantization : QuantizationStr
            Data format (bit depth) of the signal.
    """
    data: NDArray
    n_channels: int
    sr: int
    quantization: QuantizationStr


def convert_dtype(data: NDArray, target_dtype: QuantizationStr) -> NDArray:
    """
    Converts audio data to the specified quantization format.

    Args:
        data : NDArray
            Input audio data.
        target_dtype : QuantizationStr
            Target NumPy dtype as a string.

    Returns:
        NDArray 
            Converted audio data with appropriate scaling and clipping.
    """
    target_np_dtype: np.dtype[np.generic] = np.dtype(target_dtype)
    current_dtype = data.dtype

    if current_dtype == target_np_dtype:
        return data
    
    # special handling for uint8 (unsigned), to float32
    if current_dtype == np.uint8 and np.issubdtype(target_np_dtype, np.floating):
        return ((data.astype(target_np_dtype) - 128) / 128).astype(target_np_dtype)

    if np.issubdtype(current_dtype, np.floating) and target_np_dtype == np.uint8:
        return ((data * 128) + 128).clip(0, 255).astype(np.uint8)

    # integer to float: normalize
    if np.issubdtype(current_dtype, np.integer) and np.issubdtype(target_np_dtype, np.floating):
        max_val = np.iinfo(current_dtype).max
        return (data.astype(np.float32) / max_val).astype(target_np_dtype)

    # float to integer: scale and clip
    if np.issubdtype(current_dtype, np.floating) and np.issubdtype(target_np_dtype, np.integer):
        max_val = np.iinfo(target_np_dtype).max
        return (data * max_val).clip(-max_val, max_val - 1).astype(target_np_dtype)

    # integer to integer or float to float
    return data.astype(target_np_dtype)


def read_wav(filepath: Union[str, Path], quantization: QuantizationStr = "float32") -> Signal:
    """
    Reads a WAV file and returns a Signal object, optionally converting to the desired quantization.

    Args:
        filepath : str | Path
            Path to the WAV file.
        quantization : QuantizationStr
            Desired output data format (default is "float32").

    Returns:
        Signal
            The loaded and optionally converted audio signal.
    
    Notes:
        - This function uses scipy.io.wavfile
        - 24bit wav files will be stored as np.int32 (as in scipy.io.wavfile)
    
    References:
        Virtanen P et al. 2020 SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272. (doi:10.1038/s41592-019-0686-2)
    """
    sr, data = wavfile.read(filepath)
    n_channels = 1 if data.ndim == 1 else data.shape[1]

    if quantization != data.dtype.name:
        data = convert_dtype(data, quantization)

    return Signal(data, n_channels, sr, quantization)