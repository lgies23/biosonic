from scipy.io import wavfile
from scipy.signal import resample
from numpy.typing import NDArray, ArrayLike
import pandas as pd
from pathlib import Path
import numpy as np
# from dataclasses import dataclass
import traceback
from typing import Any, Literal, Union, Optional, get_args, Tuple, List, Dict

QuantizationStr = Literal["int8", "int16", "int32", "float32", "float64"]

# @dataclass
# class Signal:
#     """
#     A dataclass representing an audio signal.

#     Attributes:
#         data : NDArray
#             Audio samples as a NumPy array. 1D for mono, otherwise a 2D array with shape (n_samples, n_channels)
#         n_channels : int 
#             Number of audio channels (1 for mono, 2 for stereo, etc.).
#         sr : int
#             Sample rate in Hz.
#         quantization : QuantizationStr
#             Data format (bit depth) of the signal.
#     """
#     data: NDArray
#     n_channels: int
#     sr: int
#     quantization: QuantizationStr


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
    if target_dtype not in get_args(QuantizationStr):
        raise ValueError(f"Invalid quantization: {target_dtype}. Must be one of {get_args(QuantizationStr)}")
    
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


def resample_audio(data: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """
    Resample audio data from an original sampling rate to a target sampling rate.

    Parameters
    ----------
    data : NDArray
        Input audio data array. Shape can be (n_samples,) for mono or (n_samples, n_channels) for multi-channel audio.
    orig_sr : int
        Original sample rate of the audio data in Hz.
    target_sr : int
        Desired sample rate in Hz.

    Returns
    -------
    NDArray
        Resampled audio data with shape adjusted to the target sample rate.
        Number of channels remains unchanged.

    Notes
    -----
    - If `orig_sr` is equal to `target_sr`, the original data is returned unchanged.
    - Uses `scipy.signal.resample` internally for resampling each channel independently.
    """
    if orig_sr == target_sr:
        return data
    n_samples = round(data.shape[0] * target_sr / orig_sr)
    if data.ndim == 1:
        return resample(data, n_samples)
    else:
        return np.stack([resample(data[:, ch], n_samples) for ch in range(data.shape[1])], axis=1)


def convert_channels(data: NDArray, target_channels: int) -> NDArray:
    """
    Convert audio data to a target number of channels (mono or stereo).

    Parameters
    ----------
    data : NDArray
        Input audio data array. Shape is (n_samples,) for mono or (n_samples, n_channels) for multi-channel.
    target_channels : int
        Desired number of channels. Supported values are 1 (mono) or 2 (stereo).

    Returns
    -------
    NDArray
        Audio data converted to the target number of channels.

    Raises
    ------
    NotImplementedError
        If `target_channels` is greater than 2.

    Notes
    -----
    - Converts stereo to mono by averaging channels.
    - Converts mono to stereo by duplicating the mono channel.
    - If the data already has the target number of channels, it is returned unchanged.
    """
    if target_channels > 2: 
        raise NotImplementedError("Conversion to more than 2 channels not implemented yet.")
    
    n_ch = 1 if data.ndim == 1 else data.shape[1]

    if n_ch == target_channels:
        return data

    if target_channels == 1:
        # Convert stereo to mono by averaging channels
        return data.mean(axis=1)
        
    elif target_channels == 2:
        # Convert mono to stereo by duplicating
        return np.stack([data, data], axis=1)
        
    return data


def read_wav(
        filepath: Union[str, Path], 
        sampling_rate : Optional[int] = None,
        quantization: QuantizationStr = "float32",
        n_channels : Optional[int] = None,
    ) -> Tuple[NDArray, int, int, QuantizationStr]:
    """
    Reads a WAV file and returns a Signal object, optionally converting sample rate, number of channels,
    and quantization format.

    Parameters
    ----------
    filepath : str or Path
        Path to the WAV file to read.
    sampling_rate : int, optional
        Target sample rate in Hz. If specified and different from the file's original sample rate,
        resampling should be applied (using scipy.signal.resample - Fourier method).
        If None, the original sample rate is kept.
    quantization : {'int8', 'int16', 'int32', 'float32', 'float64'}, optional
        Desired output data format for the audio samples (default is "float32").
    n_channels : int, optional
        Target number of audio channels (e.g., 1 for mono, 2 for stereo).
        If specified and different from the original number of channels, conversion
        is be applied (downmixing from stereo to mono (np.mean) and copying from mono to stereo).
        If None, the original number of channels is kept.

    Returns
    -------
    A tuple containing:
        - data : np.ndarray
            Audio samples, shape (n_samples,) for mono or (n_samples, n_channels) otherwise.
        - sr : int
            Sample rate in Hz.
        - n_channels : int
            Number of audio channels.
        - quantization : str
            Data format of the signal.

    Notes
    -----
    - Uses `scipy.io.wavfile` to read WAV files.
    - 24-bit WAV files are stored as `np.int32` (as in `scipy.io.wavfile`).

    References
    ----------
    Virtanen P et al. 2020 SciPy 1.0: fundamental algorithms for scientific computing in Python. 
    Nat Methods 17, 261–272. (doi:10.1038/s41592-019-0686-2)
    """
    sr, data = wavfile.read(filepath)

    if sampling_rate is not None:
        if sr != sampling_rate:
            data = resample_audio(data, sr, sampling_rate)
            sr = sampling_rate
    
    n_ch = 1 if data.ndim == 1 else data.shape[1]
    if n_channels is not None:
        if n_channels != n_ch:
            data = convert_channels(data, n_channels)
            n_ch = n_channels

    if quantization != data.dtype.name:
        data = convert_dtype(data, quantization)

    return data, sr, n_ch, quantization


def batch_normalize_wav_files(
    folder_path: Union[str, Path],
    target_sr: int,
    target_channels: int,
    target_quantization: QuantizationStr,
    output_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Batch normalize all WAV files in a folder to the same sample rate, number of channels, and quantization.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing input WAV files to normalize.
    target_sr : int
        Target sample rate in Hz to convert all audio files to.
    target_channels : int
        Target number of audio channels (e.g., 1 for mono, 2 for stereo).
    target_quantization : {'int8', 'int16', 'int32', 'float32', 'float64'}
        Target bit depth / data format for the output audio files.
    output_dir : str or Path, optional
        Directory to save the normalized WAV files. If None, a 'normalized' subfolder
        will be created inside `folder_path`.

    Returns
    -------
    None
        This function saves the normalized WAV files to disk and does not return anything.

    Notes
    -----
    - Input WAV files with extensions '.wav' and '.WAV' are processed.
    - Uses `read_wav` for loading and converting audio files. This attaches to scipys wavfile.io.read function.
    - Output files are saved with the same filename in the output directory. 
    So if you set output_dir to your folder_path, **all origninal files will be overwritten!**
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir) if output_dir else folder_path / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.WAV"))
    for wav_file in wav_files:
        data, _, _, _ = read_wav(wav_file, target_sr, target_quantization, target_channels)
        out_path = output_dir / wav_file.name
        wavfile.write(out_path, target_sr, data.astype(np.dtype(target_quantization)))
        print(f"Normalized: {wav_file.name} -> {out_path}")


def batch_extract_features(
        folder_path : Union[str, Path],
        save_csv_path: Optional[str] = None
) -> pd.DataFrame :
    
    from biosonic.compute.utils import extract_all_features

    folder_path = Path(folder_path)
    feature_rows = []

    wav_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.WAV"))
    for wav_file in wav_files:
        print(f"processing {wav_file.name}")
        try:
            data, sr, _, _ = read_wav(wav_file)
            features = extract_all_features(data, sr)
            features['filename'] = wav_file.name
            feature_rows.append(features)
        except Exception as e:
            print(f"Failed to process {wav_file.name}: {e}")
            traceback.print_exc()
    
    out_df = pd.DataFrame(feature_rows)

    if save_csv_path:
        try:
            out_df.to_csv(save_csv_path, index=False)
            print(f"Features saved to: {save_csv_path}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    return out_df


def segments_from_signal(
        data : NDArray, 
        sr : int,
        boundaries : Union[Dict[str, float], ArrayLike, Tuple[float, float], List[Dict[str, float]]]
    ) -> List[NDArray]:
    """
    Extract segments from an audio signal based on time boundaries.

    Parameters:
    -----------
    data : NDArray
        The audio signal array.
    sr : int
        Sampling rate of the audio signal.
    boundaries : Union[Dict[str, float], Tuple[float, float], ArrayLike, List[Dict[str, float]]]
        Segment boundaries. Supported formats:
            - Single dict with 'begin' and 'end' keys
            - Tuple of (begin, end)
            - 2D ArrayLike of shape (n, 2), each row as (begin, end)
            - List of dicts with 'begin' and 'end' keys

    Returns:
    --------
    List[NDArray]
        A list of segmented portions of the audio signal.
    """
    segments = []

    # Normalize boundaries to a list of (begin, end) tuples
    if isinstance(boundaries, dict):
        boundaries = [(boundaries["begin"], boundaries["end"])]
    elif isinstance(boundaries, tuple) and len(boundaries) == 2:
        boundaries = [boundaries]
    elif isinstance(boundaries, list) and all(isinstance(b, dict) for b in boundaries):
        boundaries = [(b["begin"], b["end"]) for b in boundaries]
    elif isinstance(boundaries, (np.ndarray, list)):
        boundaries = list(boundaries)  # ensure list of lists or tuples
    else:
        raise ValueError("Unsupported boundary format")

    for begin, end in boundaries:
        start_idx = int(np.floor(begin * sr))
        end_idx = int(np.ceil(end * sr))
        segments.append(data[start_idx:end_idx])
    
    return segments

def boundaries_from_textgrid(
        filepath : Union[str, Path],
        tier_name : str
    ) -> List[Dict[str, float]]:
    """
    Extracts segment boundaries from a specified tier in a TextGrid file.

    Parameters:
    -----------
    filepath : Union[str, Path]
        Path to the TextGrid file.
    tier_name : str
        The name of the tier from which to extract intervals.

    Returns:
    --------
    List[Dict[str, float]]
        A list of dictionaries, each containing 'begin', 'end', and 'label' keys 
        for non-empty intervals in the specified tier.
    """
    from biosonic.praat import read_textgrid

    grid = read_textgrid(filepath)
    segments = grid.interval_tier_to_array(tier_name=tier_name)
    return [segment for segment in segments if len(segment["label"]) > 0] 


def audio_segments_from_textgrid(
        data : NDArray, 
        sr : int,
        filepath_textgrid : Union[str, Path],
        tier_name : str
    ) -> List[Dict[NDArray, str]]:
    """
    Extracts and visualizes audio segments corresponding to labeled intervals 
    in a TextGrid file.

    Parameters:
    -----------
    data : NDArray
        The audio signal array.
    sr : int
        Sampling rate of the audio signal.
    filepath_textgrid : Union[str, Path]
        Path to the TextGrid file containing segmentation information.
    tier_name : str
        Name of the tier to extract labeled segments from.

    Returns:
    --------
    List[Dict[NDArray, str]]
        A list of dictionaries, each containing:
            - The audio segment (NDArray) for each labeled interval.
            - The corresponding label (str).
    """
    from biosonic.plot import plot_boundaries_on_spectrogram

    boundaries = boundaries_from_textgrid(filepath_textgrid, tier_name)

    plot_boundaries_on_spectrogram(data, sr, boundaries)
    segments = segments_from_signal(data, sr, boundaries)
    return [{"segment": seg, "label": str(b["label"])} for seg, b in zip(segments, boundaries)]