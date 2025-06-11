import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_convert_dtype():
    from biosonic.handle import convert_dtype
    original = np.array([-32768, 0, 1246, 32767], dtype=np.int16)

    # invalid string
    with pytest.raises(ValueError):
        convert_dtype(original, "invalid")
    
    # int16 -> float32 -> int16
    float_data = convert_dtype(original, "float32")
    recovered = convert_dtype(float_data, "int16")

    # allow minor loss due to float rounding
    assert np.allclose(original, recovered, atol=1)

    # uint8 to float
    original = np.array([0, 128, 255], dtype=np.uint8)
    float_data = convert_dtype(original, "float32")
    expected = np.array([-1.0, 0.0, 0.9921875], dtype=np.float32)
    assert np.allclose(float_data, expected, atol=1e-5)


@pytest.fixture
def mono_signal():
    return np.linspace(-1.0, 1.0, 44100)

@pytest.fixture
def stereo_signal(mono_signal):
    return np.stack([mono_signal, mono_signal], axis=1)


def test_resample_audio(mono_signal, stereo_signal):
    from biosonic.handle import resample_audio

    # mono
    orig_sr = 44100
    target_sr = 22050
    resampled = resample_audio(mono_signal, orig_sr, target_sr)
    expected_length = round(len(mono_signal) * target_sr / orig_sr)
    assert resampled.shape == (expected_length,)
    assert np.isclose(np.mean(resampled), np.mean(mono_signal), atol=1e-2)

    # stereo
    orig_sr = 44100
    target_sr = 22050
    resampled = resample_audio(stereo_signal, orig_sr, target_sr)
    expected_length = round(stereo_signal.shape[0] * target_sr / orig_sr)
    assert resampled.shape == (expected_length, 2)
    assert np.isclose(np.mean(resampled), np.mean(stereo_signal), atol=1e-2)

    # no-op when sampling rate is the same
    resampled = resample_audio(mono_signal, 44100, 44100)
    assert np.shares_memory(resampled, mono_signal)


def test_convert_channels(mono_signal, stereo_signal):
    from biosonic.handle import convert_channels

    # mono
    mono = convert_channels(stereo_signal, 1)
    assert mono.ndim == 1
    assert mono.shape[0] == stereo_signal.shape[0]
    assert np.allclose(mono, stereo_signal.mean(axis=1))

    # stereo
    stereo = convert_channels(mono_signal, 2)
    assert stereo.ndim == 2
    assert stereo.shape == (mono_signal.shape[0], 2)
    assert np.allclose(stereo[:, 0], mono_signal)
    assert np.allclose(stereo[:, 1], mono_signal)

    # no-op mono
    result = convert_channels(mono_signal, 1)
    assert np.shares_memory(result, mono_signal)

    # no-op stereo
    result = convert_channels(stereo_signal, 2)
    assert np.shares_memory(result, stereo_signal)

    # invalid
    data = np.zeros((100,))
    with pytest.raises(NotImplementedError):
        convert_channels(data, 4)


def test_read_wav(tmp_path):
    from scipy.io import wavfile
    from biosonic.handle import read_wav
    # create dummy WAV file
    sr = 44100
    data = (np.sin(2 * np.pi * 440 * np.arange(44100) / sr) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "test.wav", sr, data)

    signal = read_wav(tmp_path / "test.wav", quantization="float32")

    assert signal.sr == 44100
    assert signal.n_channels == 1
    assert signal.data.dtype == np.float32


# @pytest.fixture
# def mock_wav_file(tmp_path):
#     """Creates a dummy wav file in a temporary folder"""
#     wav_file = tmp_path / "mocked.wav"
#     wav_file.write_bytes(b"RIFF....WAVE")  # minimal fake wav header
#     return wav_file

@patch("biosonic.handle.Path.glob")
@patch("biosonic.handle.read_wav")
@patch("biosonic.compute.utils.extract_all_features")
def test_batch_extract_feature(mock_extract, mock_read, mock_glob, tmp_path):
    from biosonic.handle import batch_extract_features
    # success
    mock_file = MagicMock(spec=Path)
    mock_file.name = "mocked.wav"
    mock_file.suffix = ".wav"
    mock_file.__str__.return_value = "mocked.wav"
    mock_glob.return_value = [mock_file]

    mock_signal = MagicMock()
    mock_signal.data = [0.1, 0.2, 0.3]
    mock_signal.sr = 44100
    mock_read.return_value = mock_signal
    mock_extract.return_value = {'feature1': 1.0, 'feature2': 2.0}

    df = batch_extract_features(str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2 # because glob is called two times
    assert 'filename' in df.columns
    assert 'feature1' in df.columns
    assert df['filename'].iloc[0] == "mocked.wav"

    
    # csv output
    csv_path = tmp_path / "output.csv"
    df = batch_extract_features(str(tmp_path), save_csv_path=str(csv_path))
    assert csv_path.exists()
    saved_df = pd.read_csv(csv_path)
    assert saved_df.shape[0] == 2
    assert 'filename' in saved_df.columns


    # simulate exception in feature extraction
    mock_signal = MagicMock()
    mock_signal.data = [0.1, 0.2]
    mock_signal.sr = 44100
    mock_read.return_value = mock_signal
    mock_extract.side_effect = RuntimeError("Feature extraction failed")

    df = batch_extract_features(str(tmp_path))
    assert df.empty  # should skip failed file


    # empty folder
    mock_glob.return_value = []
    df = batch_extract_features(str(tmp_path))
    assert df.empty
