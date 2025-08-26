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

    data, sr, n_channels, _ = read_wav(tmp_path / "test.wav", quantization="float32")

    assert sr == 44100
    assert n_channels == 1
    assert data.dtype == np.float32


# @pytest.fixture
# def mock_wav_file(tmp_path):
#     """Creates a dummy wav file in a temporary folder"""
#     wav_file = tmp_path / "mocked.wav"
#     wav_file.write_bytes(b"RIFF....WAVE")  # minimal fake wav header
#     return wav_file

@patch("biosonic.handle.Path.glob")
@patch("biosonic.handle.read_wav")
@patch("biosonic.compute.utils.extract_all_features")
def test_batch_extract_features(mock_extract, mock_read, mock_glob, tmp_path):
    from biosonic.handle import batch_extract_features
    # success
    mock_file = MagicMock(spec=Path)
    mock_file.name = "mocked.wav"
    mock_file.suffix = ".wav"
    mock_file.__str__.return_value = "mocked.wav"
    mock_glob.return_value = [mock_file]

    data = [0.1, 0.2, 0.3]
    sr = 44100
    mock_read.return_value = data, sr, 1, "float32"
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
    data = [0.1, 0.2]
    sr = 44100
    mock_read.return_value = data, sr, 1, "float32"
    mock_extract.side_effect = RuntimeError("Feature extraction failed")

    df = batch_extract_features(str(tmp_path))
    assert df.empty  # should skip failed file

    # empty folder
    mock_glob.return_value = []
    df = batch_extract_features(str(tmp_path))
    assert df.empty


@patch("biosonic.handle.Path.glob")
@patch("biosonic.handle.read_wav")
def test_batch_read_files(mock_read, mock_glob, tmp_path):
    from biosonic.handle import batch_read_files_to_df

    # success
    mock_file = MagicMock(spec=Path)
    mock_file.name = "mocked.wav"
    mock_file.suffix = ".wav"
    mock_file.__str__.return_value = "mocked.wav"
    mock_glob.return_value = [mock_file]

    data = [0.1, 0.2, 0.3]
    sr = 44100
    mock_read.return_value = data, sr, 1, "float32"

    df = batch_read_files_to_df(str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2 # because glob is called two times
    assert 'filename' in df.columns
    assert 'sr' in df.columns
    assert 'waveform' in df.columns
    assert df['filename'].iloc[0] == "mocked.wav"
    
    # csv output
    csv_path = tmp_path / "output.csv"
    df = batch_read_files_to_df(str(tmp_path), save_csv_path=str(csv_path))
    assert csv_path.exists()
    saved_df = pd.read_csv(csv_path)
    assert saved_df.shape[0] == 2
    assert 'filename' in saved_df.columns

    # simulate exception in feature extraction
    data = [0.1, 0.2]
    sr = 44100
    mock_read.side_effect = RuntimeError("Reading file failed")

    df = batch_read_files_to_df(str(tmp_path))
    assert df.empty  # should skip failed file

    # empty folder
    mock_glob.return_value = []
    df = batch_read_files_to_df(str(tmp_path))
    assert df.empty


@pytest.fixture
def sample_data():
    sr = 1000
    duration = 2
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    return data, sr

def test_segments_from_signal(sample_data):
    from biosonic.handle import segments_from_signal

    def assert_segment_correct(data, segment, sr, start_time, end_time):
        expected = data[int(np.floor(start_time * sr)) : int(np.ceil(end_time * sr))]
        np.testing.assert_array_equal(segment, expected)

    # dict
    data, sr = sample_data
    boundaries = {"begin": 0.5, "end": 1.0}
    segments = segments_from_signal(data, sr, boundaries)
    assert len(segments) == 1
    assert_segment_correct(data, segments[0], sr, 0.5, 1.0)

    # tuple
    data, sr = sample_data
    boundaries = (0.2, 0.6)
    segments = segments_from_signal(data, sr, boundaries)
    assert len(segments) == 1
    assert_segment_correct(data, segments[0], sr, 0.2, 0.6)

    # list of dicts
    data, sr = sample_data
    boundaries = [{"begin": 0.0, "end": 0.3}, {"begin": 1.0, "end": 1.5}]
    segments = segments_from_signal(data, sr, boundaries)
    assert len(segments) == 2
    assert_segment_correct(data, segments[0], sr, 0.0, 0.3)
    assert_segment_correct(data, segments[1], sr, 1.0, 1.5)

    # ArrayLike
    data, sr = sample_data
    boundaries = np.array([[0.1, 0.2], [0.8, 1.0]])
    segments = segments_from_signal(data, sr, boundaries)
    assert len(segments) == 2
    assert_segment_correct(data, segments[0], sr, 0.1, 0.2)
    assert_segment_correct(data, segments[1], sr, 0.8, 1.0)

    # invalid input
    data, sr = sample_data
    with pytest.raises(ValueError):
        segments_from_signal(data, sr, "not a valid boundary")



from biosonic.handle import boundaries_from_textgrid, audio_segments_from_textgrid

@pytest.fixture
def mock_selection_data():
    return [
        {"begin": 0.0, "end": 0.5, "label": "a"},
        {"begin": 0.5, "end": 1.0, "label": "b"},
        {"begin": 1.0, "end": 1.5, "label": ""},  # should be filtered out
        {"begin": 1.5, "end": 2.0, "label": "c"}
    ]

@pytest.fixture
def sample_audio():
    sr = 1000
    duration = 2  # seconds
    data = np.random.randn(sr * duration)
    return data, sr

@patch("biosonic.praat._read_textgrid")
def test_boundaries_from_textgrid(mock_read_textgrid, mock_selection_data):
    mock_grid = MagicMock()
    mock_grid.interval_tier_to_array.return_value = mock_selection_data
    mock_read_textgrid.return_value = mock_grid

    result = boundaries_from_textgrid("dummy_path.TextGrid", "words")

    assert isinstance(result, list)
    assert all("begin" in r and "end" in r and "label" in r for r in result)
    assert len(result) == 3  # one empty label should be filtered out
    assert result[0]["label"] == "a"
    assert result[1]["label"] == "b"
    assert result[2]["label"] == "c"


@patch("biosonic.handle.boundaries_from_textgrid")
@patch("biosonic.plot.plot_boundaries_on_spectrogram")  # mock plotting to avoid display
def test_audio_segments_from_textgrid(
    mock_plot, mock_boundaries_from_textgrid, sample_audio, mock_selection_data
):
    data, sr = sample_audio

    # Only keep labeled intervals
    labeled = [seg for seg in mock_selection_data if seg["label"]]
    mock_boundaries_from_textgrid.return_value = labeled

    segments = audio_segments_from_textgrid(
        data,
        sr,
        "dummy_path.TextGrid",
        "words"
    )

    assert isinstance(segments, list)
    assert len(segments) == len(labeled)

    for item, ref in zip(segments, labeled):
        assert isinstance(item, dict)
        assert "data" in item and "label" in item
        assert isinstance(item["data"], np.ndarray)
        assert item["label"] == ref["label"]

    mock_plot.assert_called_once()


from biosonic.handle import boundaries_from_raven, audio_segments_from_raven

@pytest.fixture
def mock_selection_df():
    return pd.DataFrame({
        "Selection": [1, 2, 3, 4],
        "Begin Time (s)": [0.0, 0.5, 1.0, 1.5],
        "End Time (s)": [0.5, 1.0, 1.5, 2.0],
        "Annotation": ["a", "b", "", "c"],
    })


@pytest.fixture
def sample_audio():
    sr = 1000
    duration = 2  # seconds
    data = np.random.randn(sr * duration)
    return data, sr


def test_boundaries_from_raven(mock_selection_df):
    with patch("biosonic.handle.pd.read_csv", return_value=mock_selection_df):
        result = boundaries_from_raven("dummy_path.txt")

    assert isinstance(result, list)
    assert all("begin" in r and "end" in r and "label" in r for r in result)
    assert len(result) == 4
    assert result[0]["label"] == "a"
    assert result[1]["label"] == "b"
    assert result[2]["label"] == ""
    assert result[3]["label"] == "c"


@patch("biosonic.handle.boundaries_from_raven")
@patch("biosonic.plot.plot_boundaries_on_spectrogram")  # mock plotting to avoid display
def test_audio_segments_from_raven(
    mock_plot, mock_boundaries, sample_audio, mock_selection_df
):
    data, sr = sample_audio

    # Only keep labeled intervals
    labeled = mock_selection_df[mock_selection_df["Annotation"] != ""]
    mock_boundaries.return_value = [
        {"begin": b, "end": e, "label": lab}
        for b, e, lab in zip(
            labeled["Begin Time (s)"], labeled["End Time (s)"], labeled["Annotation"]
        )
    ]

    segments = audio_segments_from_raven(
        data,
        sr,
        "dummy_path.txt"
    )

    assert isinstance(segments, list)
    assert len(segments) == len(mock_boundaries.return_value)

    for item, ref in zip(segments, mock_boundaries.return_value):
        assert isinstance(item, dict)
        assert "data" in item and "label" in item
        assert isinstance(item["data"], np.ndarray)
        assert item["label"] == ref["label"]

    mock_plot.assert_called_once()