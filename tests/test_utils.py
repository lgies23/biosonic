import unittest

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestAudioSignal(unittest.TestCase):
    def test_init(self):
        from biosonic.compute.utils import AudioSignal

        # valid initialization
        data = np.array([0.0, 1.0, -1.0, 0.5])
        sampling_rate = 44100
        signal = AudioSignal(data, sampling_rate)
        assert_array_equal(signal.data, data)
        assert signal.srate == sampling_rate
        assert signal.numchannels == 1

        # non-positive sampling rate
        with pytest.raises(ValueError, match="Sampling rate must be a positive integer."):
            AudioSignal(data, 0)

        # sampling rate not transformable to int
        with pytest.raises(TypeError, match="Sampling rate not transformable to integer"):
            AudioSignal(data, "not_an_int")

        # normalization check
        data = np.array([0, 2, -2, 1])
        signal = AudioSignal(data, 44100)
        expected_data = data / 2.0
        assert_array_equal(signal.data, expected_data)

        # 2 channel data
        data = np.array([[0.0, 1.0], [-1.0, 0.5]])
        signal = AudioSignal(data, 44100)
        assert signal.numchannels == 2

        # 3 channel data
        data = np.array([[0.0, 1.0], [-1.0, 0.5], [0.3, -0.3]])
        signal = AudioSignal(data, 44100)
        assert signal.numchannels == 3

        # empty data
        data = np.array([])
        with pytest.raises(AssertionError, match="'data' must not be empty"):
            AudioSignal(data, 44100)

        # all zeros data
        data = np.array([0.0, 0.0, 0.0])
        with pytest.raises(AssertionError, match="'data' contains no nonzero values"):
            AudioSignal(data, 44100)

        # data is not array-like
        data = "not an array"
        with pytest.raises(AssertionError, match="'data' must be array-like"):
            AudioSignal(data, 44100)


def test_exclude_trailing_and_leading_zeros():
    from biosonic.compute.utils import exclude_trailing_and_leading_zeros

    # check valid case
    arr = np.array([0, 0, 1, 2, 3, 0])
    expected = np.array([1, 2, 3])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # ceck array with no zeros
    arr = np.array([1, 2, 3])
    expected = np.array([1, 2, 3])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # check array with only zeros
    arr = np.array([0, 0, 0, 0])
    expected = np.array([])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # check array with zeros in the middle (should not be removed)
    arr = np.array([0, 1, 0, 2, 0, 3, 0])
    expected = np.array([1, 0, 2, 0, 3])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # check array with single non-zero element
    arr = np.array([0, 0, 5, 0, 0])
    expected = np.array([5])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # check empty array
    arr = np.array([])
    expected = np.array([])
    assert_array_equal(exclude_trailing_and_leading_zeros(arr), expected)

    # check invalid input (2D array)
    arr = np.array([[0, 1, 2], [0, 0, 3]])
    with pytest.raises(ValueError, match="Input array must be 1D."):
        exclude_trailing_and_leading_zeros(arr)


def test_transform_spectrogram_for_nn():
    from biosonic.compute.utils import transform_spectrogram_for_nn

    # helper spectrogram
    def make_spec(h, w, value_fn=np.random.rand):
        spec = value_fn(h, w) * 255
        t = np.linspace(0, 1, w)
        f = np.linspace(0, 8000, h)
        return spec, t, f

    # normalization
    spec, t, f = make_spec(32, 32)
    transformed = transform_spectrogram_for_nn((spec, t, f), add_channel=False)
    assert np.isclose(transformed.max(), 1.0)
    assert np.isclose(transformed.min(), 0.0)
    assert transformed.shape == (32, 32)

    # type casting
    spec = (np.random.rand(64, 64) * 255).astype("uint8")
    t = np.linspace(0, 1, 64)
    f = np.linspace(0, 8000, 64)

    transformed = transform_spectrogram_for_nn(
        (spec, t, f),
        values_type="float64",
        add_channel=False
    )
    assert transformed.dtype == np.float64

    spec = spec.astype("float32")
    transformed = transform_spectrogram_for_nn(
        (spec, t, f),
        values_type="float64",
        add_channel=False
    )
    assert transformed.dtype == np.float64

    # channel addition (first)
    spec, t, f = make_spec(32, 32)
    transformed = transform_spectrogram_for_nn(
        (spec, t, f),
        add_channel=True,
        data_format="channels_first"
    )
    assert transformed.shape == (1, 32, 32)

    # channel addition (last)
    transformed = transform_spectrogram_for_nn(
        (spec, t, f),
        add_channel=True,
        data_format="channels_last"
    )
    assert transformed.shape == (32, 32, 1)

    # test no channel addition
    transformed = transform_spectrogram_for_nn(
        (spec, t, f),
        add_channel=False
    )
    assert transformed.shape == (32, 32)

    # zero-information input (all zeros)
    spec = np.zeros((16, 16))
    t = np.linspace(0, 1, 16)
    f = np.linspace(0, 8000, 16)

    with pytest.warns(RuntimeWarning, match="Spectrogram contains no information"):
        transform_spectrogram_for_nn((spec, t, f))

    # constant-value input
    spec = np.full((10, 10), fill_value=5.0)
    t = np.linspace(0, 1, 10)
    f = np.linspace(0, 8000, 10)

    with pytest.warns(RuntimeWarning, match="Spectrogram contains no information"):
        transform_spectrogram_for_nn((spec, t, f))


def test_shannon_enropy():
    from biosonic.compute.utils import shannon_entropy

    # uniform distribution
    dist = np.array([0.25, 0.25, 0.25, 0.25])
    entropy, max_val = shannon_entropy(dist, unit="bits", norm=False)
    expected = np.log2(4)
    assert entropy == 0.0
    assert np.isclose(max_val, expected)

    dist = np.array([0.25, 0.25, 0.25, 0.25])
    entropy, max_val = shannon_entropy(dist, unit="bits", norm=True)
    assert entropy == 0.0
    assert max_val == 1.0

    dist = np.array([1/3, 1/3, 1/3])
    entropy, max_val = shannon_entropy(dist, unit="nat", norm=True)
    assert entropy == 0.0
    assert max_val == 1.0

    # skewed distribution
    dist = np.array([0.9, 0.1])
    entropy, max_val = shannon_entropy(dist, unit="bits", norm=False)
    assert 0.0 < entropy < max_val

    dist = np.array([0.6, 0.4])
    entropy, max_val = shannon_entropy(dist, unit="dits", norm=False)
    expected = -np.sum(dist * np.log10(dist))
    assert np.isclose(entropy, expected)
    assert np.isclose(max_val, np.log10(2))

    # invalid unit
    dist = np.array([0.5, 0.5])
    try:
        _ = shannon_entropy(dist, unit="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Invalid unit" in str(e)

    # output type
    dist = np.array([0.7, 0.3])
    entropy, max_val = shannon_entropy(dist)
    assert isinstance(entropy, float)
    assert isinstance(max_val, float)


def test_hz_to_mel():
    from biosonic.compute.utils import hz_to_mel

    # scalar oshaughnessy
    freq = 1000.0
    mel = hz_to_mel(freq, after="oshaughnessy")
    expected = 2595.0 * np.log(1 + freq / 700.0)
    assert np.isclose(mel, expected)

    # array oshaughnessy
    freq = np.array([0.0, 500.0, 1000.0])
    mel = hz_to_mel(freq, after="oshaughnessy")
    expected = 2595.0 * np.log(1 + freq / 700.0)
    np.testing.assert_allclose(mel, expected)

    # fant defaults
    freq = 1000.0
    mel = hz_to_mel(freq, after="fant")
    expected = 1000.0 * np.log(1 + freq / 1000.0)
    assert np.isclose(mel, expected)

    # corner frequency
    freq = 1000.0
    mel = hz_to_mel(freq, corner_frequency=500.0)
    expected = 500.0 * np.log(1 + freq / 500.0)
    assert np.isclose(mel, expected)

    # umesh
    freq = 1000.0
    mel = hz_to_mel(freq, after="umesh")
    expected = freq / (0.0004 * freq + 0.603)
    assert np.isclose(mel, expected)

    # custom params
    freq = 1000.0
    mel = hz_to_mel(freq, a=3000.0, b=800.0, after="oshaughnessy")
    expected = 3000.0 * np.log(1 + freq / 800.0)
    assert np.isclose(mel, expected)

    # koenig
    try:
        hz_to_mel(1000.0, after="koenig")
        assert False, "Expected NotImplementedError"
    except NotImplementedError as e:
        assert "koenig" in str(e)

    # invalid string
    try:
        hz_to_mel(1000.0, after="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unknown Mel scale method" in str(e)

    # output type
    scalar_result = hz_to_mel(500.0)
    array_result = hz_to_mel(np.array([500.0, 1000.0]))
    assert isinstance(scalar_result, float) or np.isscalar(scalar_result)
    assert isinstance(array_result, np.ndarray)
