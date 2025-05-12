import numpy as np
import pytest
from numpy.testing import assert_array_equal

def test_check_sr_format():
    from biosonic.compute.utils import check_sr_format

    # check valid case
    sr = check_sr_format(44100)
    assert np.isclose(sr, 44100) and isinstance(sr, int)
    sr = check_sr_format(44100.0)
    assert np.isclose(sr, 44100) and isinstance(sr, int)
    sr = check_sr_format("44100")
    assert np.isclose(sr, 44100) and isinstance(sr, int)

    with pytest.raises(TypeError, match="Sample rate not transformable to integer."):
        check_sr_format("string")

    # check greater zero cases
    with pytest.raises(ValueError, match="Sample rate must be greater than zero."):
        check_sr_format(0)

    with pytest.raises(ValueError, match="Sample rate must be greater than zero."):
        check_sr_format(-16000)


def test_check_signal_format():
    from biosonic.compute.utils import check_signal_format

    # check valid case
    signal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    returned_signal = check_signal_format(signal)
    assert len(returned_signal) == len(signal)

    # check integer array
    signal = np.array([1, 2, 3], dtype=np.int32)
    returned_signal = check_signal_format(signal)
    assert len(returned_signal) == len(signal)

    # check non-numpy array
    signal = [0.1, 0.2, 0.3]
    returned_signal = check_signal_format(signal)
    assert len(returned_signal) == len(signal)

    # check string array
    signal = np.array(["1", "2", "3"])
    returned_signal = check_signal_format(signal)
    assert len(returned_signal) == len(signal)

    # check wrong dimensions
    with pytest.raises(ValueError, match="Signal must be a 1D array."):
        check_signal_format(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))


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

    # test normalization
    spectrogram = np.array(np.random.rand(32, 32) * 255, dtype='float32')
    transformed = transform_spectrogram_for_nn(spectrogram, add_channel=False)
    assert np.isclose(transformed.max(), 1.0)
    assert np.isclose(transformed.min(), 0.0)
    assert transformed.shape == (32, 32)

    # test type casting   
    spectrogram = np.array(np.random.rand(64, 64) * 255, dtype = 'uint8')
    transformed = transform_spectrogram_for_nn(spectrogram, values_type='float64', add_channel=False)
    assert transformed.dtype == np.float64
    
    spectrogram = np.array(np.random.rand(64, 64) * 255, dtype = 'float32')
    transformed = transform_spectrogram_for_nn(spectrogram, values_type='float64', add_channel=False)
    assert transformed.dtype == np.float64

    # test channel addition (first)
    spectrogram = np.array(np.random.rand(32, 32) * 255)
    transformed = transform_spectrogram_for_nn(spectrogram, add_channel=True, data_format='channels_first')
    assert transformed.shape == (1, 32, 32)

    # test channel addition (last)
    spectrogram = np.array(np.random.rand(32, 32) * 255)
    transformed = transform_spectrogram_for_nn(spectrogram, add_channel=True, data_format='channels_last')
    assert transformed.shape == (32, 32, 1)

    # test no channel addition
    spectrogram = np.array(np.random.rand(32, 32) * 255)
    transformed = transform_spectrogram_for_nn(spectrogram, add_channel=False)
    assert transformed.shape == (32, 32)

    # test zero (information) input
    spectrogram = np.zeros((16, 16))
    with pytest.warns(RuntimeWarning, match="Spectrogram contains no information"):
        transformed = transform_spectrogram_for_nn(spectrogram)

    spectrogram = np.full((10, 10), fill_value=5)
    with pytest.warns(RuntimeWarning, match="Spectrogram contains no information"):
        transformed = transform_spectrogram_for_nn(spectrogram)