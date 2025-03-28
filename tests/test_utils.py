import numpy as np
import pytest
from numpy.testing import assert_array_equal

def test_check_sr_format():
    from ..utils import check_sr_format

    # check valid case
    check_sr_format(44100)

    # check non integer cases
    with pytest.raises(TypeError, match="sr must be of type integer."):
        check_sr_format(44100.0)

    with pytest.raises(TypeError, match="sr must be of type integer."):
        check_sr_format("44100")

    # check greater zero cases
    with pytest.raises(ValueError, match="sr must be greater than zero."):
        check_sr_format(0)

    with pytest.raises(ValueError, match="sr must be greater than zero."):
        check_sr_format(-16000)


def test_check_signal_format():
    from ..utils import check_signal_format

    # check valid case
    signal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    check_signal_format(signal)  # Should not raise

    # check integer array
    check_signal_format(np.array([1, 2, 3], dtype=np.int32))

    # check non-numpy array
    check_signal_format([0.1, 0.2, 0.3])

    # check string array
    check_signal_format(np.array(["1", "2", "3"], dtype=np.int32))

    # check wrong dimensions
    with pytest.raises(ValueError, match="data must be a 1D array."):
        check_signal_format(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))


def test_exclude_trailing_and_leading_zeros():
    from ..utils import exclude_trailing_and_leading_zeros

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
