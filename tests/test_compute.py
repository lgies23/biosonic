import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

def test_amplitude_envelope():
    from ..compute import amplitude_envelope
    
    # check basic signal without smoothing
    signal = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
    expected = np.array([0.73623886, 1.58098335, 2.38174929, 3, 2.38174929, 1.58098335, 0.73623886])
    assert_array_almost_equal(amplitude_envelope(signal), expected, decimal=5)
    
    # check with kernel smoothing
    expected_smoothed = np.array([0.7724074, 1.56632383, 2.32091088, 2.58783286, 2.32091088, 1.56632383, 0.7724074])
    assert_array_almost_equal(amplitude_envelope(signal, kernel_size=3), expected_smoothed, decimal=4)
    
    # check signal with leading and trailing zeros
    # TODO

    # check empty signal
    signal = np.array([], dtype=np.float64)
    expected = np.array([])
    assert_array_almost_equal(amplitude_envelope(signal), expected, decimal=5)
    
    # check signal with all zeros
    signal = np.array([0, 0, 0, 0], dtype=np.float64)
    expected = np.array([])
    assert_array_almost_equal(amplitude_envelope(signal), expected, decimal=5)
    
    # check invalid input (2D array)
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    with pytest.raises(ValueError, match="data must be a 1D array"):
        amplitude_envelope(signal)