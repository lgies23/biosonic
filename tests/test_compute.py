import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

def test_amplitude_envelope():
    from biosonic.compute.temporal import amplitude_envelope
    
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
    with pytest.warns(RuntimeWarning, match="Input signal is empty; returning an empty array."):
        assert len(amplitude_envelope(np.array([]))) == 0
    
    # check signal with all zeros
    signal = np.array([0, 0, 0, 0], dtype=np.float64)
    expected = np.array([])
    assert_array_almost_equal(amplitude_envelope(signal), expected, decimal=5)
    
    # check invalid input (2D array)
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    with pytest.raises(ValueError, match="Signal must be a 1D array"):
        amplitude_envelope(signal)

def test_duration():
    from biosonic.compute.temporal import duration

    # check basic duration calculation without silence exclusion
    signal = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    sr = 10  # Sample rate
    expected = 0.5  # 5 samples / 10 samples per second
    assert duration(signal, sr, exclude_surrounding_silences=False) == expected
    
    # check duration calculation with leading and trailing zeros (exclude_surrounding_silences=True)
    signal = np.array([0, 0, 1, 2, 3, 4, 5, 0, 0], dtype=np.float64)
    expected = 0.5  # only non-zero part -> 5 samples
    assert duration(signal, sr, exclude_surrounding_silences=True) == expected
    
    # check duration with leading and trailing zeros but exclude_surrounding_silences=False
    expected = 0.9  # 9 samples / 10 samples per second
    assert duration(signal, sr, exclude_surrounding_silences=False) == expected
    
    # check empty signal
    signal = np.array([], dtype=np.float64)
    expected = 0.0
    assert duration(signal, sr, exclude_surrounding_silences=True) == expected
    assert duration(signal, sr, exclude_surrounding_silences=False) == expected
    
    # check all-zero signal
    signal = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    expected_trimmed = 0.0
    expected_full = 0.5  # 5 samples / 10 samples per second
    assert duration(signal, sr, exclude_surrounding_silences=True) == expected_trimmed
    assert duration(signal, sr, exclude_surrounding_silences=False) == expected_full
    
    # check invalid sample rate
    with pytest.raises(ValueError, match="Sample rate must be greater than zero"):
        duration(signal, 0)
    with pytest.raises(ValueError, match="Sample rate must be greater than zero"):
        duration(signal, -10)
    
    # check invalid input (2D array)
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    with pytest.raises(ValueError, match="Signal must be a 1D array"):
        duration(signal, sr)


def test_temporal_quartiles():
    from biosonic.compute.temporal import temporal_quartiles
    
    # check basic case
    signal = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
    sr = 10
    q1, median, q3 = temporal_quartiles(signal, sr)
    assert 0 <= q1 < median < q3 <= len(signal) / sr
    # TODO check actual values
    
    # check with a longer signal
    signal = np.array([0] * 10 + [1] * 80 + [0] * 10, dtype=np.float64)  # 100 samples
    sr = 20
    q1, median, q3 = temporal_quartiles(signal, sr)
    assert 0 <= q1 < median < q3 <= 5
    # TODO check actual values
    
    # check empty signal
    signal = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="Input is empty"):
        temporal_quartiles(signal, sr)
    
    # check all-zero signal
    signal = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="Signal contains no nonzero values"):
        temporal_quartiles(signal, sr)
    
    # check invalid sample rate
    with pytest.raises(ValueError, match="Sample rate must be greater than zero"):
        temporal_quartiles(signal, 0)
    with pytest.raises(ValueError, match="Sample rate must be greater than zero"):
        temporal_quartiles(signal, -10)
    
    # check 2D input (invalid case)
    signal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    with pytest.raises(ValueError, match="Signal must be a 1D array"):
        temporal_quartiles(signal, sr)
    
def test_temporal_sd():
    pass

def test_temporal_skew():
    pass

def test_temporal_kurtosis():
    pass

def test_spectrum():
    import numpy as np
    import pytest
    from biosonic.compute.spectral import spectrum

    # basic amplitude spectrum
    signal = np.array([0, 1, 0, -1], dtype=np.float64)
    spec = spectrum(signal, mode='amplitude')
    assert isinstance(spec, np.ndarray)
    assert spec.shape == signal.shape
    assert np.all(spec >= 0)

    # basic power spectrum
    power_spec = spectrum(signal, mode='power')
    assert np.allclose(power_spec, np.abs(np.fft.fft(signal))**2)

    # check arbitrary exponent
    spec3 = spectrum(signal, mode=3)
    assert np.allclose(spec3, np.abs(np.fft.fft(signal))**3)

    # check that amplitude and mode=1 give same result
    spec1 = spectrum(signal, mode=1)
    spec_amp = spectrum(signal, mode='amplitude')
    assert np.allclose(spec1, spec_amp)

    # check that invalid string raises error
    with pytest.raises(ValueError, match="Invalid string mode"):
        spectrum(signal, mode='invalid')

    # check that invalid type raises error
    with pytest.raises(TypeError, match="must be a string, int or float"):
        spectrum(signal, mode=(1,2))

    # check empty input
    empty_signal = np.array([], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="Input signal is empty"):
        spec_empty = spectrum(empty_signal, mode='amplitude')
        assert spec_empty.size == 0
        assert isinstance(spec_empty, np.ndarray)


    # check default = "amplitude"
    signal = np.array([0, 1, 0, -1], dtype=np.float64)
    spec = spectrum(signal)
    assert isinstance(spec, np.ndarray)
    assert spec.shape == signal.shape
    assert np.all(spec >= 0)
    assert np.allclose(spec, np.abs(np.fft.fft(signal)))


def test_peak_frequency():
    from biosonic.compute.spectral import peak_frequency
    sampling_rate = 1000

    # check single sine wave
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50.0 * t)
    freq_est = peak_frequency(signal, sampling_rate)
    print(freq_est)
    assert np.isclose(freq_est, 50)

    # check mixed signal (strongest component is 120 Hz)
    sampling_rate = 2000
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 120.0 * t) + 0.5 * np.sin(2 * np.pi * 300.0 * t)
    freq_est = peak_frequency(signal, sampling_rate)
    assert np.isclose(freq_est, 120)

    # check sine wave with noise
    np.random.seed(42)
    sampling_rate = 1000
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 80.0 * t) + 0.3 * np.random.randn(len(t))
    freq_est = peak_frequency(signal, sampling_rate)
    assert np.isclose(freq_est, 80)

    # check empty signal
    with pytest.warns(RuntimeWarning, match="Input signal is empty; returning NaN for peak frequency."):
        assert peak_frequency(np.array([], dtype=np.float64), sampling_rate) is None


    # check invalid shape (not 1D)
    with pytest.raises(ValueError, match="Signal must be a 1D array"):
        peak_frequency(np.array([[1, 2, 3]]), sampling_rate)

    # check DC signal
    signal = np.ones(1024, dtype=np.float64)
    freq_est = peak_frequency(signal, sampling_rate)
    assert freq_est == 0.0


def test_spectrogram():
    from biosonic.compute.spectral import spectrogram

    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    sine_wave = np.sin(2 * np.pi * freq * t)

    Sx, times, freqs = spectrogram(
        data=sine_wave,
        sr=sr,
        window="hann",
        window_length=512,
        overlap=0.5
    )

    assert isinstance(Sx, np.ndarray)
    assert np.iscomplexobj(Sx)
    assert isinstance(times, np.ndarray) and times.ndim == 1, "Times must be 1D array"
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1, "Frequencies must be 1D array"
    assert Sx.shape[0] == len(freqs)
    assert Sx.shape[1] == len(times)




def test_transform_spectrogram_for_nn():
    from biosonic.compute.spectral import transform_spectrogram_for_nn
    from typing import Literal

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