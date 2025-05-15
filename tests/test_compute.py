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
    t = np.linspace(0, 1.0, 100, endpoint=False)
    signal = np.sin(2 * np.pi * 50.0 * t)
    freqs, spec = spectrum(signal, mode='amplitude')
    assert freqs is None
    assert isinstance(spec, np.ndarray)
    assert len(spec) == (len(signal) // 2 +1)
    assert np.all(spec >= 0)

    # basic power spectrum
    _, power_spec = spectrum(signal, mode='power')
    assert np.allclose(power_spec, np.abs(np.fft.rfft(signal))**2)

    # check arbitrary exponent
    _, spec = spectrum(signal, mode=3)
    assert np.allclose(spec, np.abs(np.fft.rfft(signal))**3)

    # check that amplitude and mode=1 give same result
    _, spec = spectrum(signal, mode=1)
    _, spec_amp = spectrum(signal, mode='amplitude')
    assert np.allclose(spec, spec_amp)

    # check that invalid string raises error
    with pytest.raises(ValueError, match="Invalid string mode"):
        spectrum(signal, mode='invalid')

    # check that invalid type raises error
    with pytest.raises(TypeError, match="must be a string, int or float"):
        spectrum(signal, mode=(1,2))

    # check empty input
    empty_signal = np.array([], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="Input signal is empty"):
        freqs_empty, spec_empty = spectrum(empty_signal, mode='amplitude')
        assert spec_empty.size == 0
        assert isinstance(spec_empty, np.ndarray)
        assert freqs_empty == None

    # check default = "amplitude"
    signal = np.array([0, 1, 0, -1], dtype=np.float64)
    freqs, spec = spectrum(signal, sr=4)
    assert isinstance(spec, np.ndarray)
    assert len(spec) == (len(signal) // 2 + 1)
    assert np.all(spec >= 0)
    assert np.allclose(spec, np.abs(np.fft.rfft(signal)))
    assert freqs.shape == spec.shape


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
    with pytest.raises(ValueError, match="Input signal is empty; could not determine peak frequency."):
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

def test_spectral_quartiles(monkeypatch):
    from biosonic.compute.spectral import spectral_quartiles
    
    # basic sinosoid
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 20 * t)
    q1, q2, q3 = spectral_quartiles(x, sr)
    assert q1 <= q2 <= q3
    assert np.isclose(q2, 20, atol=5)

    # empty input
    with pytest.raises(ValueError, match="Input is empty"):
        spectral_quartiles([], 1000)

    # zero signal
    with pytest.raises(ValueError, match="Signal contains no nonzero values"):
        spectral_quartiles(np.zeros(1000), 1000)

    # mismatched frequencies
    def fake_spectrum(*args, **kwargs):
        return np.array([0, 1, 2]), np.array([1.0])  # Mismatched lengths
    monkeypatch.setattr("biosonic.compute.spectral.spectrum", fake_spectrum)

    with pytest.raises(ValueError, match="Freuency bins don't match envelope"):
        spectral_quartiles(np.ones(100), 1000)

def test_flatness():
    from biosonic.compute.spectral import flatness

    # basic sinusoid
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 100 * t)
    f = flatness(x)
    assert 0 <= f < 0.5, f"Expected low flatness for a tone, got {f}"

    # white noise
    np.random.seed(0)
    noise = np.random.randn(sr)
    f_noise = flatness(noise)
    assert 0.5 < f_noise <= 1.0, f"Expected high flatness for noise, got {f_noise}"

    # all zeros
    with pytest.raises(ValueError, match="Input signal contained only zero values"):
        flatness(np.zeros(1000))

    # empty input
    with pytest.raises(ValueError, match="Input signal contained only zero values"):
        flatness([])
        
    # check output type
    y = flatness(np.random.randn(1024))
    assert isinstance(y, float) or isinstance(y, np.floating), f"Expected float output, got {type(y)}"


def test_bandwidth():
    #TODO
    from biosonic.compute.spectral import bandwidth

    # basic case
    # data = [1, 2, 3, 4, 5]
    # expected_std = np.std(np.abs(fft.rfft(data))**2, ddof=0)
    # assert np.isclose(standard_deviation(data, sr=1), expected_std)

    # constant signal
    data = [3, 3, 3, 3]
    expected_std = 0.0
    with pytest.warns(RuntimeWarning, match="Bandwidth of signal is 0, returning NaN for skewness and kurtosis"):
        assert np.isclose(bandwidth(data, sr=1), expected_std)

    # single sample
    data = [7]
    expected_std = 0.0
    with pytest.warns(RuntimeWarning, match="Bandwidth of signal is 0, returning NaN for skewness and kurtosis"):
        assert np.isclose(bandwidth(data, sr=1), expected_std)

    # # with numpy array input
    # t = np.linspace(0, 1, 1000, endpoint=False)
    # data = np.sin(2 * np.pi * 100 * t)
    # expected_std = 63.61767414
    # assert np.isclose(bandwidth(data, sr=1000), expected_std)

    # return type
    assert isinstance(bandwidth([1, 2, 3], sr=1), float)


def test_centroid():
    from biosonic.compute.spectral import centroid

    t = np.linspace(0, 1, 1000, endpoint=False)
    data = np.sin(2 * np.pi * 100 * t)
    expected = 100
    assert np.isclose(centroid(data, sr=1000), expected)

    t = np.linspace(0, 1, 1000, endpoint=False)
    data = np.sin(2 * np.pi * 40 * t)
    expected = 40
    assert np.isclose(centroid(data, sr=1000), expected)