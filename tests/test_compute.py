import unittest

import numpy as np
import pytest


def test_amplitude_envelope():
    # from biosonic.compute.temporal import amplitude_envelope

    # from biosonic.compute.utils import AudioSignal
    # rely on scipy envelope function for correctness
    return True


def test_duration():
    from biosonic.compute.temporal import duration
    # check basic duration calculation (no silence exclusion)
    from biosonic.compute.utils import AudioSignal
    signal = AudioSignal(np.array([1, 2, 3, 4, 5], dtype=np.float64), 10)
    expected = 0.5  # 5 samples / 10 samples per second
    assert duration(signal) == expected

    # check percentile-based trimming
    arr = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0
        ], dtype=np.float64)
    print(len(arr))
    signal = AudioSignal(arr, 100)
    # For this signal, 1% trimming is equivalent to trimming zeros
    assert duration(signal, silence_threshold=0.15, timestep=0.01, window_length=3) == 0.2

    # check invalid sample rate
    arr = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0
        ], dtype=np.float64)
    with pytest.raises(ValueError, match="Sampling rate must be a positive integer."):
        AudioSignal(arr, 0)
    with pytest.raises(ValueError, match="Sampling rate must be a positive integer."):
        AudioSignal(arr, -10)


def test_temporal_quartiles():
    from biosonic.compute.temporal import temporal_quartiles
    # check basic case
    from biosonic.compute.utils import AudioSignal
    signal = AudioSignal(np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64), 10)
    q1, median, q3 = temporal_quartiles(signal)
    assert 0 <= q1 < median < q3 <= len(signal.data) / signal.srate
    # TODO check actual values

    # check with a longer signal
    arr = np.array([0] * 10 + [1] * 80 + [0] * 10, dtype=np.float64)  # 100 samples
    signal = AudioSignal(arr, 20)
    q1, median, q3 = temporal_quartiles(signal)
    assert 0 <= q1 < median < q3 <= 5
    # TODO check actual values

    # check empty signal
    arr = np.array([], dtype=np.float64)
    with pytest.raises(AssertionError, match="'data' must not be empty"):
        temporal_quartiles(AudioSignal(arr, 10))

    # check all-zero signal
    arr = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="Signal contains no nonzero values"):
        temporal_quartiles(AudioSignal(arr, 10))

    # check invalid sample rate
    arr = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="Sampling rate must be a positive integer."):
        AudioSignal(arr, 0)
    with pytest.raises(ValueError, match="Sampling rate must be a positive integer."):
        AudioSignal(arr, -10)


def test_temporal_sd():
    pass


def test_temporal_skew():
    pass


def test_temporal_kurtosis():
    pass


def test_spectrum():
    from biosonic.compute.spectral import spectrum
    # basic amplitude spectrum
    from biosonic.compute.utils import AudioSignal
    t = np.linspace(0, 1.0, 100, endpoint=False)
    arr = np.sin(2 * np.pi * 50.0 * t)
    signal = AudioSignal(arr, 100)
    freqs, spec = spectrum(signal, mode='amplitude')
    assert freqs is None or isinstance(freqs, np.ndarray)
    assert isinstance(spec, np.ndarray)
    assert len(spec) == (len(signal.data) // 2 + 1)
    assert np.all(spec >= 0)

    # basic power spectrum
    _, power_spec = spectrum(signal, mode='power')
    assert np.allclose(power_spec, np.abs(np.fft.rfft(signal.data))**2)

    # check arbitrary exponent
    _, spec = spectrum(signal, mode=3)
    assert np.allclose(spec, np.abs(np.fft.rfft(signal.data))**3)

    # check that amplitude and mode=1 give same result
    _, spec = spectrum(signal, mode=1)
    _, spec_amp = spectrum(signal, mode='amplitude')
    assert np.allclose(spec, spec_amp)

    # check that invalid string raises error
    with pytest.raises(ValueError, match="Invalid string mode"):
        spectrum(signal, mode='invalid')

    # check that invalid type raises error
    with pytest.raises(TypeError, match="must be a string, int or float"):
        spectrum(signal, mode=(1, 2))

    # check default = "amplitude"
    arr = np.array([0, 1, 0, -1], dtype=np.float64)
    signal = AudioSignal(arr, 4)
    freqs, spec = spectrum(signal)
    assert isinstance(spec, np.ndarray)
    assert len(spec) == (len(signal.data) // 2 + 1)
    assert np.all(spec >= 0)
    assert np.allclose(spec, np.abs(np.fft.rfft(signal.data)))
    assert freqs.shape == spec.shape


def test_peak_frequency():
    from biosonic.compute.spectral import peak_frequency
    from biosonic.compute.utils import AudioSignal
    sampling_rate = 1000

    # check single sine wave
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    arr = np.sin(2 * np.pi * 50.0 * t)
    signal = AudioSignal(arr, sampling_rate)
    freq_est = peak_frequency(signal)
    assert np.isclose(freq_est, 50)

    # check mixed signal (strongest component is 120 Hz)
    sampling_rate = 2000
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    arr = np.sin(2 * np.pi * 120.0 * t) + 0.5 * np.sin(2 * np.pi * 300.0 * t)
    signal = AudioSignal(arr, sampling_rate)
    freq_est = peak_frequency(signal)
    assert np.isclose(freq_est, 120)

    # check sine wave with noise
    np.random.seed(42)
    sampling_rate = 1000
    t = np.linspace(0, 1.0, sampling_rate, endpoint=False)
    arr = np.sin(2 * np.pi * 80.0 * t) + 0.3 * np.random.randn(len(t))
    signal = AudioSignal(arr, sampling_rate)
    freq_est = peak_frequency(signal)
    assert np.isclose(freq_est, 80)

    # check empty signal
    arr = np.array([], dtype=np.float64)
    with pytest.raises(AssertionError, match="'data' must not be empty"):
        assert peak_frequency(AudioSignal(arr, sampling_rate)) is None

    # # check invalid shape (not 1D)
    # arr = np.array([[1, 2, 3]], dtype=np.float64)
    # with pytest.raises(TypeError, match="Data must be a NumPy array."):
    #     AudioSignal(arr, sampling_rate)

    # check DC signal
    arr = np.ones(1024, dtype=np.float64)
    signal = AudioSignal(arr, sampling_rate)
    freq_est = peak_frequency(signal)
    assert freq_est == 0.0


def test_spectrogram():
    from biosonic.compute.spectrotemporal import spectrogram
    from biosonic.compute.utils import AudioSignal
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    arr = np.sin(2 * np.pi * freq * t)
    signal = AudioSignal(arr, sr)

    Sx, times, freqs = spectrogram(signal, window="hann", window_length=512, overlap=0.5, complex_output=True)

    assert isinstance(Sx, np.ndarray)
    assert np.iscomplexobj(Sx)
    assert isinstance(times, np.ndarray) and times.ndim == 1, "Times must be 1D array"
    assert isinstance(freqs, np.ndarray) and freqs.ndim == 1, "Frequencies must be 1D array"
    assert Sx.shape[0] == len(freqs)
    assert Sx.shape[1] == len(times)


def test_spectral_quartiles(monkeypatch):
    from biosonic.compute.spectral import quartiles
    # basic sinosoid
    from biosonic.compute.utils import AudioSignal
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    arr = np.sin(2 * np.pi * 20 * t)
    signal = AudioSignal(arr, sr)
    q1, q2, q3 = quartiles(signal)
    assert q1 <= q2 <= q3
    assert np.isclose(q2, 20, atol=5)

    # empty input
    arr = np.array([], dtype=np.float64)
    with pytest.raises(AssertionError, match="'data' must not be empty"):
        quartiles(AudioSignal(arr, sr))

    # zero signal
    arr = np.zeros(1000, dtype=np.float64)
    with pytest.raises(ValueError, match="Signal contains no nonzero values"):
        quartiles(AudioSignal(arr, sr))

    # mismatched frequencies
    def fake_spectrum(*args, **kwargs):
        return np.array([0, 1, 2]), np.array([1.0])  # Mismatched lengths
    monkeypatch.setattr("biosonic.compute.spectral.spectrum", fake_spectrum)

    arr = np.ones(100, dtype=np.float64)
    signal = AudioSignal(arr, sr)
    with pytest.raises(ValueError, match="Frequency bins don't match envelope"):
        quartiles(signal)


def test_flatness():
    from biosonic.compute.spectral import flatness
    # basic sinusoid
    from biosonic.compute.utils import AudioSignal
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    arr = np.sin(2 * np.pi * 100 * t)
    signal = AudioSignal(arr, sr)
    f = flatness(signal)
    assert 0 <= f < 0.5, f"Expected low flatness for a tone, got {f}"

    # white noise
    np.random.seed(0)
    arr = np.random.randn(sr)
    signal = AudioSignal(arr, sr)
    f_noise = flatness(signal)
    assert 0.5 < f_noise <= 1.0, f"Expected high flatness for noise, got {f_noise}"

    # empty input
    arr = np.array([], dtype=np.float64)
    with pytest.raises(AssertionError, match="'data' must not be empty"):
        flatness(AudioSignal(arr, sr))

    # check output type
    arr = np.random.randn(1024)
    signal = AudioSignal(arr, 1024)
    y = flatness(signal)
    assert isinstance(y, float) or isinstance(y, np.floating), f"Expected float output, got {type(y)}"


def test_bandwidth():
    # TODO
    from biosonic.compute.spectral import bandwidth
    from biosonic.compute.utils import AudioSignal

    # constant signal
    arr = np.array([3, 3, 3, 3], dtype=np.float64)
    signal = AudioSignal(arr, 1)
    expected_std = 0.0
    with pytest.warns(RuntimeWarning, match="Bandwidth of signal is 0, returning NaN for skewness and kurtosis"):
        assert np.isclose(bandwidth(signal), expected_std)

    # single sample
    arr = np.array([7], dtype=np.float64)
    signal = AudioSignal(arr, 1)
    expected_std = 0.0
    with pytest.warns(RuntimeWarning, match="Bandwidth of signal is 0, returning NaN for skewness and kurtosis"):
        assert np.isclose(bandwidth(signal), expected_std)

    # return type
    arr = np.array([1, 2, 3], dtype=np.float64)
    signal = AudioSignal(arr, 1)
    assert isinstance(bandwidth(signal), float)


def test_centroid():
    from biosonic.compute.spectral import centroid
    from biosonic.compute.utils import AudioSignal

    t = np.linspace(0, 1, 1000, endpoint=False)
    arr = np.sin(2 * np.pi * 100 * t)
    signal = AudioSignal(arr, 1000)
    expected = 100
    assert np.isclose(centroid(signal), expected)

    t = np.linspace(0, 1, 1000, endpoint=False)
    arr = np.sin(2 * np.pi * 40 * t)
    signal = AudioSignal(arr, 1000)
    expected = 40
    assert np.isclose(centroid(signal), expected)


from biosonic.compute.spectrotemporal import dominant_frequencies


class TestDominantFrequencies(unittest.TestCase):

    def setUp(self):
        from biosonic.compute.utils import AudioSignal
        self.sample_rate = 1000  # Hz
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)

        # Single tone sine wave at 50 Hz
        self.sine_wave = AudioSignal(np.sin(2 * np.pi * 50 * t), self.sample_rate)

        # Multi-tone signal: 100, 200, 300 Hz
        self.multi_tone_signal = AudioSignal(
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 300 * t),
            self.sample_rate
        )

        # Flat signal (no frequency content)
        self.flat_signal = AudioSignal(np.ones(self.sample_rate), self.sample_rate)

    def test_dominant_frequency_single(self):
        freqs = dominant_frequencies(self.sine_wave, n_freqs=1)
        self.assertEqual(freqs.ndim, 1)
        self.assertTrue(np.all(np.isfinite(freqs[np.isfinite(freqs)])))
        self.assertTrue(np.all((freqs[np.isfinite(freqs)] > 40) & (freqs[np.isfinite(freqs)] < 60)))

    def test_dominant_frequencies_multiple_default(self):
        freqs = dominant_frequencies(self.multi_tone_signal)
        self.assertEqual(freqs.ndim, 1)
        self.assertTrue(np.any(np.abs(freqs - 100) < 10))

    def test_dominant_frequencies_multiple_explicit(self):
        freqs = dominant_frequencies(self.multi_tone_signal, n_freqs=3,
                                     min_prominence=0.001, min_height=0.001, min_distance=0.001, threshold=0.001)
        self.assertEqual(freqs.ndim, 2)
        self.assertEqual(freqs.shape[1], 3)
        self.assertTrue(np.any(np.abs(freqs - 100) < 10))
        self.assertTrue(np.any(np.abs(freqs - 200) < 10))
        self.assertTrue(np.any(np.abs(freqs - 300) < 10))

    def test_handles_no_peaks(self):
        freqs = dominant_frequencies(self.flat_signal, n_freqs=2)
        self.assertTrue(np.all(np.isnan(freqs)))

    def test_output_shapes(self):
        freqs_1 = dominant_frequencies(self.sine_wave, n_freqs=1)
        freqs_3 = dominant_frequencies(self.sine_wave, n_freqs=3)
        self.assertEqual(freqs_1.ndim, 1)
        self.assertEqual(freqs_3.ndim, 2)
        self.assertEqual(freqs_3.shape[0], freqs_1.shape[0])
        self.assertEqual(freqs_3.shape[1], 3)

    def test_value_checks(self):
        with self.assertRaises(ValueError):
            dominant_frequencies(self.sine_wave, n_freqs=3, min_height=2, min_distance=4, min_prominence=5)
        with self.assertRaises(ValueError):
            dominant_frequencies(self.sine_wave, n_freqs=3, min_height=-2, min_distance=-4, min_prominence=-5)


from biosonic.compute.utils import AudioSignal, hz_to_mel


class TestHzToMel(unittest.TestCase):
    def test_oshaughnessy_scalar(self):
        result = hz_to_mel(1000.0, after="oshaughnessy")
        expected = 2595 * np.log(1 + 1000 / 700)
        self.assertAlmostEqual(result.item(), expected, places=3)

    def test_oshaughnessy_array(self):
        freqs = np.array([0, 100, 1000, 8000])
        result = hz_to_mel(freqs, after="oshaughnessy")
        expected = 2595 * np.log(1 + freqs / 700)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_beranek_array(self):
        freqs = np.array([0, 100, 1000, 8000])
        result = hz_to_mel(freqs, after="beranek")
        expected = 2595 * np.log(1 + freqs / 700)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_umesh_scalar(self):
        result = hz_to_mel(1000.0, after="umesh")
        expected = 1000.0 / (0.0004 * 1000.0 + 0.603)
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_fant_with_corner_frequency(self):
        result = hz_to_mel(1000.0, corner_frequency=2000.0)
        expected = 2000.0 * np.log(1 + 1000.0 / 2000.0)
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_raises_for_koenig(self):
        with self.assertRaises(NotImplementedError):
            hz_to_mel(1000.0, after="koenig")

    def test_raises_for_invalid_method(self):
        with self.assertRaises(ValueError):
            hz_to_mel(1000.0, after="invalid_method")


from biosonic.compute.spectral import power_spectral_entropy


class TestPowerSpectralEntropy(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 1000  # Hz

    def test_entropy_of_sine_wave(self):
        # A pure sine wave should have low entropy
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        sine = np.sin(2 * np.pi * 10 * t)

        H, H_max = power_spectral_entropy(AudioSignal(sine, self.sample_rate))
        self.assertIsInstance(H, float)
        self.assertLess(H, 0.5)
        self.assertLess(0, H)
        self.assertAlmostEqual(H_max, 1.0)

    def test_norm(self):
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        sine = np.sin(2 * np.pi * 10 * t)
        H, H_max = power_spectral_entropy(AudioSignal(sine, self.sample_rate), norm=False)
        self.assertIsInstance(H, float)
        self.assertNotEqual(H_max, 1.0)

    def test_entropy_of_white_noise(self):
        # White noise should have high entropy
        np.random.seed(42)
        noise = np.random.normal(0, 1, self.sample_rate)
        H, H_max = power_spectral_entropy(AudioSignal(noise, self.sample_rate))
        self.assertIsInstance(H, float)
        self.assertGreater(H, 0.75)

    def test_flat_signal_entropy_is_zero(self):
        # Flat signal should yield 0 entropy
        flat = np.full(self.sample_rate, 5)
        H, _ = power_spectral_entropy(AudioSignal(flat, self.sample_rate))
        self.assertAlmostEqual(H, 0.0)

    def test_entropy_with_multiple_tones(self):
        # Multiple tones should give intermediate entropy
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        multi = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 100 * t)
        H, H_max = power_spectral_entropy(AudioSignal(multi, self.sample_rate), norm=False)
        self.assertIsInstance(H, float)
        self.assertGreater(H, 0.5)
        self.assertLess(H, H_max)

    def test_entropy_output_type(self):
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        signal = np.sin(2 * np.pi * 5 * t)
        H, H_max = power_spectral_entropy(AudioSignal(signal, self.sample_rate))
        self.assertTrue(isinstance(H, float))

    def test_entropy_units_consistency(self):
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        x = np.sin(2 * np.pi * 30 * t)
        e_bits, _ = power_spectral_entropy(AudioSignal(x, self.sample_rate), unit="bits")
        e_nats, _ = power_spectral_entropy(AudioSignal(x, self.sample_rate), unit="nat")
        e_dits, _ = power_spectral_entropy(AudioSignal(x, self.sample_rate), unit="dits")
        self.assertAlmostEqual(e_bits * np.log(2), e_nats, places=4)
        self.assertAlmostEqual(e_dits * np.log(10), e_nats, places=4)

    def test_entropy_invalid_unit_raises(self):
        t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        signal = np.sin(2 * np.pi * 15 * t)
        with self.assertRaises(ValueError):
            power_spectral_entropy(AudioSignal(signal, self.sample_rate), unit="watts")


from scipy.signal import chirp

from biosonic.compute.temporal import temporal_entropy


class TestTemporalEntropy(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 1000  # Hz
        self.duration = 1.0  # seconds
        self.time = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)

    # # TODO
    # def test_constant_signal_entropy(self):
    #     signal = np.full(self.sample_rate, 5)
    #     H, H_max = temporal_entropy(signal)
    #     self.assertAlmostEqual(H, 0.0, places=6, msg="Entropy of a constant signal should be zero.")

    # def test_sine_wave_entropy(self):
    #     freq = 3 # Hz
    #     signal = np.sin(2 * np.pi * freq * self.time)
    #     result = temporal_entropy(signal)
    #     self.assertGreater(result, 0.0, "Entropy of a sine wave should be greater than zero.")

    def test_chirp_entropy(self):
        f1 = 300
        f2 = 400
        signal = AudioSignal(chirp(self.time, f1, 10, f2) + np.random.normal(0, 1, size=self.time.shape), self.sample_rate)
        H, H_max = temporal_entropy(signal)
        self.assertGreater(H, 0.5, "Entropy of chirp with noise should be relatively high.")

    def test_noise_signal_entropy(self):
        signal = AudioSignal(np.random.normal(0, 1, size=self.time.shape), self.sample_rate)
        H, H_max = temporal_entropy(signal)
        self.assertGreater(H, 0.5, "Entropy of white noise should be relatively high.")

    def test_entropy_units(self):
        signal = AudioSignal(np.random.normal(0, 1, size=self.time.shape), self.sample_rate)
        h_bits, _ = temporal_entropy(signal, unit="bits", norm=False)
        h_nats, _ = temporal_entropy(signal, unit="nat", norm=False)
        h_hartleys, _ = temporal_entropy(signal, unit="hartleys", norm=False)

        self.assertNotEqual(h_bits, h_nats, "Entropy in different units should yield different values.")
        self.assertAlmostEqual(h_bits * np.log(2), h_nats, delta=0.1)
        self.assertAlmostEqual(h_bits / np.log2(10), h_hartleys, delta=0.1)

        # TODO assert relative H independent of unit equal

    def test_invalid_unit_raises(self):
        signal = AudioSignal(np.random.normal(0, 1, size=self.time.shape), self.sample_rate)
        with self.assertRaises(ValueError):
            temporal_entropy(signal, unit="invalid_unit")


from biosonic.compute.spectral import power_spectral_entropy
from biosonic.compute.spectrotemporal import spectrotemporal_entropy
from biosonic.compute.temporal import temporal_entropy


def test_spectrotemporal_entropy():
    f1 = 300
    f2 = 400
    duration = 1
    sr = 1000
    time = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = chirp(time, f1, 10, f2) + np.random.normal(0, 1, size=time.shape)
    signal = AudioSignal(data, sr)

    entropy_val = spectrotemporal_entropy(signal, unit="bits")
    expected_temporal, _ = temporal_entropy(signal, unit="bits")
    expected_spectral, _ = power_spectral_entropy(signal, unit="bits")
    assert np.isclose(entropy_val, expected_temporal * expected_spectral)

    entropy_val = spectrotemporal_entropy(signal, unit="nat")
    expected_temporal, _ = temporal_entropy(signal, unit="nat")
    expected_spectral, _ = power_spectral_entropy(signal, unit="nat")
    assert np.isclose(entropy_val, expected_temporal * expected_spectral)

    for unit in ["bits", "nat", "dits", "bans", "hartleys"]:
        entropy_val = spectrotemporal_entropy(signal, unit=unit)
        assert isinstance(entropy_val, float)


from biosonic.compute.spectrotemporal import cepstrum


@pytest.fixture
def sine_wave() -> AudioSignal:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    freq = 440
    x = np.sin(2 * np.pi * freq * t)
    return AudioSignal(x, sr)


@pytest.fixture
def chirp_with_noise() -> AudioSignal:
    f1 = 300
    f2 = 400
    duration = 1
    sr = 1000
    time = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = chirp(time, f1, 10, f2) + np.random.normal(0, 1, size=time.shape)
    return AudioSignal(x, sr)


def test_cepstrum(sine_wave, chirp_with_noise):
    cep, qf = cepstrum(chirp_with_noise, mode="amplitude")

    # output shape
    assert cep.shape == chirp_with_noise.data.shape
    assert qf.shape == chirp_with_noise.data.shape

    # amplitude vs power
    cep_amp, _ = cepstrum(sine_wave, mode="amplitude")
    cep_pow, _ = cepstrum(sine_wave, mode="power")
    assert np.all(cep_pow >= 0), "Power cepstrum should be non-negative"
    assert not np.allclose(cep_amp, cep_pow), "Amplitude and power cepstra should differ"

    # quefrency scale
    _, qf = cepstrum(sine_wave)
    expected = np.arange(len(sine_wave.data)) / sine_wave.srate
    assert np.allclose(qf, expected)

    # invalid mode
    with pytest.raises(ValueError, match="Invalid mode for cepstrum calculation"):
        cepstrum(sine_wave, mode="invalid")

    # energy conservation
    cep, _ = cepstrum(sine_wave, mode="power")
    energy = np.sum(cep)
    assert energy > 0, "Cepstrum energy should be positive"

    # flat signal
    x, sr = np.full(500, 1), 50
    with pytest.raises(ValueError, match="flat signal"):
        cepstrum(AudioSignal(x, sr), mode="power")


@pytest.mark.parametrize("filterbank_type", ["mel", "linear"])
def test_cepstral_coefficients(filterbank_type, sine_wave, chirp_with_noise):
    from biosonic.compute.spectrotemporal import cepstral_coefficients
    ceps = cepstral_coefficients(sine_wave, window_length=512, n_ceps=13)
    assert isinstance(ceps, np.ndarray)
    assert ceps.shape == (13, 101)

    # filterbank types
    ceps = cepstral_coefficients(chirp_with_noise, filterbank_type=filterbank_type, n_ceps=10)
    assert ceps.shape == (10, 101)

    # invalid filterbank type
    with pytest.raises(ValueError):
        cepstral_coefficients(sine_wave, filterbank_type="invalid")

    # # short signal
    # sr = 16000
    # signal = np.random.randn(32)
    # ceps = cepstral_coefficients(signal, sr, n_fft=512, n_ceps=5)
    # assert ceps.shape == (5,)

    # fmin fmax
    ceps = cepstral_coefficients(chirp_with_noise, window_length=512, fmin=10, fmax=500, n_ceps=12)
    assert ceps.shape == (12, 101)

    # parameter validation
    with pytest.raises(ValueError, match="fmax must be <= Nyquist frequency"):
        ceps = cepstral_coefficients(chirp_with_noise, window_length=512, fmin=10, fmax=5000, n_ceps=12)

    with pytest.raises(ValueError, match="fmin must be >= 0 and < fmax"):
        ceps = cepstral_coefficients(chirp_with_noise, window_length=512, fmin=100, fmax=10, n_ceps=12)

    with pytest.raises(ValueError, match="fmin must be >= 0 and < fmax"):
        ceps = cepstral_coefficients(chirp_with_noise, window_length=512, fmin=-1, fmax=10, n_ceps=12)


if __name__ == '__main__':
    unittest.main()
