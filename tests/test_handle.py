import numpy as np

def test_convert_dtype():
    from biosonic.handle import convert_dtype
    
    # int16 -> float32 -> int16
    original = np.array([-32768, 0, 1246, 32767], dtype=np.int16)
    float_data = convert_dtype(original, "float32")
    recovered = convert_dtype(float_data, "int16")

    # allow minor loss due to float rounding
    assert np.allclose(original, recovered, atol=1)

    # uint8 to float
    original = np.array([0, 128, 255], dtype=np.uint8)
    float_data = convert_dtype(original, "float32")
    expected = np.array([-1.0, 0.0, 0.9921875], dtype=np.float32)
    assert np.allclose(float_data, expected, atol=1e-5)

def test_read_wav(tmp_path):
    from scipy.io import wavfile
    from biosonic.handle import read_wav
    # Create a dummy WAV file
    sr = 44100
    data = (np.sin(2 * np.pi * 440 * np.arange(44100) / sr) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "test.wav", sr, data)

    signal = read_wav(tmp_path / "test.wav", quantization="float32")

    assert signal.sr == 44100
    assert signal.n_channels == 1
    assert signal.data.dtype == np.float32