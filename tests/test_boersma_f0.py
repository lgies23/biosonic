import numpy as np
import parselmouth
import pytest

from biosonic.compute.fundamental import boersma
from biosonic.compute.utils import AudioSignal
from biosonic.handle import read_wav


def _get_f0_error_bounds(f, sr):
    # Since we do not do the sinc-based interpolation (yet?), frequencies
    # are a bit off from Praat. Additionally, since (candidate) f0 ~ sr / lag,
    # the sampling error on frequencies is not uniform. The following should
    # give a rough estimate of the error bounds:
    # Half-distance between two successive lags' (n & n+1) frequencies:
    #   (sr / n - sr / (n+1)) / 2 =
    #   ((n+1) sr - n sr) / (2 n (n+1)) =
    #   sr / (2 n (n+1))
    # We don't know the lag of a specific f0 candidate, but since
    #   f ~ sr / n
    # we can bound
    #   n = ceil(sr / f)
    # Either way, it seems good enough for now, passing our tests.
    n = np.ceil(f / sr)
    margin = sr / (2 * n * (n + 1))
    return f - margin, f + margin


@pytest.mark.parametrize(
    "wav_path, min_f0, max_f0, timestep",
    [
        ("./docs/source/example_data/201.wav", 2000, 6000, 0.02),
    ],
)
def test_boersma_matches_parselmouth(
    wav_path: str,
    min_f0: int,
    max_f0: int,
    timestep: float,
):
    # Load audio
    audio_signal: AudioSignal = read_wav(wav_path)
    sound = parselmouth.Sound(wav_path)

    # Basic consistency check
    assert sound.duration == pytest.approx(len(audio_signal.data) / audio_signal.srate)

    # Parselmouth pitch
    pm_pitch = sound.to_pitch(
        pitch_floor=min_f0,
        pitch_ceiling=max_f0,
        time_step=timestep,
    )
    # Biosonic f0
    py_times, py_f0, py_all_candidates, py_intensities = boersma(
        audio_signal,
        min_f0=min_f0,
        max_f0=max_f0,
        timestep=timestep
    )

    assert pm_pitch.get_number_of_frames() == len(py_times)
    assert pm_pitch.xs() == pytest.approx(py_times)

    assert [frame.intensity for frame in pm_pitch] == pytest.approx(py_intensities)
    assert [len(frame) for frame in pm_pitch] == [len(candidates) for candidates in py_all_candidates]

    pm_f0 = pm_pitch.selected_array["frequency"]

    pm_voiced, py_voiced = pm_f0 != 0, py_f0 != 0
    assert np.all(pm_voiced == py_voiced)

    lower, upper = _get_f0_error_bounds(pm_f0[pm_voiced], audio_signal.srate)
    assert np.all((lower <= py_f0[py_voiced]) & (py_f0[py_voiced] <= upper))

    pm_candidates = np.sort(pm_pitch.to_array(), axis=0)
    for i, candidates in enumerate(py_all_candidates):
        k = len(candidates)
        print(py_times[i], len(candidates), len(pm_pitch[i]))
        candidates = sorted(candidates)
        frequencies, strengths = map(np.array, zip(*candidates))

        lower, upper = _get_f0_error_bounds(pm_candidates['frequency'][1:k, i], audio_signal.srate)
        assert np.all((lower <= frequencies[1:]) & (frequencies[1:] <= upper))

        # 20% relative error is a wide margin, but I've got no better guess
        assert strengths == pytest.approx(pm_candidates['strength'][:k, i], rel=0.2)
