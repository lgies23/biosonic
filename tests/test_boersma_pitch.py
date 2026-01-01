import numpy as np
import parselmouth
import pytest

from biosonic.compute.pitch import boersma
from biosonic.handle import read_wav


@pytest.mark.parametrize(
    "wav_path, min_pitch, max_pitch, timestep",
    [
        ("./example_data/201.wav", 2000, 6000, 0.02),
    ],
)
def test_boersma_matches_parselmouth(
    wav_path: str,
    min_pitch: int,
    max_pitch: int,
    timestep: float,
):
    # Load audio
    data, sr, _, _ = read_wav(wav_path)
    sound = parselmouth.Sound(wav_path)

    # Basic consistency check
    assert sound.duration == pytest.approx(len(data) / sr)

    # Parselmouth pitch
    pm_pitch = sound.to_pitch(
        pitch_floor=min_pitch,
        pitch_ceiling=max_pitch,
        time_step=timestep,
    )
    print(pm_pitch.selected_array["frequency"])

    # Biosonic pitch
    py_times, py_f0 = boersma(
        data,
        sr,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        timestep=timestep,
        octave_cost=0.01,
        voiced_unvoiced_cost=0.14,
        transition_cost=0.35,
    )
    print(py_f0)

    # Frame alignment
    assert pm_pitch.get_number_of_frames() == len(py_times)

    pm_f0 = pm_pitch.selected_array["frequency"]

    # Compare only voiced frames
    mask = (pm_f0 > 0) & (py_f0 > 0)
    assert np.any(mask), "No overlapping voiced frames found"

    # Numerical agreement
    assert np.allclose(
        pm_f0[mask],
        py_f0[mask],
        rtol=1e-3,
        atol=0,
    )
