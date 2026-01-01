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

    # Biosonic pitch (unpack new return signature)
    result = boersma(
        data,
        sr,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        timestep=timestep
    )
    py_times, py_f0, all_candidates, intensities = result
    print("Biosonic times:", py_times)
    print("Parselmouth times:", pm_pitch.xs())
    print("Biosonic f0:", py_f0)
    print("Parselmouth f0:", pm_pitch.selected_array["frequency"])

    # print("\nFrame-by-frame candidate strengths and voicing threshold (first 10 frames):")
    # for i in range(min(10, len(all_candidates))):
    #     print(f"Frame {i} t={py_times[i]:.4f}s:")
    #     print("  Candidates (freq, strength):", all_candidates[i])
    #     print("  Intensity:", intensities[i])
    #     print("  Voicing threshold:", voicing_thresh)

    # Frame alignment
    assert pm_pitch.get_number_of_frames() == len(py_times)

    pm_f0 = pm_pitch.selected_array["frequency"]

    # Compare only voiced frames
    mask = (pm_f0 > 0) & (py_f0 > 0)
    assert np.any(mask), "No overlapping voiced frames found"

    # Print the five frequencies with the biggest mismatch (absolute difference)
    diffs = np.abs(pm_f0[mask] - py_f0[mask])
    if diffs.size > 0:
        top5_idx = np.argsort(diffs)[-5:][::-1]
        print("Top 5 mismatches (Praat, Biosonic, abs diff):")
        for idx in top5_idx:
            print(f"Praat: {pm_f0[mask][idx]:.2f} Hz, Biosonic: {py_f0[mask][idx]:.2f} Hz, Diff: {diffs[idx]:.2f} Hz")
            print("  Candidates (freq, strength):", all_candidates[idx])
            print("  Intensity:", intensities[idx])

    # All frequencies close?
    assert np.allclose(
        pm_f0[mask],
        py_f0[mask],
        rtol=.11
    )
