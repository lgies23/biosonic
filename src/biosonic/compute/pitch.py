from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
from scipy.signal import windows

from biosonic.compute.utils import frame_signal


def _difference_function(x: ArrayLike, max_lag: int) -> ArrayLike:
    """YIN difference function d(τ) = sum_j (x_j - x_{j+τ})^2"""
    N = len(x)
    d = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        d[lag] = np.sum((x[:N - lag] - x[lag:]) ** 2)
    return d


def _cumulative_mean_normalized_difference(d: ArrayLike) -> ArrayLike:
    """CMND function from the difference function."""
    cmndf = np.zeros_like(d)
    cmndf[0] = 1  # Avoid divide-by-zero
    cumsum = np.cumsum(d[1:])
    cmndf[1:] = d[1:] * np.arange(1, len(d)) / cumsum
    return cmndf


def _parabolic_interpolation(cmndf: ArrayLike, tau: int) -> float:
    """Refine τ estimate using parabolic interpolation."""
    if tau <= 0 or tau >= len(cmndf) - 1:
        return float(tau)
    alpha = cmndf[tau - 1]
    beta = cmndf[tau]
    gamma = cmndf[tau + 1]
    denominator = alpha + gamma - 2 * beta
    if denominator == 0:
        return float(tau)
    shift = 0.5 * (alpha - gamma) / denominator
    return float(tau + shift)


def _yin_single_window(
    x: ArrayLike,
    sr: int,
    window_length: int,
    time_step: int,
    bounds: Tuple[int, int],
    threshold: float = 0.1
) -> Optional[float]:
    """Estimate fundamental frequency from a single frame using YIN."""
    min_lag, max_lag = bounds
    frame = x[time_step:time_step + window_length]
    if len(frame) < window_length:
        return None

    # remove DC offset
    frame = frame - np.mean(frame)

    d = _difference_function(frame, max_lag)
    cmndf = _cumulative_mean_normalized_difference(d)

    max_lag = min(max_lag, len(cmndf) - 1)
    for i in range(min_lag, max_lag):
        if cmndf[i] < threshold:
            refined_tau = _parabolic_interpolation(cmndf, i)
            return sr / refined_tau

    return None


def yin(
    data: ArrayLike,
    sr: int,
    window_length: int,
    time_step_sec: float,
    flim: Tuple[int, int],
    threshold: float = 0.1
) -> Tuple[ArrayLike, ArrayLike]:
    """YIN pitch tracking over an entire signal. **Not yet finished**

    Returns
    -------
        times : ArrayLike
            time points (in seconds)
        frequencies : ArrayLike
            estimated f₀ at each time point
    """
    step_size = int(time_step_sec * sr)
    num_steps = (len(data) - window_length) // step_size

    min_lag = int(sr / flim[0])
    max_lag = int(sr / flim[1])

    times = []
    frequencies = []

    for i in range(num_steps):
        time_step = i * step_size
        time_sec = time_step / sr
        f0 = _yin_single_window(
            data,
            sr,
            window_length,
            time_step,
            bounds=(min_lag, max_lag),
            threshold=threshold
        )
        if f0 is not None:
            times.append(time_sec)
            frequencies.append(f0)

    return np.array(times), np.array(frequencies)


def _preprocess_for_pitch_(
        data: ArrayLike,
        sr: int
    ) -> ArrayLike:
    """
    Soft upsampling via frequency filtering and iFFT with longer FFT size
    to remove sidelobe of the FT of the Hanning window near f_nyquist as described in [1].

    References:
    ----------
    1. Boersma P. 1993 Accurate short-term analysis of the fundamental
    frequency and the harmonics-to-noise ratio of a sampled sound.
    IFA Proceedings 17, 97–110.
    """
    spectrum = rfft(data)
    nyquist = sr / 2
    freqs = np.linspace(0, nyquist, len(spectrum))

    # Soft taper from 95% to 100% Nyquist
    taper = np.ones_like(freqs)
    taper_start = 0.95 * nyquist
    taper_end = nyquist
    taper_region = (freqs >= taper_start) & (freqs <= taper_end)
    taper[taper_region] = 0.5 * (1 + np.cos(np.pi * (freqs[taper_region] - taper_start) / (taper_end - taper_start)))
    taper[freqs > nyquist] = 0

    spectrum *= taper

    # inverse FFT back to time domain
    filtered_signal = irfft(spectrum, n=len(data))

    return filtered_signal


def _sinc_interpolation(
        y: NDArray[np.floating],
        x: float,
        max_depth: int
        ) -> float:
    midleft = np.floor(x).astype(int)
    midright = midleft + 1
    if x == midleft:
        return np.float64(y[midleft])

    max_depth = min(min(max_depth, midleft), len(y) - midright)
    assert max_depth > 2

    left = midright - max_depth
    right = midleft + max_depth

    result = 0.0

    a = np.pi * (x - midleft)
    halfsina = 0.5 * np.sin(a)
    aa = a / (x - left + 1)
    daa = np.pi / (x - left + 1)
    for i in range(midleft, left - 1, -1):
        d = halfsina / a * (1 + np.cos(aa))
        result += y[i] * d
        a += np.pi
        aa += daa
        halfsina = -halfsina

    a = np.pi * (midright - x)
    halfsina = 0.5 * np.sin(a)
    aa = a / (right - x + 1)
    daa = np.pi / (right - x + 1)
    for i in range(midright, right + 1):
        d = halfsina / a * (1 + np.cos(aa))
        result += y[i] * d
        a += np.pi
        aa += daa
        halfsina = -halfsina

    return result


def _improve_sinc_maximum(y: ArrayLike, x: float, max_depth: int) -> Tuple[float, float]:
    """
    Refine the maximum using sinc interpolation and Brent's method.
    Returns (refined_x, value_at_refined_x)
    """
    assert 0 <= x < len(y)

    def _neg_sinc_interp(x_: float) -> float:
        return -_sinc_interpolation(y, x_, max_depth)

    xmin, fval, _, _ = scipy.optimize.brent(
        _neg_sinc_interp,
        brack=(x - 1, x, x + 1),
        tol=1e-10,
        full_output=True,
        maxiter=60
    )
    return xmin, -fval


def _find_pitch_candidates_(
        ac: ArrayLike,
        sr: int,
        min_pitch: int,
        max_pitch: int,
        num_candidates: int = 4,
        octave_cost: float = 0.01
        ) -> ArrayLike:
    """
    Find pitch candidates based on autocorrelation peaks.
    """
    min_lag = int(sr / max_pitch)
    max_lag = int(sr / min_pitch)

    candidates: list[Tuple[float, float]] = []

    max_depth = 8  # window for sinc interpolation, can be tuned
    for lag in range(min_lag + 1, max_lag - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
            # Use sinc interpolation for sub-sample lag refinement
            refined_lag, _ = _improve_sinc_maximum(ac, float(lag), max_depth)

            # cost function (Boersma 1993, eq 26)
            r_tau = _sinc_interpolation(ac, refined_lag, max_depth)
            strength = r_tau - octave_cost * 2 * np.log(min_pitch * refined_lag)

            # convert to pitch
            pitch = sr / refined_lag if refined_lag != 0 else 0
            candidates.append((pitch, strength))

    # Sort by strength and take top N-1
    candidates = sorted(candidates, key=lambda x: -x[1])[:num_candidates - 1]

    # normalize to [0,1]
    max_strength = max(s for _, s in candidates) if candidates else 1.0
    if max_strength > 0:
        candidates = [(p, s / max_strength) for p, s in candidates]
    else:
        candidates = [(p, 0.0) for p, s in candidates]

    return candidates


def _transition_cost(
        F1: float,
        F2: float,
        voiced_unvoiced_cost: float,
        octave_jump_cost: float
    ) -> float:
    if F1 == 0.0 and F2 == 0.0:
        return 0.0
    elif F1 == 0.0 or F2 == 0.0:
        return voiced_unvoiced_cost
    else:
        return float(octave_jump_cost * abs(np.log2(F1 / F2)))


def _viterbi_pitch_path(
        all_candidates: List[List[Tuple[float, float]]],
        voiced_unvoiced_cost: float = 0.2,
        octave_jump_cost: float = 0.2
    ) -> List[float]:
    """
    Finds the globally optimal pitch path using dynamic programming.

    Parameters
    ----------
    all_candidates: List of lists of (pitch in Hz, strength)
    voiced_unvoiced_cost: Cost of voiced/unvoiced transition
    octave_jump_cost: Cost of pitch discontinuity in octaves

    Returns
    -------
    path
        List of chosen pitch values, one per frame
    """
    num_frames = len(all_candidates)
    path_costs = []
    back_pointers: List[List[Any]] = []

    # initialization
    prev_costs = [-strength for _, strength in all_candidates[0]]
    path_costs.append(prev_costs)
    back_pointers.append([None] * len(all_candidates[0]))

    # dynamic programming
    for t in range(1, num_frames):
        frame_costs = []
        frame_back_ptrs = []
        for j, (curr_pitch, curr_strength) in enumerate(all_candidates[t]):
            best_cost = float('inf')
            best_prev_idx = None
            for i, (prev_pitch, _) in enumerate(all_candidates[t-1]):
                trans_cost = _transition_cost(
                    prev_pitch, curr_pitch,
                    voiced_unvoiced_cost, octave_jump_cost
                )
                total_cost = path_costs[t-1][i] + trans_cost - curr_strength
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev_idx = i
            frame_costs.append(best_cost)
            frame_back_ptrs.append(best_prev_idx)
        path_costs.append(frame_costs)
        back_pointers.append(frame_back_ptrs)

    # backtrace
    path = [0.0] * num_frames
    idx = int(np.argmin(path_costs[-1]))
    for t in reversed(range(num_frames)):
        pitch, _ = all_candidates[t][idx]
        path[t] = pitch
        idx = back_pointers[t][idx] if back_pointers[t][idx] is not None else 0

    return path


# def hnr() -> float:
#     """
#     Calculate harmonics-to-noise ratio as in Boersma 1993. Returns value in dB.
#     """
#     r_tmax =
#     return 10 * np.log10(r_tmax/1-r_tmax)


def _autocorr(
        frame: ArrayLike,
        pad_width_for_pow2: int
    ) -> NDArray:
    # 3.5 and 3.6 append half a window length of zeroes
    # plus enough until the length is a power of two
    frame = np.pad(frame, (0, pad_width_for_pow2), mode='constant', constant_values=0)

    # 3.7 perform fft
    spec = rfft(frame)

    # 3.8 square samples in frequency domain
    power_spec = spec * np.conj(spec)

    # 3.9 ifft of power spectrum
    lag_domain = irfft(power_spec)

    return lag_domain[: len(frame) // 2]


def boersma(
        data: ArrayLike,
        sr: int,
        min_pitch: int = 75,
        max_pitch: int = 600,
        timestep: float = 0.01,
        silence_thresh: float = 0.03,
        voicing_thresh: float = 0.45,
        max_candidates: int = 15,
        octave_cost: float = 0.01,
        octave_jump_cost: float = 0.35,
        voiced_unvoiced_cost: float = 0.14,
        plot: bool = False,
        **kwargs: Any
    ) -> Tuple[ArrayLike, ArrayLike, List[List[Tuple[float, float]]], ArrayLike]:
    """
    Estimate the fundamental frequency track of a signal using a Praat/Boersma-style autocorrelation method.

    Parameters
    ----------
    data : ArrayLike
        Input audio signal (mono, normalized to [-1, 1]).
    sr : int
        Sampling rate (Hz).
    min_pitch : int, optional
        Minimum pitch to search for (Hz). Default is 75.
    max_pitch : int, optional
        Maximum pitch to search for (Hz). Default is 600.
    timestep : float, optional
        Time step between pitch frames (seconds). Default is 0.01.
    silence_thresh : float, optional
        Silence threshold for voicing decision. Default is 0.03.
    voicing_thresh : float, optional
        Voicing threshold for candidate selection. Default is 0.45.
    max_candidates : int, optional
        Maximum number of pitch candidates per frame. Default is 15.
    octave_cost : float, optional
        Cost for octave errors in candidate selection. Default is 0.01.
    octave_jump_cost : float, optional
        Cost for octave jumps in dynamic programming. Default is 0.35.
    voiced_unvoiced_cost : float, optional
        Cost for voiced/unvoiced transitions in dynamic programming. Default is 0.14.
    plot : bool, optional
        If True, plot the pitch track on a spectrogram. Default is False.
    **kwargs : Any
        Additional keyword arguments for plotting.

    Returns
    -------
    time_points : np.ndarray
        Array of time points (seconds) for each pitch frame.
    pitch_track : np.ndarray
        Array of estimated F0 values (Hz) for the best path.
    all_candidates : list of list of (float, float)
        List of candidate (frequency, strength) tuples for each frame.
    intensities : np.ndarray
        Array of frame intensities (relative to global peak).

    References
    ----------
    1. Boersma P. 1993 Accurate short-term analysis of the fundamental
       frequency and the harmonics-to-noise ratio of a sampled sound.
       IFA Proceedings 17, 97–110.
    2. Anikin A. 2019. Soundgen: an open-source tool for synthesizing
       nonverbal vocalizations. Behavior Research Methods, 51(2), 778-792.
    """

    # make sure, the signal is inside the bounds [-1, 1]
    if np.max(np.abs(data)) > 1:
        raise ValueError("the signal needs to be within the bounds [-1, 1]")

    if min_pitch >= max_pitch or max_pitch >= sr / 2:
        raise ValueError("max_pitch should be greater than min_pitch and below the nyquist frequency.")

    # not enough resolution above half the niquist frequency
    # -> amend pitch ceiling if applicable. From Soundgen (see references)
    # max_pitch = min(max_pitch, sr / 4)

    window_length = 3 * (1 / min_pitch)  # three periods of minimum frequency
    data_preprocessed = data  # _preprocess_for_pitch_(data, sr)
    global_peak: float = np.max(np.abs(data_preprocessed - np.mean(data_preprocessed)))

    # global_peak: float = np.max(np.abs(data_preprocessed))
    window_length_samples = int(window_length * sr)

    # precalculate for padding to power of two (step 3.6)
    # - I do this here to save computation time despite it being a bit less readable
    n = window_length_samples + np.floor(window_length_samples/2)
    next_pow2 = 2 ** np.ceil(np.log2(n)).astype(int)
    pad_width_for_pow2 = next_pow2 - window_length_samples  # full pad length needed including half a window size

    # 1. windowing
    framed_signal = frame_signal(data_preprocessed, sr, window_length_samples, timestep, normalize=False)
    window = windows.get_window("hann", window_length_samples)
    autocorr_hann = _autocorr(window, pad_width_for_pow2)
    autocorr_hann /= np.max(autocorr_hann)
    all_candidates = []
    intensities = []

    for frame in framed_signal:
        # 3.2 subtract local average
        frame = frame - np.mean(frame)
        local_peak: float = np.max(np.abs(frame))

        # 3.3 see 3.11

        # 3.4 multiply by window function
        windowed_frame = frame * window

        # 3.5-3.9
        lag_domain = _autocorr(windowed_frame, pad_width_for_pow2)

        # 3.10 divide by autocorrelation of window
        sampled_autocorr = lag_domain / autocorr_hann

        # only include up to half the window length because unreliable above (p. 100, fig)
        sampled_autocorr = sampled_autocorr[:(window_length_samples//2)]

        # 3.11 find places and heights of maxima
        unvoiced_strength = voicing_thresh + max(0, 2 - ((local_peak / global_peak) /
                                                         (silence_thresh / (1 + voicing_thresh))))

        voiced_candidates = _find_pitch_candidates_(
                sampled_autocorr,
                sr,
                min_pitch,
                max_pitch,
                max_candidates,
                octave_cost
            )

        candidates = [(0.0, unvoiced_strength)] + voiced_candidates
        all_candidates.append(candidates)
        intensities.append(local_peak / global_peak if global_peak > 0 else 0.0)

    # frame timing: match Yannicks code (centered on window)
    n_frames = len(framed_signal)
    t0 = 0.5 * (len(data) / sr - n_frames * timestep + timestep)
    time_points = t0 + np.arange(n_frames) * timestep

    # Praat-style: scale transition costs by time step as in Yannicks code
    dt = timestep
    time_step_correction = 0.01 / dt if dt else 1.0
    octave_jump_cost_scaled = octave_jump_cost * time_step_correction
    voiced_unvoiced_cost_scaled = voiced_unvoiced_cost * time_step_correction

    pitch_track = _viterbi_pitch_path(
        all_candidates,
        voiced_unvoiced_cost=voiced_unvoiced_cost_scaled,
        octave_jump_cost=octave_jump_cost_scaled
    )

    if plot:
        from biosonic.plot import plot_pitch_on_spectrogram
        plot_pitch_on_spectrogram(
            data,
            sr,
            time_points,
            pitch_track,
            **kwargs)

    return np.asarray(time_points), np.asarray(pitch_track), all_candidates, np.asarray(intensities)
