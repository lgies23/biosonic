from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy.fft import rfft, irfft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brent
from scipy.signal import windows
from typing import Tuple, List, Optional, Any
import matplotlib.pyplot as plt

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
        data : ArrayLike, 
        sr : int
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


def _find_pitch_candidates_(
        ac : ArrayLike, 
        sr : int, 
        min_pitch : int, 
        max_pitch : int, 
        num_candidates : int = 4, 
        octave_cost : float = 0.01
        ) -> ArrayLike:
    """
    Find pitch candidates based on autocorrelation peaks.
    """
    min_lag = int(sr / max_pitch)
    max_lag = int(sr / min_pitch)

    candidates : list[Tuple[float, float]] = []

    for lag in range(min_lag + 1, max_lag - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
            # parabolic interpolation
            y_m1 = ac[lag - 1]
            y_0 = ac[lag]
            y_p1 = ac[lag + 1]
            denom = (y_m1 - 2 * y_0 + y_p1)

            if denom == 0:
                refined_lag = lag
            else:
                refined_lag = lag + 0.5 * (y_m1 - y_p1) / denom

            # cost function (Boersma 1993, eq 26)
            r_tau = np.interp(refined_lag, np.arange(len(ac)), ac)
            strength = r_tau - octave_cost * 2 * np.log(min_pitch * refined_lag)

            # convert to pitch
            pitch = sr / refined_lag if refined_lag != 0 else 0
            candidates.append((pitch, strength))
    # # interpolation for higher accuracy
    # interp_ac = interp1d(np.arange(len(ac)), ac, kind='cubic', fill_value="extrapolate")

    # def cost_fn(
    #         lag : float
    #         ) -> float:
        
    #     if lag < min_lag or lag >= max_lag:
    #         return -np.inf
    #     r_tau = float(interp_ac(lag))
    #     return float(r_tau - octave_cost * 2 * np.log(min_pitch * lag))

    # candidates = []
    # for lag in range(min_lag, max_lag):
    #     if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
    #         res = minimize_scalar(lambda x: -cost_fn(x), bounds=(lag-1, lag+1), method='bounded')
    #         pitch = sr / res.x
    #         strength = -res.fun
    #         candidates.append((pitch, strength))

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
        F1 : float, 
        F2 : float, 
        voiced_unvoiced_cost : float, 
        octave_jump_cost : float
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
    all_candidates : List of lists of (pitch in Hz, strength)
    voiced_unvoiced_cost : Cost of voiced/unvoiced transition
    octave_jump_cost : Cost of pitch discontinuity in octaves

    Returns
    -------
    path 
        List of chosen pitch values, one per frame
    """
    num_frames = len(all_candidates)
    path_costs = []
    back_pointers : List[List[Any]] = []

    # initialization
    prev_costs = [ -strength for _, strength in all_candidates[0] ]
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
    # return 10 * np.log10(r_tmax/1-r_tmax)


def _autocorr(
        frame : ArrayLike,
        pad_width_for_pow2 : int
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
        data : ArrayLike, 
        sr : int, 
        min_pitch : int = 75, 
        max_pitch : int = 600,
        timestep : float = 0.01, 
        silence_thresh : float = 0.05, 
        voicing_thresh : float = 0.4,
        max_candidates :int = 5, 
        octave_cost : float = 0.01
    ) -> Tuple[ArrayLike, ArrayLike]:
    """
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

    window_length = 3 * (1 / min_pitch) # three periods of minimum frequency
    data_preprocessed = _preprocess_for_pitch_(data, sr)
    global_peak : float = np.max(np.abs(data_preprocessed))
    window_length_samples = int(window_length * sr)

    # precalculate for padding to power of two (step 3.6) 
    # - I do this here to save computation time despite it being a bit less readable
    n = window_length_samples + np.floor(window_length_samples/2)
    next_pow2 = 2 ** np.ceil(np.log2(n)).astype(int)
    pad_width_for_pow2 = next_pow2 - window_length_samples # full pad length needed including half a window size

    # 1. windowing
    framed_signal = frame_signal(data_preprocessed, sr, window_length_samples, timestep, normalize=False)
    window = windows.get_window("hann", window_length_samples)
    autocorr_hann = _autocorr(window, pad_width_for_pow2)
    autocorr_hann /= np.max(autocorr_hann)
    all_candidates = []

    for frame in framed_signal:
        # 3.2 subtract local average
        frame = frame - np.mean(frame)
        local_peak : float = np.max(np.abs(frame))

        # 3.3 see 3.11

        # 3.4 multipy by window function
        windowed_frame = frame * window

        # 3.5-3.9
        lag_domain = _autocorr(windowed_frame, pad_width_for_pow2)
        # normalize to range [-1,1]
        lag_domain = 2 * (lag_domain-float(np.min(lag_domain))) / (float(np.max(lag_domain))-float(np.min(lag_domain))) - 1 

        # 3.10 divide by autocorrelation of window
        sampled_autocorr = lag_domain / autocorr_hann
        # only include up to half the window length because unreliable above (p. 100, fig)
        sampled_autocorr = sampled_autocorr[:(window_length_samples//2)]
        
        # 3.11 find places and heights of maxima
        unvoiced_strength = voicing_thresh + max(0, 2 - ((local_peak / global_peak) / \
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

    # for i, cands in enumerate(all_candidates[:-5]):
    #     print(f"Frame {i}:")
    #     for pitch, strength in cands:
    #         print(f"  Pitch: {pitch:.2f} Hz, Strength: {strength:.3f}")

    # fig, axs = plt.subplots(2, 3)
    # axs[0][0].plot(frame)
    # axs[0][1].plot(window)
    # axs[0][2].plot(windowed_frame)
    # axs[1][0].plot(lag_domain)
    # axs[1][1].plot(autocorr_hann)
    # axs[1][2].plot(sampled_autocorr)

    time_points = np.arange(len(framed_signal)) * timestep
    pitch_track = _viterbi_pitch_path(
        all_candidates,
        voiced_unvoiced_cost=0.2,
        octave_jump_cost=0.2
    )
    return time_points, pitch_track