from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
from scipy.signal import windows

from biosonic.compute.utils import frame_signal
from biosonic.handle import AudioSignal


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
    """YIN f0 tracking over an entire signal. **Not yet finished**

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


def _preprocess_for_f0_(
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


def _find_f0_candidates_(
        ac: ArrayLike,
        sr: int,
        min_f0: int,
        max_f0: int,
        num_candidates: int,
        voicing_thresh: float,
        octave_cost: float
        ) -> ArrayLike:
    """
    Find f0 candidates based on autocorrelation peaks.
    """
    # TODO Keep this or not?
    # min_lag = int(sr / max_f0)  # Described by Boersma (1993) but not used in Praat implementation
    max_lag = int(sr / min_f0)

    candidates: list[Tuple[float, float]] = []

    # max_depth = 8  # window for sinc interpolation, can be tuned
    for lag in range(2, max_lag - 1):
        if ac[lag] > 0.5 * voicing_thresh and ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
            dr = 0.5 * (ac[lag + 1] - ac[lag - 1])
            d2r = 2 * ac[lag] - ac[lag - 1] - ac[lag + 1]
            refined_lag = lag + dr / d2r
            # TODO Reenable?
            # # Use sinc interpolation for sub-sample lag refinement
            # refined_lag, _ = _improve_sinc_maximum(ac, float(lag), max_depth)

            # cost function (Boersma 1993, eq 26)
            # r_tau = _sinc_interpolation(ac, refined_lag, max_depth)
            r_tau = ac[lag]
            if r_tau > 1:  # TODO Not sure if this ever happens, but Praat does it?
                r_tau = 1 / r_tau
            f0 = sr / refined_lag

            # convert to f0
            candidates.append((f0, r_tau))

    # Sort by strength and take top N-1
    candidates = sorted(candidates, key=lambda x: x[1] - octave_cost * np.log2(min_f0 / x[0]))[-num_candidates + 1::]

    return candidates


def _correct_candidate_strenghts(all_candidates, intensities, ceiling, silence_thresh, voicing_thresh, octave_cost):
    def get_corrected_strength(f0, strength, intensity):
        if not 0.0 < f0 < ceiling:  # TODO Praat keeps candidates above ceiling, since ceiling can change (??)
            return voicing_thresh + max(0.0, 0.0 if silence_thresh <= 0 else 2 - intensity / (silence_thresh / (1 + voicing_thresh)))
        else:
            return strength - octave_cost * np.log2(ceiling / f0)

    return [
        [
            (f0, get_corrected_strength(f0, strength, intensity))
            for f0, strength in candidates
        ]
        for candidates, intensity in zip(all_candidates, intensities)
    ]


def _viterbi_f0_path(
        all_candidates: List[List[Tuple[float, float]]],
        ceiling: float,
        voiced_unvoiced_cost: float,
        octave_jump_cost: float
    ) -> List[float]:
    """
    Finds the globally optimal f0 path using dynamic programming.

    Parameters
    ----------
    all_candidates: List of lists of (f0 in Hz, corrected strength)
    voiced_unvoiced_cost: Cost of voiced/unvoiced transition
    octave_jump_cost: Cost of f0 discontinuity in octaves

    Returns
    -------
    path
        List of chosen f0 values, one per frame
    """
    def _is_unvoiced(f):
        return not 0.0 < f < ceiling

    def _transition_cost(f1, f2) -> float:
        if _is_unvoiced(f1) and _is_unvoiced(f2):
            return 0.0
        elif _is_unvoiced(f1) or _is_unvoiced(f2):
            return voiced_unvoiced_cost
        else:
            return float(octave_jump_cost * abs(np.log2(f1 / f2)))

    num_frames = len(all_candidates)
    path_costs = []
    back_pointers: List[List[Any]] = []

    # Praat maximizes total strength minus transition costs (potayto)
    # This code minimizes transition costs minus total strength (potahto)

    # initialization
    prev_costs = [-strength for _, strength in all_candidates[0]]
    path_costs.append(prev_costs)
    back_pointers.append([None] * len(all_candidates[0]))

    # dynamic programming
    for i in range(1, num_frames):
        frame_costs = []
        frame_back_ptrs = []
        for curr_f0, curr_strength in all_candidates[i]:
            best_cost = np.inf
            best_prev_idx = None
            for j, (prev_f0, _) in enumerate(all_candidates[i-1]):
                total_cost = prev_costs[j] + _transition_cost(prev_f0, curr_f0) - curr_strength
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev_idx = j
            frame_costs.append(best_cost)
            frame_back_ptrs.append(best_prev_idx)
        path_costs.append(frame_costs)
        back_pointers.append(frame_back_ptrs)
        prev_costs = frame_costs

    # backtrace
    path = [None] * num_frames
    idx = int(np.argmin(path_costs[-1]))
    for t in reversed(range(num_frames)):
        f0, _ = all_candidates[t][idx]
        path[t] = f0
        idx = back_pointers[t][idx]

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
    ) -> NDArray[Any]:
    # 3.5 and 3.6 append half a window length of zeroes
    # plus enough until the length is a power of two
    padded_frame = np.pad(frame, (0, pad_width_for_pow2), mode='constant', constant_values=0)

    # 3.7 perform fft
    spec = rfft(padded_frame)

    # 3.8 square samples in frequency domain
    power_spec = spec * np.conj(spec)

    # 3.9 ifft of power spectrum
    lag_domain = irfft(power_spec)

    return lag_domain[:len(frame)]


def _frame_signal_praat(samples, sr, window_duration, time_step):
    assert window_duration > 0
    assert time_step > 0
    duration = len(samples) / sr

    if window_duration > duration:
        raise ValueError("Duration shorter than window length")

    n_frames = int(np.floor((duration - window_duration) / time_step)) + 1
    assert n_frames >= 1

    first_time = 0.5 * (duration - n_frames * time_step + time_step)

    return n_frames, first_time


def boersma(
        signal: AudioSignal,
        *,
        min_f0: int = 75,
        max_f0: int = 600,
        timestep: float = 0.01,
        silence_thresh: float = 0.03,
        voicing_thresh: float = 0.45,
        max_candidates: int = 15,
        octave_cost: float = 0.01,
        octave_jump_cost: float = 0.35,
        voiced_unvoiced_cost: float = 0.14,
        match_praat: bool = True,     # TODO Document
        very_accurate: bool = False,  # TODO Document
        plot: bool = False,
        **kwargs: Any
    ) -> Tuple[ArrayLike, ArrayLike, List[List[Tuple[float, float]]], ArrayLike]:
    """
    Estimate the fundamental frequency track of a signal using a Praat/Boersma-style autocorrelation method.

    Parameters
    ----------
    signal : AudioSignal
        Input audio signal (mono, normalized to [-1, 1]).
    min_f0 : int, optional
        Minimum f0 to search for (Hz). Default is 75.
    max_f0 : int, optional
        Maximum f0 to search for (Hz). Default is 600.
    timestep : float, optional
        Time step between f0 frames (seconds). Default is 0.01.
    silence_thresh : float, optional
        Silence threshold for voicing decision. Default is 0.03.
    voicing_thresh : float, optional
        Voicing threshold for candidate selection. Default is 0.45.
    max_candidates : int, optional
        Maximum number of f0 candidates per frame. Default is 15.
    octave_cost : float, optional
        Cost for octave errors in candidate selection. Default is 0.01.
    octave_jump_cost : float, optional
        Cost for octave jumps in dynamic programming. Default is 0.35.
    voiced_unvoiced_cost : float, optional
        Cost for voiced/unvoiced transitions in dynamic programming. Default is 0.14.
    plot : bool, optional
        If True, plot the f0 track on a spectrogram. Default is False.
    **kwargs : Any
        Additional keyword arguments for plotting.

    Returns
    -------
    time_points : np.ndarray
        Array of time points (seconds) for each f0 frame.
    f0_track : np.ndarray
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
    assert isinstance(signal, AudioSignal), "'signal' must be of type AudioSignal."
    assert not very_accurate, "very_accurate not yet implemented"  # TODO ?

    # Constants for AC (not "very exact") method:
    periods_per_window = 3  # three periods of minimum frequency
    brent_depth = 70  # 70 samples of sinc interpolation  # TODO: Use  # noqa: F841
    interpolation_depth = 0.5  # TODO: Forgot what this is  # noqa: F841

    assert signal.data.ndim == 1

    if min_f0 >= max_f0 or max_f0 > signal.srate / 2:
        raise ValueError("max_f0 should be greater than min_f0 and not greater than the nyquist frequency.")

    if max_candidates <= 1:
        raise ValueError("max_candidates should be greater than 1.")

    assert timestep > 0
    # TODO Auto time-step, as in Praat?

    duration = (1 / signal.srate) * len(signal.data)
    if min_f0 < periods_per_window / duration:
        raise ValueError(f"min_f0 must not be less than {periods_per_window / duration} Hz.")

    # not enough resolution above half the niquist frequency
    # -> amend f0 ceiling if applicable. From Soundgen (see references)
    # max_f0 = min(max_f0, signal.srate / 4)
    # Praat actually just enforces Nyquist frequency, but that's already hard enforced above

    data_preprocessed = signal.data  # _preprocess_for_f0_(signal.data, signal.srate)
    global_peak: float = np.max(np.abs(data_preprocessed - np.mean(data_preprocessed)))
    assert global_peak > 0  # TODO: What to do for a constant signal? Praat returns early.

    window_length = periods_per_window * (1 / min_f0)
    window_length_samples = int(window_length * signal.srate)  # window_length * signal.srate is positive, so `int()` floors
    if match_praat:
        nsamp_period = int(signal.srate / min_f0)  # Positive, so truncting is flooring
        halfnsamp_period = nsamp_period // 2 + 1

        halfnsamp_window = window_length_samples // 2 - 1
        if halfnsamp_window < 2:
            raise ValueError("Analysis window too short")  # TODO: Can this happen?
        window_length_samples = halfnsamp_window * 2

    # TODO: min_lag and max_lag?

    # precalculate for padding to power of two (step 3.6)
    # - I do this here to save computation time despite it being a bit less readable
    n = (1 + interpolation_depth) * window_length_samples
    next_pow2 = 2 ** np.ceil(np.log2(n)).astype(int)
    pad_width_for_pow2 = next_pow2 - window_length_samples  # full pad length needed including half a window size

    # 1. windowing
    if match_praat:
        n_frames, t1 = _frame_signal_praat(data_preprocessed, signal.srate, window_length, timestep)
        times = t1 + np.arange(n_frames) * timestep
        left_idx = np.floor((times - 0.5 / signal.srate) * signal.srate).astype(int)  # TODO Check again. Some off by one?
        framed_signal = [data_preprocessed[i + 1 - halfnsamp_window:i + halfnsamp_window + 1] for i in left_idx]
    else:
        framed_signal = frame_signal(AudioSignal(data_preprocessed, signal.srate), window_length_samples, timestep, normalize=False)
        times = 0.5 * (window_length_samples / signal.srate) + np.arange(len(framed_signal)) * timestep
    if match_praat:
        # To match Praat, we need to:
        # - Have a symmetric window (probably good to have anyway)
        # - Exclude the first and last sample of the window, which are 0 in SciPy
        #   (0.5 - 0.5 * cos(2 pi n / (M - 1)), for 0 <= n <= M-1)
        # This probably also explains the `halfnsamp_window = window_length_samples // 2 - 1` line above,
        # which has - instead of +, and seems to lose 2 samples.
        window = windows.get_window("hann", window_length_samples + 2, fftbins=False)[1:-1]
    else:
        window = windows.get_window("hann", window_length_samples, fftbins=False)
    autocorr_hann = _autocorr(window, pad_width_for_pow2)
    autocorr_hann /= autocorr_hann[0]  # Or np.max(autocorr_hann), but that's should be the same
    # TODO: brent_imax?

    all_candidates = []
    intensities = []

    for frame in framed_signal:
        # 3.2 subtract local average
        if match_praat:
            assert halfnsamp_window >= nsamp_period  # Only AC: periods_per_window == 3
            local_mean = np.mean(frame[halfnsamp_window - nsamp_period:halfnsamp_window + nsamp_period])
        else:
            local_mean = np.mean(frame)

        frame = (frame - local_mean) * window  # Frame are views of the array! Don't modify in-place!

        if match_praat:
            assert halfnsamp_window >= halfnsamp_period  # Only AC: periods_per_window == 3
            local_peak = np.max(np.abs(frame[halfnsamp_window - halfnsamp_period:halfnsamp_window + halfnsamp_period]))
        else:
            local_peak = np.max(np.abs(frame))

        # 3.3 see 3.11

        # 3.4 multiply by window function
        # windowed_frame = frame * window  # TODO Window before or after local_peak??
        windowed_frame = frame

        # 3.5-3.9
        lag_domain = _autocorr(windowed_frame, pad_width_for_pow2)

        # 3.10 divide by autocorrelation of window
        sampled_autocorr = lag_domain / (lag_domain[0] * autocorr_hann)

        # only include up to half the window length because unreliable above (p. 100, fig)
        sampled_autocorr = sampled_autocorr[:(window_length_samples//2)]

        # 3.11 find places and heights of maxima
        voiced_candidates = _find_f0_candidates_(
                sampled_autocorr,
                signal.srate,
                min_f0,
                max_f0,
                max_candidates,
                voicing_thresh,
                octave_cost
            )

        candidates = [(0.0, 0.0)] + voiced_candidates
        all_candidates.append(candidates)
        intensities.append(min(1.0, local_peak / global_peak))  # global_peak can't be 0; see above

    time_step_correction = 0.01 / timestep
    octave_jump_cost_scaled = octave_jump_cost * time_step_correction
    voiced_unvoiced_cost_scaled = voiced_unvoiced_cost * time_step_correction
    corrected_strength_candidates = _correct_candidate_strenghts(all_candidates, intensities, max_f0, silence_thresh, voicing_thresh, octave_cost)
    f0_track = _viterbi_f0_path(
        corrected_strength_candidates,
        max_f0,
        voiced_unvoiced_cost=voiced_unvoiced_cost_scaled,
        octave_jump_cost=octave_jump_cost_scaled
    )

    if plot:
        from biosonic.plot import plot_f0_on_spectrogram
        plot_f0_on_spectrogram(
            signal,
            times,
            f0_track,
            **kwargs)

    return times, np.asarray(f0_track), all_candidates, np.asarray(intensities)
