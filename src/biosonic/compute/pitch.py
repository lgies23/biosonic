from numpy.typing import ArrayLike
import numpy as np
from scipy.fft import rfft, irfft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brent
from scipy.signal import windows, correlate
from typing import Tuple, List
import matplotlib.pyplot as plt

from biosonic.compute.utils import window_signal, frame_signal


def afc(
        data : ArrayLike, 
        window_length : int, 
        time_step : int, 
        max_lag : int
) -> ArrayLike:
    """
    Compute ACF values for all lags from 0 to max_lag with boundary checking.
    """
    acf_vals = []
    x = data[time_step : time_step + window_length]
    if len(x) < window_length:
        return np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        start = time_step + lag
        y = data[start : start + window_length]
        if len(y) != window_length:
            break  # avoid broadcasting error
        acf_vals.append(np.sum(x * y))
    
    return np.array(acf_vals)


def cmnd(
        data: ArrayLike, 
        window_length: int, 
        time_step: int, 
        bounds: Tuple[int, int]
) -> ArrayLike:
    min_lag, max_lag = bounds

    acf = afc(data, window_length, time_step, max_lag)

    actual_max_lag = len(acf) - 1
    if actual_max_lag < max_lag:
        max_lag = actual_max_lag

    df = []
    for lag in range(min_lag, max_lag + 1):
        seg = data[time_step + lag : time_step + lag + window_length]
        if len(seg) < window_length:
            break
        df_val = acf[0] + np.sum(seg ** 2) - 2 * acf[lag]
        df.append(df_val)

    df = np.array(df)
    cmndf = np.zeros_like(df)
    cmndf[0] = 1  # Avoid divide-by-zero
    cumsum = np.cumsum(df)
    cmndf[1:] = df[1:] * np.arange(1, len(df)) / cumsum[1:]

    return cmndf


def fundamental_frequency(
    data: ArrayLike,
    sr: int,
    window_length: int,
    time_step: int,
    bounds: Tuple[int, int],
    threshold: float = 0.1
) -> float:
    cmndf = cmnd(data, window_length, time_step, bounds)
    for i, val in enumerate(cmndf):
        if val < threshold:
            return sr / (i + bounds[0])
    return sr / (np.argmin(cmndf) + bounds[0])


def yin(
    data: ArrayLike,
    sr: int,
    window_length: int,
    time_step_sec: float,
    bounds: Tuple[int, int],
    threshold: float = 0.1
) -> ArrayLike:
    step_size = int(time_step_sec * sr)
    num_steps = (len(data) - window_length) // step_size
    return np.array([
        fundamental_frequency(
            data,
            sr,
            window_length,
            i * step_size,
            bounds,
            threshold
        )
        for i in range(num_steps)
    ])
    

def preprocess_for_pitch_(
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
    N = len(data)
    spectrum = rfft(data)
    nyquist = sr / 2
    freqs = np.linspace(0, nyquist, len(spectrum))

    # Linear taper to zero from 95% to 100% Nyquist
    taper_start = 0.95 * nyquist
    taper = np.ones_like(spectrum)
    taper[freqs > taper_start] = 1 - (freqs[freqs > taper_start] - taper_start) / (nyquist - taper_start)
    taper[freqs > nyquist] = 0
    spectrum *= taper

    new_N = 2**int(np.ceil(np.log2(N)) + 1)  # One order higher
    filtered_signal = irfft(spectrum, n=new_N)
    return filtered_signal[:N]


# def get_frames_(
#         data : ArrayLike, 
#         sr : int, 
#         frame_step : float, 
#         min_pitch : int, 
#         for_hnr : bool = False
#     ) -> ArrayLike:
#     """
#     Slice the signal into overlapping frames based on min pitch.
#     """
#     periods = 6 if for_hnr else 3
#     win_len_sec = periods / min_pitch
#     win_len_samples = int(win_len_sec * sr)
#     step_samples = int(frame_step * sr)
#     frames = []
#     for start in range(0, len(data) - win_len_samples + 1, step_samples):
#         frames.append(data[start:start + win_len_samples])
#     return np.array(frames)

# def compute_autocorrelation_(
#         frame : ArrayLike
#         ) -> ArrayLike:
#     """
#     Window, pad, FFT, square, IFFT to get autocorrelation.
#     """
#     window = np.hanning(len(frame))
#     win_frame = frame * window

#     # Padding
#     padded_len = int(2**np.ceil(np.log2(2 * len(win_frame))))
#     padded_frame = np.zeros(padded_len)
#     padded_frame[:len(win_frame)] = win_frame

#     spectrum = rfft(padded_frame)
#     power_spectrum = spectrum * np.conj(spectrum)
#     ac = irfft(power_spectrum)
#     return ac / np.max(ac)  # normalize

# def find_pitch_candidates_(
#         ac : ArrayLike, 
#         sr : int, 
#         min_pitch : int, 
#         max_pitch : int, 
#         num_candidates : int = 4, 
#         octave_cost : float = 0.01
#         ) -> ArrayLike:
#     """
#     Find pitch candidates based on autocorrelation peaks.
#     """
#     min_lag = int(sr / max_pitch)
#     max_lag = int(sr / min_pitch)

#     # Interpolation for higher accuracy
#     lags = np.arange(min_lag, max_lag)
#     interp_ac = interp1d(np.arange(len(ac)), ac, kind='cubic', fill_value="extrapolate")

#     def cost_fn(
#             lag : float
#             ) -> float:
        
#         if lag < min_lag or lag >= max_lag:
#             return -np.inf
#         r_tau = float(interp_ac(lag))
#         return float(r_tau - octave_cost * 2 * np.log(min_pitch * lag))


#     candidates = []
#     for lag in range(min_lag, max_lag):
#         if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
#             res = minimize_scalar(lambda x: -cost_fn(x), bounds=(lag-1, lag+1), method='bounded')
#             pitch = sr / res.x
#             strength = -res.fun
#             candidates.append((pitch, strength))

#     candidates = sorted(candidates, key=lambda x: -x[1])[:num_candidates - 1]
#     return candidates


# def hnr() -> float:
#     """
#     Calculate harmonics-to-noise ratio as in Boersma 1993. Returns value in dB.
#     """
#     r_tmax = 
    # return 10 * np.log10(r_tmax/1-r_tmax)


def find_pitch_candidates_(
    sampled_autocorr: np.ndarray,
    sr: int,
    min_pitch: float,
    max_pitch: float,
    max_candidates: int,
    octave_cost: float
) -> List[Tuple[float, float]]:
    """
    Find voiced pitch candidates using autocorrelation peak detection.

    Returns a list of (frequency, strength) pairs.
    """
    # Convert frequency bounds to lag bounds (in samples)
    min_lag = int(np.floor(sr / max_pitch))
    max_lag = int(np.ceil(sr / min_pitch))
    
    def neg_autocorr(lag: float) -> float:
        """Negative interpolated autocorrelation for optimization."""
        i = int(np.floor(lag))
        frac = lag - i
        if i + 1 >= len(sampled_autocorr):
            return 0.0
        return -(1 - frac) * sampled_autocorr[i] - frac * sampled_autocorr[i + 1]

    # Find rough peaks (local maxima)
    rough_peaks = []
    for i in range(min_lag + 1, min(max_lag, len(sampled_autocorr) - 1)):
        if sampled_autocorr[i] > sampled_autocorr[i - 1] and sampled_autocorr[i] > sampled_autocorr[i + 1]:
            # Refine using Brent's method in a small interval around the peak
            try:
                lag_opt = brent(neg_autocorr, brack=(i - 1, i, i + 1), tol=1e-3)
                if lag_opt == 0:
                    continue  # Avoid division by zero
                strength = -neg_autocorr(lag_opt)
                pitch = sr / lag_opt
                if min_pitch <= pitch <= max_pitch:
                    # Apply octave cost: favor higher frequencies (lower lags)
                    cost = 1 - octave_cost * (np.log2(sr / lag_opt / min_pitch))**2
                    score = strength * cost
                    rough_peaks.append((pitch, score))
            except ValueError:
                continue

    # Sort by strength and take top candidates
    rough_peaks.sort(key=lambda x: x[1], reverse=True)
    return rough_peaks[:max_candidates - 1]


def autocorr_(frame, pad_width_for_pow2):
    # 3.5 and 3.6 append half a window length of zeroes 
    # plus enough until the length is a power of two
    frame = np.pad(frame, (0, pad_width_for_pow2), mode='constant', constant_values=0)
    
    # 3.7 perform fft
    spec = rfft(frame)

    # 3.8 square samples in frequency domain
    power_spec = spec ** 2 # TODO check if this is the correct way

    # 3.9 ifft of power spectrum
    lag_domain = irfft(power_spec)

    return lag_domain[len(frame) // 2 :]


def boersma(
        data : ArrayLike, 
        sr : int, 
        min_pitch : int = 75, 
        max_pitch : int = 600,
        timestep : float = 0.01, 
        silence_thresh : float = 0.05, 
        voicing_thresh : float = 0.4,
        max_candidates :int = 4, 
        octave_cost : float = 0.01
    ) -> Tuple[ArrayLike, ArrayLike]:
    """
    References:
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
    max_pitch = min(max_pitch, sr / 4) 

    window_length = 3 * (1 / min_pitch) # three periods of minimum frequency
    #data_preprocessed = preprocess_for_pitch_(data, sr) # TODO am I doing this correctly?
    data_preprocessed = data
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
    autocorr_hann = autocorr_(window, pad_width_for_pow2)
    # autocorr_hann /= autocorr_hann
    all_candidates = []

    for frame in framed_signal:
        frame = frame - np.mean(frame)
        # TODO step 3.3
        local_peak : float = np.max(np.abs(frame))

        # 3.4 multipy by window function
        frame = frame * window

        # 3.5-3.9
        lag_domain = autocorr_(frame, pad_width_for_pow2)

        # 3.10 divide by autocorrelation of window
        sampled_autocorr = lag_domain / autocorr_hann

        # 3.11 find places and heights of maxima
        unvoiced_strength = voicing_thresh + max(0, 2 - (local_peak / global_peak)) / \
                            (silence_thresh * (1 + voicing_thresh))
        
        voiced_candidates = find_pitch_candidates_( # TODO check candidate detection
                sampled_autocorr, 
                sr, 
                min_pitch, 
                max_pitch,
                max_candidates,
                octave_cost
            )
        
        candidates = [(0.0, unvoiced_strength)] + voiced_candidates
        all_candidates.append(candidates)

    fig, ax = plt.subplots()
    ax.plot(sampled_autocorr)
    # ax.plot(autocorr_hann)
    ax.plot(frame)

    time_points = np.arange(len(framed_signal)) * timestep
    return time_points, all_candidates