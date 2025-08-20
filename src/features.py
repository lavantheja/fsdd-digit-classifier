# src/features.py
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import librosa


@dataclass
class FeatureConfig:
    # Target sampling rate for all audio (FSDD is 8 kHz)
    target_sr: int = 8000

    # STFT / windowing tuned for telephony-like speech
    n_fft: int = 512
    win_length: int = 400       # 50 ms @ 8 kHz
    hop_length: int = 160       # 20 ms @ 8 kHz

    # Mel filterbank tuned for 8 kHz
    n_mels: int = 32            # 26–40 is typical; 32 is a good middle
    fmin: float = 20.0
    fmax: Optional[float] = None  # will default to min(3800, sr/2 - 1)

    # Pre-emphasis (light high-frequency boost). 0.0 disables.
    pre_emphasis: float = 0.97

    # Trim silence
    trim_top_db: float = 30.0

    # Feature options
    use_deltas: bool = True
    use_delta_deltas: bool = True

    # Log-mel power -> dB floor
    amin: float = 1e-10
    ref: float = 1.0
    top_db: float = 80.0


def _pre_emphasize(y: np.ndarray, coeff: float) -> np.ndarray:
    if coeff is None or coeff <= 0.0:
        return y
    # y'[0] = y[0], y'[t] = y[t] - a * y[t-1]
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out


def _logmel(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    # silence trim (energy-based)
    y_trim, _ = librosa.effects.trim(
        y, top_db=cfg.trim_top_db,
        frame_length=cfg.win_length,
        hop_length=cfg.hop_length
    )
    if y_trim.size == 0:
        y_trim = y  # fallback if trimming nuked everything

    # resample if needed
    if sr != cfg.target_sr:
        y_trim = librosa.resample(y_trim, orig_sr=sr, target_sr=cfg.target_sr)
        sr = cfg.target_sr

    # light pre-emphasis
    y_trim = _pre_emphasize(y_trim, cfg.pre_emphasis)

    # safe fmax for 8 kHz
    fmax = cfg.fmax
    if fmax is None:
        fmax = min(3800.0, (sr / 2.0) - 1.0)

    # mel spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y_trim,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=fmax,
        power=2.0,
        center=True,
        window="hann",
        pad_mode="reflect",
    )
    # convert to log-dB
    S_db = librosa.power_to_db(S, ref=cfg.ref, amin=cfg.amin, top_db=cfg.top_db)
    return S_db  # shape: (n_mels, T)


def _stats_over_time(F: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Take time-pooled stats from a 2D feature [D x T].
    Returns a 1D vector [2*D] (mean + std) and feature names.
    """
    mean = np.mean(F, axis=1)
    std = np.std(F, axis=1)
    x = np.concatenate([mean, std], axis=0)

    names = [f"mel_mean_{i}" for i in range(F.shape[0])] + \
            [f"mel_std_{i}" for i in range(F.shape[0])]
    return x, names


def _stack_stats(v: np.ndarray, names: List[str],
                 add: np.ndarray, add_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    return np.concatenate([v, add], axis=0), names + add_names


def extract_features(
    X_audio: List[np.ndarray],
    sr_arr: List[int],
    feature_set: str = "logmel",
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[np.ndarray, List[str]]:
    """
    Main entry point used by training/eval.
    Returns:
      X: 2D array [N, D] of utterance-level features
      names: list of D feature names
    """
    if feature_set != "logmel":
        raise ValueError("Only 'logmel' is supported in this feature module.")

    feats = []
    for y, sr in zip(X_audio, sr_arr):
        S_db = _logmel(y, int(sr), cfg)  # [n_mels, T]

        # base stats
        x, names = _stats_over_time(S_db)

        # Δ and ΔΔ over time (then stats)
        if cfg.use_deltas:
            d1 = librosa.feature.delta(S_db, order=1, width=9, mode="nearest")
            d1_stats, d1_names = _stats_over_time(d1)
            d1_names = [n.replace("mel_", "d1_") for n in d1_names]
            x, names = _stack_stats(x, names, d1_stats, d1_names)

        if cfg.use_delta_deltas:
            d2 = librosa.feature.delta(S_db, order=2, width=9, mode="nearest")
            d2_stats, d2_names = _stats_over_time(d2)
            d2_names = [n.replace("mel_", "d2_") for n in d2_names]
            x, names = _stack_stats(x, names, d2_stats, d2_names)

        # append duration (in seconds) — simple but helpful
        duration_sec = S_db.shape[1] * (cfg.hop_length / cfg.target_sr)
        x, names = _stack_stats(x, names,
                                np.array([duration_sec], dtype=np.float32),
                                ["duration_sec"])

        feats.append(x.astype(np.float32))

    X = np.stack(feats, axis=0)  # [N, D]
    return X, names


import numpy as np

def add_noise(y, snr_db=20):
    """
    Add Gaussian noise to a waveform at a given SNR (dB).
    y: audio waveform (numpy array)
    snr_db: desired signal-to-noise ratio in dB
    """
    # Calculate RMS values
    rms_signal = np.sqrt(np.mean(y**2))
    rms_noise = rms_signal / (10**(snr_db / 20))

    # Generate white Gaussian noise
    noise = np.random.normal(0, rms_noise, y.shape[0])

    return y + noise
