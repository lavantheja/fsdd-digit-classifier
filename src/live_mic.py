# src/live_mic.py
from __future__ import annotations
import argparse
import time
from pathlib import Path
from fractions import Fraction
from collections import deque, Counter

import numpy as np
import sounddevice as sd
from joblib import load as joblib_load
import scipy.signal as sps

from features import extract_features


def load_bundle(path: str | Path):
    obj = joblib_load(path)
    # We saved a dict in train.py: {"model": final_model, "sr": sr, "feature_conf": {...}}
    if isinstance(obj, dict) and "model" in obj:
        return obj
    # Fallback if someone saved a bare estimator
    return {"model": obj, "sr": 8000, "feature_conf": {}}


def db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def topk_from_proba(probs: np.ndarray, k: int = 3):
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


def resample_audio(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    frac = Fraction(sr_out, sr_in).limit_denominator()
    up, down = frac.numerator, frac.denominator
    return sps.resample_poly(y, up, down)


def stable_decision(history: deque[tuple[int, float]], min_conf: float, smooth: int) -> tuple[bool, int, float]:
    """
    Majority vote over the last `smooth` frames; if the winning class's
    mean confidence >= min_conf, return (True, digit, mean_conf). Otherwise (False, _, _).
    """
    if len(history) < smooth:
        return False, -1, 0.0
    recent = list(history)[-smooth:]
    labels = [d for d, _ in recent]
    counts = Counter(labels)
    winner, _ = counts.most_common(1)[0]
    confs = [c for d, c in recent if d == winner]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return (mean_conf >= min_conf, winner, mean_conf)


def run_stream(args, model, model_sr):
    want_len = int(round(args.block_dur * args.samplerate))
    print(f"Mic SR={args.samplerate} Hz, model SR={model_sr} Hz, block={args.block_dur:.2f}s ({want_len} samples)")

    # Simple rolling history for smoothing
    hist: deque[tuple[int, float]] = deque(maxlen=max(1, args.smooth))

    stream = sd.InputStream(
        samplerate=args.samplerate,
        channels=1,
        dtype="float32",
        blocksize=want_len,
        device=args.device,
    )

    try:
        stream.start()
        while True:
            t0 = time.perf_counter()
            in_audio, _ = stream.read(want_len)  # (N, 1)
            y = in_audio.squeeze().astype(np.float32) * args.gain

            # Skip very quiet frames (simple VAD)
            level_db = db(y)
            if level_db < args.vad_db:
                t_total = (time.perf_counter() - t0) * 1000.0
                if args.verbose:
                    print(f"[silence {level_db:.1f} dB] total:{t_total:.1f}ms")
                if args.pause_after > 0:
                    time.sleep(args.pause_after)
                continue

            # Resample to model SR
            y_rs = resample_audio(y, args.samplerate, model_sr)

            # Features
            t_feat0 = time.perf_counter()
            X_mat, _ = extract_features(X_audio=[y_rs], sr_arr=[model_sr])  # (1, D)
            t_feat = (time.perf_counter() - t_feat0) * 1000.0

            # Predict
            t_inf0 = time.perf_counter()
            probs = model.predict_proba(X_mat)[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            t_inf = (time.perf_counter() - t_inf0) * 1000.0

            hist.append((pred, conf))
            tops = topk_from_proba(probs, k=args.topk)
            tops_str = " ".join([f"{d}:{p:.2f}" for d, p in tops])
            t_total = (time.perf_counter() - t0) * 1000.0
            print(f"Digit={pred} | {tops_str} | feat:{t_feat:.1f}ms infer:{t_inf:.1f}ms total:{t_total:.1f}ms")

            decided, digit, mean_conf = stable_decision(hist, args.min_conf, args.smooth)
            if decided:
                print(f"DETECTED: {digit} (mean_conf over last {args.smooth} frames = {mean_conf:.2f})")
                if args.one_shot:
                    return
                # small pause to avoid spamming repeated detections
                if args.pause_after > 0:
                    time.sleep(args.pause_after)

    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass


def run_push_to_talk(args, model, model_sr):
    want_len = int(round(args.block_dur * args.samplerate))
    print(f"[Push-to-talk] Press Enter to capture ~{args.block_dur:.2f}s and predict once. (Ctrl+C to quit)")
    input("Ready? Press Enter...")
    in_audio = sd.rec(want_len, samplerate=args.samplerate, channels=1, dtype="float32")
    sd.wait()
    y = in_audio.squeeze().astype(np.float32) * args.gain

    if db(y) < args.vad_db:
        print("Too quiet / silence detected. Try again closer to the mic.")
        return

    y_rs = resample_audio(y, args.samplerate, model_sr)
    X_mat, _ = extract_features(X_audio=[y_rs], sr_arr=[model_sr])
    probs = model.predict_proba(X_mat)[0]
    pred = int(np.argmax(probs))
    tops = topk_from_proba(probs, k=args.topk)
    tops_str = " ".join([f"{d}:{p:.2f}" for d, p in tops])
    print(f"Digit={pred} | {tops_str}")


def main():
    parser = argparse.ArgumentParser(description="Live microphone digit predictor")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--samplerate", type=int, default=16000, help="Mic sampling rate (Hz)")
    parser.add_argument("--block_dur", type=float, default=0.9, help="Seconds per inference block")
    parser.add_argument("--device", type=str, default=None, help="sounddevice input device name/index (optional)")
    parser.add_argument("--gain", type=float, default=1.0, help="Post-capture gain multiplier")
    parser.add_argument("--vad_db", type=float, default=-60.0, help="Very simple VAD threshold; skip below this")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--one_shot", action="store_true", help="Exit after one stable & confident detection")
    parser.add_argument("--min_conf", type=float, default=0.85, help="Confidence required to accept detection")
    parser.add_argument("--smooth", type=int, default=3, help="Majority vote over last K frames")
    parser.add_argument("--pause_after", type=float, default=0.4, help="Pause after a detection (seconds)")
    parser.add_argument("--push_to_talk", action="store_true", help="Capture one block on Enter and predict once")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bundle = load_bundle(args.model_path)
    model = bundle["model"]
    model_sr = int(bundle.get("sr", 8000))

    try:
        if args.push_to_talk:
            run_push_to_talk(args, model, model_sr)
        else:
            run_stream(args, model, model_sr)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")


if __name__ == "__main__":
    main()
