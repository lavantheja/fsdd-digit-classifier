from __future__ import annotations
import argparse, time, sys
import numpy as np
import soundfile as sf
import librosa
from joblib import load
from features import extract_features, add_noise

def load_model_bundle(path: str):
    obj = load(path)
    # Dict bundle (new train.py)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], int(obj.get("sr", 8000)), obj.get("feature_conf", {})
    # Plain estimator (old save)
    return obj, 8000, {}  # default SR if not stored

def _maybe_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="polyphase")

def _predict_proba_safely(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        if z.ndim == 1:
            z = np.stack([-z, z], axis=1)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)
    y_pred = model.predict(X)
    n_classes = int(np.max(y_pred)) + 1
    p = np.zeros((len(y_pred), n_classes), dtype=float)
    p[np.arange(len(y_pred)), y_pred] = 1.0
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--wav_path", required=True)
    ap.add_argument("--snr_db", type=float, default=None)
    args = ap.parse_args()

    model, target_sr, _ = load_model_bundle(args.model_path)

    t0 = time.perf_counter()
    y, sr = sf.read(args.wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = _maybe_resample(y, sr, target_sr)
    if args.snr_db is not None:
        y = add_noise(y, args.snr_db)
    t_load = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    X, feat_names = extract_features([y], [target_sr])
    t_feat = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    proba = _predict_proba_safely(model, X)[0]
    pred = int(np.argmax(proba))
    t_pred = (time.perf_counter() - t2) * 1000

    top_idx = np.argsort(proba)[::-1][:3]
    print(f"Predicted digit: {pred}")
    print("Top-3 probs:", ", ".join([f"{d}:{proba[d]:.3f}" for d in top_idx]))
    print(f"Latency ms — load:{t_load:.1f} feat:{t_feat:.1f} infer:{t_pred:.1f} total:{t_load + t_feat + t_pred:.1f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
