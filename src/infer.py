from __future__ import annotations
import argparse
import soundfile as sf
import joblib
from features import extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str)
    parser.add_argument("--model_path", type=str, default="../models/baseline.joblib")
    args = parser.parse_args()

    y, sr = sf.read(args.wav_path, dtype="float32", always_2d=False)
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    sr_model = bundle["sr"]
    if sr != sr_model:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_model)
        sr = sr_model

    feat = extract_features(y, sr).reshape(1, -1)
    pred = model.predict(feat)[0]
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        conf = proba(feat)[0][pred]
        print(f"Predicted: {pred}  (p={conf:.2f})")
    else:
        print(f"Predicted: {pred}")

if __name__ == "__main__":
    main()
