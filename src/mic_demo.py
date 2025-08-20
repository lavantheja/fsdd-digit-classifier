from __future__ import annotations
import argparse
import sounddevice as sd
import joblib
from features import extract_features

def record_seconds(seconds: float, sr: int):
    print(f"Recording {seconds:.2f}s at {sr} Hz...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/baseline.joblib")
    parser.add_argument("--sr", type=int, default=8000, help="Record rate; FSDD is 8000 Hz")
    parser.add_argument("--window", type=float, default=1.0, help="Seconds per capture window")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    model_sr = bundle["sr"]
    if args.sr != model_sr:
        print(f"Warning: model trained at {model_sr} Hz, recording at {args.sr} Hz. Will resample.")

    print("Press Ctrl+C to stop.")
    try:
        while True:
            y = record_seconds(args.window, args.sr)
            if args.sr != model_sr:
                import librosa
                y = librosa.resample(y, orig_sr=args.sr, target_sr=model_sr)
            feat = extract_features(y, model_sr).reshape(1, -1)
            pred = model.predict(feat)[0]
            print(f">>> {pred}")
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
