# src/eval.py
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib

from data import load_fsdd_local_parquet, dataset_to_arrays
from features import extract_features

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--metrics_out", default="artifacts/test_metrics.json")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Load data
    ds = load_fsdd_local_parquet(split=args.split, include_test_in_train=False)
    X_audio, y, sr_arr, _ = dataset_to_arrays(ds)

    # 2) Features (batch)
    X, feat_names = extract_features(X_audio=X_audio, sr_arr=sr_arr)

    # 3) Load predict-able model/pipeline
    model = joblib.load(args.model_path)

    # 4) Predict + metrics
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy on {args.split}: {acc:.4f}\n")
    print(classification_report(y, y_pred, digits=4))

    # 5) Save metrics
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "accuracy": float(acc)}, indent=2))
    print(f"Wrote metrics to {out.resolve()}")

if __name__ == "__main__":
    main()
