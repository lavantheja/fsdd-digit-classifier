# src/train.py
from __future__ import annotations
import argparse, json
import numpy as np
from joblib import dump
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from features import extract_features, add_noise
# OLD: from data import load_fsdd_hf, dataset_to_arrays
from data import load_fsdd_local_parquet, dataset_to_arrays
from models import make_baseline_model, ModelConfig
from utils import ensure_dirs, MODELS_DIR, ARTIFACTS_DIR

def main():
    print("Stay tuned...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["logreg","svm"], default="logreg")
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--snr_db", type=float, default=None, help="Optional noise augmentation during training")
    parser.add_argument("--model_name", type=str, default="baseline.joblib")
    parser.add_argument("--split", type=str, default="train", help="train or test (train=default)")
    parser.add_argument("--include_test_in_train", action="store_true",
                        help="If set and split=train, concatenate test into training data (more total clips).")
    args = parser.parse_args()

    ensure_dirs()

    # ---- Load local Parquet (offline) ---------------------------------------
    ds = load_fsdd_local_parquet(split=args.split, include_test_in_train=args.include_test_in_train)
    X_audio, y, sr_arr, speakers = dataset_to_arrays(ds)
    sr = int(sr_arr[0])


# ---- Feature extraction --------------------------------------------------
# Build an augmented list first (noise if requested), then extract in one call
    X_audio_aug = []
    for wav in X_audio:
        if args.snr_db is not None:
            wav = add_noise(wav, args.snr_db)
        X_audio_aug.append(wav)

    # extract_features expects (list_of_waveforms, list_of_sample_rates)
    X, feat_names = extract_features(X_audio=X_audio_aug, sr_arr=sr_arr)
    # X is a 2D numpy array [N, D]



    # ---- Group K-Fold by speaker --------------------------------------------
    groups = np.array(speakers)
    n_splits = min(5, len(np.unique(groups)))
    n_splits = max(n_splits, 2)
    gkf = GroupKFold(n_splits=n_splits)
    accs = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        model = make_baseline_model(ModelConfig(algo=args.algo, C=args.C))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        acc = accuracy_score(y[va], pred)
        accs.append(float(acc))
        print(f"Fold {fold} acc: {acc:.4f}")

    print(f"Mean GKF acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ---- Final model on all data --------------------------------------------
    final_model = make_baseline_model(ModelConfig(algo=args.algo, C=args.C))
    final_model.fit(X, y)

    out_path = MODELS_DIR / args.model_name
    # Save just the fitted estimator/pipeline (predict-able object)
    dump(final_model, out_path)
    print(f"Saved model to {out_path}")

    metrics_path = ARTIFACTS_DIR / "cv_metrics.json"
    metrics_path.write_text(json.dumps(
        {"fold_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))},
        indent=2
    ))
    print(f"Wrote CV metrics to {metrics_path}")


if __name__ == "__main__":
    main()
