# src/data.py
import io
from pathlib import Path
import numpy as np
import soundfile as sf
from datasets import load_dataset, Features, Value, concatenate_datasets



DATA_ROOT = Path(__file__).resolve().parents[1] / "free-spoken-digit-dataset" / "data"

# Force a non-Audio schema so 'datasets' doesn't import torchcodec
FSDD_FEATURES = Features({
    "audio": {
        "bytes": Value("binary"),
        "path": Value("string"),
    },
    "label": Value("int64"),
})


def _load_one_parquet(parquet_name: str):
    parquet_path = DATA_ROOT / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
    return load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        features=FSDD_FEATURES,
    )


def load_fsdd_local_parquet(split: str = "train", include_test_in_train: bool = False):
    """
    Load FSDD from the locally cloned mteb parquet files.
    split: "train" or "test"
    include_test_in_train: if True and split=="train", merge test into train
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if split == "train":
        ds_train = _load_one_parquet("train-00000-of-00001.parquet")
        if include_test_in_train:
            ds_test = _load_one_parquet("test-00000-of-00001.parquet")
            ds = concatenate_datasets([ds_train, ds_test])
        else:
            ds = ds_train
    else:  # split == "test"
        ds = _load_one_parquet("test-00000-of-00001.parquet")

    # Give raw python dicts back; no auto decoding
    ds = ds.with_format("python")
    return ds


def dataset_to_arrays(ds):
    """
    Turn dataset into arrays by reading audio from in-memory bytes.
    Returns: X_audio (list of 1D float32 arrays), y (np.int64), sr_arr, speakers (list)
    """
    X_audio, y, sr_arr, speakers = [], [], [], []

    # If you later want a fixed target rate, do resampling in features.py
    for ex in ds:
        audio = ex["audio"]
        label = int(ex["label"])

        # Prefer bytes from parquet; no file system dependency
        wav = None
        sr = None
        if audio and audio.get("bytes") is not None:
            bio = io.BytesIO(audio["bytes"])
            wav, sr = sf.read(bio, dtype="float32", always_2d=False)
        else:
            # Rare fallback: try resolving a filename if present
            path = audio.get("path") if isinstance(audio, dict) else None
            if path:
                from os.path import isabs, join, basename
                # if you ever add recordings/, update this root:
                maybe_rel = join("free-spoken-digit-dataset", "recordings", path)
                real_path = path if isabs(path) else maybe_rel
                wav, sr = sf.read(real_path, dtype="float32", always_2d=False)
            else:
                raise ValueError(f"Unexpected audio format: {audio}")

        X_audio.append(wav.astype("float32", copy=False))
        y.append(label)
        sr_arr.append(sr)

        # Try to infer speaker from filename like "0_george_10.wav"
        spk = "unknown"
        if isinstance(audio, dict) and audio.get("path"):
            name = Path(audio["path"]).name
            parts = name.split("_")
            if len(parts) >= 2:
                spk = parts[1]
        speakers.append(spk)

    return np.array(X_audio, dtype=object), np.array(y), np.array(sr_arr), np.array(speakers)
