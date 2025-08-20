from datasets import load_dataset, Features, Value
from pathlib import Path

def main():
    data_dir = Path("free-spoken-digit-dataset/data")
    train_path = data_dir / "train-00000-of-00001.parquet"

    # Explicitly override schema: audio = string (path), label = int
    features = Features({
        "audio": {
            "bytes": Value("binary"),
            "path": Value("string")
        },
        "label": Value("int64")
    })

    ds = load_dataset(
        "parquet",
        data_files=str(train_path),
        split="train",
        features=features
    )

    print("Dataset columns:", ds.column_names)
    print("First row (raw):", ds[0])   # now should be a plain dict, no decoding

if __name__ == "__main__":
    main()
