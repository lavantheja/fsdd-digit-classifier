# scripts/export_samples.py
import argparse, io, os
from collections import defaultdict

from datasets import load_dataset, Audio
import soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out_dir", default="samples")
    ap.add_argument("--per_class", type=int, default=1,
                    help="How many files to export per digit (0-9)")
    args = ap.parse_args()

    base = os.path.join("free-spoken-digit-dataset", "data")
    file = "train-00000-of-00001.parquet" if args.split == "train" else "test-00000-of-00001.parquet"
    path = os.path.join(base, file)

    # Load Parquet and prevent Audio decoding (no torchcodec needed)
    ds = load_dataset("parquet", data_files=path, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    os.makedirs(args.out_dir, exist_ok=True)

    counts = defaultdict(int)
    for ex in ds:
        lbl = int(ex["label"])
        if counts[lbl] >= args.per_class:
            continue

        # Get WAV bytes; read with soundfile (gets true sample rate from header)
        b = ex["audio"]["bytes"]
        if b is None:
            # Fallback: if bytes are missing (unlikely), skip
            continue

        y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        out = os.path.join(args.out_dir, f"{lbl}_{counts[lbl]}.wav")
        sf.write(out, y, sr)
        counts[lbl] += 1
        print(f"Wrote {out} (sr={sr}, n={len(y)})")

        if sum(counts.values()) >= 10 * args.per_class:
            break

if __name__ == "__main__":
    main()
