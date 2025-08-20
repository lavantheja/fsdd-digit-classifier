import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_fsdd_hf, dataset_to_arrays


ds = load_fsdd_hf("train")
print("First few raw paths:", [ds[i]["audio"]["path"] for i in range(5)])
Xa, y, sr_arr, spk = dataset_to_arrays(ds.select(range(5)))
print("Loaded clips:", len(Xa), "sr:", int(sr_arr[0]))
print("Labels:", y.tolist())
print("Speakers:", spk.tolist()) 
