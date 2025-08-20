import numpy as np
from src.features import extract_features

sr = 8000
t = np.linspace(0, 1, sr, endpoint=False)
y = 0.1*np.sin(2*np.pi*440*t).astype("float32")

f = extract_features(y, sr)
print("Feature vector length:", f.shape[0])  # expect 160
