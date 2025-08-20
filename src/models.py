from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

@dataclass
class ModelConfig:
    algo: Literal["logreg", "svm"] = "logreg"
    C: float = 2.0

def make_baseline_model(cfg: ModelConfig) -> Pipeline:
    if cfg.algo == "logreg":
        clf = LogisticRegression(max_iter=2000, C=cfg.C, n_jobs=None, multi_class="auto")
    elif cfg.algo == "svm":
        clf = SVC(C=cfg.C, kernel="rbf", gamma="scale", probability=True)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
