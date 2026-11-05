from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {'auc': 0.0, 'acc': 0.0, 'rmse': 0.0, 'num_samples': 0}
    auc = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else float('nan')
    return {'auc': auc, 'acc': float(accuracy_score(y_true, (y_prob >= 0.5).astype(np.int64))), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_prob))), 'num_samples': int(y_true.size)}
