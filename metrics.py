"""Scoring helpers. The positive class is is_loss = 1 (the line loses money)."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

MIN_PRECISION = 0.90


def evaluate(y, p, threshold=None) -> dict:
    """PR-AUC, best recall while precision stays >= 0.90, and optional precision/recall at a fixed threshold."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out = {"pr_auc": float(average_precision_score(y, p))}
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= MIN_PRECISION
    if ok.any():
        idx = np.where(ok)[0][np.argmax(rec[:-1][ok])]
        out["recall_at_p90"] = float(rec[idx])
        out["threshold_p90"] = float(thr[idx])
    else:
        out["recall_at_p90"] = 0.0
        out["threshold_p90"] = None
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        out["precision_at_threshold"] = float(precision_score(y, pred, zero_division=0))
        out["recall_at_threshold"] = float(recall_score(y, pred, zero_division=0))
        out["flagged"] = int(pred.sum())
        out["caught"] = int(((pred == 1) & (y == 1)).sum())
        out["missed"] = int(((pred == 0) & (y == 1)).sum())
        out["false_alarms"] = int(((pred == 1) & (y == 0)).sum())
    return out
