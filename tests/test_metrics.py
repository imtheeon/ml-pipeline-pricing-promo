import numpy as np

from metrics import evaluate


def test_perfect_scores_reach_full_recall():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    out = evaluate(y, p)
    assert out["pr_auc"] == 1.0
    assert out["recall_at_p90"] == 1.0


def test_random_scores_sit_near_the_loss_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.2).astype(int)
    out = evaluate(y, rng.random(20_000))
    assert abs(out["pr_auc"] - 0.2) < 0.02  # PR-AUC of a coin flip is the base rate, not 0.5


def test_counts_at_threshold_add_up():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.8, 0.1, 0.7, 0.2])
    out = evaluate(y, p, threshold=0.5)
    assert out["caught"] + out["missed"] == y.sum()
    assert out["flagged"] == out["caught"] + out["false_alarms"]
