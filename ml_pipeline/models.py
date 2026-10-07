"""Model builders. Every model is preprocessing + classifier in one object, fit on train only."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import FEATURES, make_preprocessor

XGB_DEFAULTS = dict(
    n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="aucpr", random_state=0, n_jobs=2,
)


def build(clf):
    return Pipeline([("prep", make_preprocessor()), ("clf", clf)])


def logistic(C=1.0):
    return build(LogisticRegression(C=C, max_iter=2000))


def forest(min_samples_leaf=1):
    return build(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=min_samples_leaf, random_state=0, n_jobs=2))


class RuleModel:
    """The repo's own rule: discount above 20% = loss. No fitting."""
    classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = (X["discount"].to_numpy() > 0.2).astype(float)
        return np.c_[1 - p, p]


class DummyModel(RuleModel):
    """Never flags a loss."""

    def predict_proba(self, X):
        n = len(X)
        return np.c_[np.ones(n), np.zeros(n)]


class EarlyStoppedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost with early stopping on the LAST 15% (by time) of the rows it is given.

    Rows must arrive in date order. The caller's validation set is never touched.
    """

    def __init__(self, max_depth=4, learning_rate=0.05, holdout_frac=0.15, rounds=30):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.holdout_frac = holdout_frac
        self.rounds = rounds

    def fit(self, X, y):
        y = np.asarray(y)
        self.prep_ = make_preprocessor()
        self.prep_.fit(X[FEATURES])
        Z = self.prep_.transform(X[FEATURES])
        cut = int(len(Z) * (1 - self.holdout_frac))
        params = dict(XGB_DEFAULTS, max_depth=self.max_depth, learning_rate=self.learning_rate)
        self.clf_ = XGBClassifier(early_stopping_rounds=self.rounds, **params)
        self.clf_.fit(Z[:cut], y[:cut], eval_set=[(Z[cut:], y[cut:])], verbose=False)
        self.best_trees_ = int(self.clf_.best_iteration) + 1
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.prep_.transform(X[FEATURES]))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
