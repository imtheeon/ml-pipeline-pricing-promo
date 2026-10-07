import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys
    import time

    import matplotlib

    matplotlib.use("Agg")
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    sys.path.insert(0, "ml_pipeline")
    import guard
    from metrics import evaluate
    from models import DummyModel, EarlyStoppedXGB, RuleModel, forest, logistic
    from preprocess import FEATURES

    train = pd.read_pickle("ml_pipeline/data/train.pkl").sort_values("order_date").reset_index(drop=True)
    val = pd.read_pickle("ml_pipeline/data/val.pkl").sort_values("order_date").reset_index(drop=True)
    ytr, yva = train["is_loss"].to_numpy(), val["is_loss"].to_numpy()
    print(len(train), len(val))
    return (DummyModel, EarlyStoppedXGB, FEATURES, RuleModel, TimeSeriesSplit, clone, evaluate, forest, guard,
            joblib, json, logistic, np, pd, plt, time, train, val, ytr, yva)


@app.cell
def _(DummyModel, FEATURES, RuleModel, evaluate, pd, train, val, ytr, yva):
    # Step 9: baselines on validation (nothing is fit, or only the rule is applied)
    rows = []
    for _name, _m in [("dummy (never flags)", DummyModel()), ("rule: discount > 20%", RuleModel())]:
        _p = _m.fit(train[FEATURES], ytr).predict_proba(val[FEATURES])[:, 1]
        _r = evaluate(yva, _p, threshold=0.5)
        _r.update(model=_name, family="baseline")
        rows.append(_r)
    base = pd.DataFrame(rows)
    print(base[["model", "pr_auc", "recall_at_p90", "precision_at_threshold", "recall_at_threshold", "flagged", "caught", "missed"]].round(3).to_string())
    return (rows,)


@app.cell
def _(EarlyStoppedXGB, FEATURES, TimeSeriesSplit, clone, evaluate, forest, logistic, np, pd, time, train, ytr):
    # Steps 10-11: forward-chaining CV on TRAIN only to pick each family's settings
    grids = {
        "logistic": [(f"C={c}", logistic(C=c)) for c in (0.1, 1, 10)],
        "random forest": [(f"min_leaf={m}", forest(min_samples_leaf=m)) for m in (1, 3, 5)],
        "xgboost": [(f"depth={d}, lr={lr}", EarlyStoppedXGB(max_depth=d, learning_rate=lr)) for d in (3, 5) for lr in (0.05, 0.1)],
    }
    tscv = TimeSeriesSplit(n_splits=4)
    cv_rows = []
    X = train[FEATURES]
    for _fam, _cands in grids.items():
        for _label, _m in _cands:
            _t = time.time()
            _scores = []
            for _a, _b in tscv.split(X):
                _mm = clone(_m).fit(X.iloc[_a], ytr[_a])
                _scores.append(evaluate(ytr[_b], _mm.predict_proba(X.iloc[_b])[:, 1])["pr_auc"])
            cv_rows.append(dict(family=_fam, setting=_label, cv_pr_auc=float(np.mean(_scores)), cv_sd=float(np.std(_scores)), secs=round(time.time() - _t, 1)))
            print(cv_rows[-1])
    cv = pd.DataFrame(cv_rows)
    best = {f: g.sort_values("cv_pr_auc", ascending=False).iloc[0]["setting"] for f, g in cv.groupby("family")}
    print(best)
    return best, cv, grids


@app.cell
def _(FEATURES, best, clone, evaluate, grids, joblib, pd, rows, train, val, ytr, yva):
    # Step 12: fit the chosen setting of each family on all of train, score on validation
    fitted = {}
    for _fam, _cands in grids.items():
        _m = dict(_cands)[best[_fam]]
        _m = clone(_m).fit(train[FEATURES], ytr)
        fitted[_fam] = _m
        _p = _m.predict_proba(val[FEATURES])[:, 1]
        _r = evaluate(yva, _p)
        _r = evaluate(yva, _p, threshold=_r["threshold_p90"]) if _r["threshold_p90"] is not None else _r
        _r.update(model=f"{_fam} ({best[_fam]})", family=_fam)
        rows.append(_r)
        if hasattr(_m, "best_trees_"):
            print("xgboost trees kept by early stopping:", _m.best_trees_)
    res = pd.DataFrame(rows)
    print(res[["model", "pr_auc", "recall_at_p90", "threshold_p90", "precision_at_threshold", "recall_at_threshold", "flagged", "caught", "missed", "false_alarms"]].round(3).to_string())
    res.to_csv("ml_pipeline/runs.csv", index=False)
    joblib.dump(fitted, "ml_pipeline/data/fitted_models.joblib")
    return fitted, res


@app.cell
def _(guard, plt, res):
    _m = res.set_index("model")
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].barh(_m.index, _m["pr_auc"])
    _ax[0].set_title("Validation PR-AUC (higher is better)")
    _ax[1].barh(_m.index, _m["recall_at_p90"])
    _ax[1].set_title("Share of losses caught while flags are 90% right")
    _ax[1].set_yticklabels([])
    guard.fig(
        10,
        "model_comparison",
        _fig,
        "Each model scored on the validation period. The left bars rank every cut-off at once; the right bars show how many "
        "money-losing lines each model finds if we only accept flags that are right 90% of the time.",
    )
    return


@app.cell
def _(FEATURES, fitted, guard, np, plt, val, yva):
    from sklearn.metrics import precision_recall_curve

    _fig, _ax = plt.subplots(figsize=(6.5, 4.5))
    for _name, _m in fitted.items():
        _pr, _rc, _ = precision_recall_curve(yva, _m.predict_proba(val[FEATURES])[:, 1])
        _ax.plot(_rc, _pr, label=_name)
    _ax.axhline(0.9, ls="--", color="gray")
    _ax.set_xlabel("recall (share of losses found)")
    _ax.set_ylabel("precision (flags that are right)")
    _ax.legend()
    _ax.set_title("Precision vs recall on validation")
    guard.fig(
        12,
        "evaluation",
        _fig,
        "Curves show the trade-off: the further a line stays up and to the right, the more losses it finds with fewer wrong flags. "
        "The dashed line is the 90% precision bar we set in advance.",
    )
    return


@app.cell
def _(cv, json, res):
    # Apply the pre-declared selection rule: highest validation PR-AUC; within 0.005 the simpler model wins
    _order = {"logistic": 0, "random forest": 1, "xgboost": 2}
    _m = res[res["family"].isin(_order)].copy()
    _top = _m["pr_auc"].max()
    _close = _m[_m["pr_auc"] >= _top - 0.005].copy()
    _close["rank"] = _close["family"].map(_order)
    _win = _close.sort_values("rank").iloc[0]
    print("selected:", _win["model"], "| val pr_auc", round(_win["pr_auc"], 4), "| top", round(_top, 4), "| within-0.005 candidates:", list(_close["model"]))
    sel = {"family": _win["family"], "model": _win["model"], "val_pr_auc": float(_win["pr_auc"]), "threshold": float(_win["threshold_p90"]), "top_pr_auc": float(_top), "close": list(_close["model"])}
    json.dump(sel, open("ml_pipeline/selected.json", "w"), indent=1)
    cv.to_csv("ml_pipeline/cv_results.csv", index=False)
    return (sel,)


if __name__ == "__main__":
    app.run()
