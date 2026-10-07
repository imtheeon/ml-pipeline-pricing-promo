import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import joblib
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from metrics import evaluate
    from models import DummyModel, RuleModel
    from preprocess import FEATURES

    sel = json.load(open("ml_pipeline/selected.json"))
    fitted = joblib.load("ml_pipeline/data/fitted_models.joblib")  # all fit on train only
    test = pd.read_pickle("ml_pipeline/data/test.pkl")
    return DummyModel, FEATURES, RuleModel, evaluate, fitted, guard, json, sel, test


@app.cell
def _(DummyModel, FEATURES, RuleModel, evaluate, fitted, guard, json, sel, test):
    # ONE pass over the locked exam rows. Every pre-declared predictor is scored inside this single call.
    allp = {"dummy": DummyModel(), "rule": RuleModel(), **fitted}
    store = {}

    def predict_fn(X):
        for _n, _m in allp.items():
            store[_n] = _m.predict_proba(X[FEATURES])[:, 1]
        return store[sel["family"]]

    def pr_auc(y, p):
        return evaluate(y, p)["pr_auc"]

    score = guard.final_test(predict_fn, test, target="is_loss", metric_fn=pr_auc)
    y = test["is_loss"].to_numpy()
    out = {}
    for _n, _p in store.items():
        out[_n] = evaluate(y, _p, threshold=(sel["threshold"] if _n == sel["family"] else 0.5))
    print("selected:", sel["model"], "exam PR-AUC", round(score, 4))
    for _n, _r in out.items():
        print(_n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _r.items() if k != "threshold_p90"})
    json.dump({"selected": sel["model"], "threshold_from_validation": sel["threshold"], "results": out}, open("ml_pipeline/final_results.json", "w"), indent=1)
    return


if __name__ == "__main__":
    app.run()
