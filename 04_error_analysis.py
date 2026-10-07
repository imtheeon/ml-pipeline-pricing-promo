import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from preprocess import FEATURES

    sel = json.load(open("ml_pipeline/selected.json"))
    model = joblib.load("ml_pipeline/data/fitted_models.joblib")[sel["family"]]
    val = pd.read_pickle("ml_pipeline/data/val.pkl")
    val["p"] = model.predict_proba(val[FEATURES])[:, 1]
    val["flag"] = (val["p"] >= sel["threshold"]).astype(int)
    val["band"] = pd.cut(val["discount"], [-0.001, 0, 0.2, 0.4, 1], labels=["0%", "1-20%", "21-40%", ">40%"])
    return guard, np, pd, plt, sel, val


@app.cell
def _(guard, pd, plt, val):
    _g = val.groupby("band", observed=True).apply(
        lambda d: pd.Series({
            "lines": len(d), "losses": int(d["is_loss"].sum()),
            "caught": int(((d["flag"] == 1) & (d["is_loss"] == 1)).sum()),
            "missed": int(((d["flag"] == 0) & (d["is_loss"] == 1)).sum()),
            "false_alarms": int(((d["flag"] == 1) & (d["is_loss"] == 0)).sum()),
        }), include_groups=False)
    print(_g.to_string())
    _g.to_csv("ml_pipeline/error_by_band.csv")
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _x = range(len(_g))
    _ax.bar(_x, _g["caught"], label="caught")
    _ax.bar(_x, _g["missed"], bottom=_g["caught"], label="missed")
    _ax.set_xticks(list(_x)); _ax.set_xticklabels(_g.index)
    _ax.set_ylabel("money-losing lines"); _ax.legend(); _ax.set_title("Validation: losses caught and missed, by discount band")
    guard.fig(13, "error_analysis", _fig,
              "Where the selected model still misses losses. All 63 missed losses sit in the 1-20% discount band, the group "
              "the simple rule cannot see, so that is where any gain over the rule has to come from.")
    return


@app.cell
def _(np, val):
    # calibration: do scores mean what they say?
    _b = pd_cut = None
    import pandas as _pd
    _q = _pd.cut(val["p"], [0, .1, .3, .5, .7, .9, 1.0001], include_lowest=True)
    print(val.groupby(_q, observed=True)["is_loss"].agg(["mean", "size"]).round(3).to_string())
    _m = val[(val["band"] == "1-20%")]
    print("1-20% band: losses", int(_m["is_loss"].sum()), "of", len(_m), "| caught", int(((_m["flag"] == 1) & (_m["is_loss"] == 1)).sum()))
    _miss = val[(val["flag"] == 0) & (val["is_loss"] == 1)]
    print("missed by category:", _miss["category"].value_counts().to_dict())
    print("missed by sub_category (top 5):", _miss["sub_category"].value_counts().head(5).to_dict())
    return


if __name__ == "__main__":
    app.run()
