import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from preprocess import CATEGORICAL, FEATURES, LEAKY, MONEY, NUMERIC, build_features, make_preprocessor

    return CATEGORICAL, FEATURES, LEAKY, MONEY, NUMERIC, build_features, guard, make_preprocessor, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_parquet("data/clean/lines.parquet")
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["is_loss"] = (df["profit"] < 0).astype(int)
    print(df.shape)
    return (df,)


@app.cell
def _(MONEY, df, guard, np, plt):
    # Step 4: cleaning checks (inspection; nothing is removed unless a rule below says so)
    print("missing:", int(df.isna().sum().sum()), "| exact duplicate lines:", int(df.duplicated().sum()))
    print("sales <= 0:", int((df["sales"] <= 0).sum()), "| quantity < 1:", int((df["quantity"] < 1).sum()))
    print("discount range:", df["discount"].min(), "to", df["discount"].max())
    _gap = (df["sales"] - df["gross_sales"] * (1 - df["discount"])).abs().max()
    print("max gap between sales and gross*(1-discount):", round(float(_gap), 4))

    _cols = MONEY[:3] + ["quantity"]
    _share = {}
    for _c in _cols:
        _q1, _q3 = df[_c].quantile([0.25, 0.75])
        _iqr = _q3 - _q1
        _share[_c] = float(((df[_c] < _q1 - 1.5 * _iqr) | (df[_c] > _q3 + 1.5 * _iqr)).mean() * 100)
    print({k: round(v, 1) for k, v in _share.items()})
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.barh(list(_share.keys()), list(_share.values()))
    _ax.set_xlabel("share of lines flagged as unusually large (%)")
    _ax.set_title("Outlier share by money / size column")
    guard.fig(
        4,
        "outliers",
        _fig,
        "Share of lines with unusually large dollar values or quantities. These are genuine big orders, "
        "not errors, so they are kept; a log transform tames them for the linear model.",
    )
    return


@app.cell
def _(FEATURES, LEAKY, build_features, df):
    # Step 5: one row per order line, features known at pricing time only
    feat = build_features(df)
    model_df = feat.assign(order_id=df["order_id"], order_date=df["order_date"], is_loss=df["is_loss"])
    leaked = [c for c in LEAKY if c in model_df.columns]
    assert not leaked, f"leaky columns present: {leaked}"
    assert list(feat.columns) == FEATURES
    print("model table:", model_df.shape, "| feature count:", len(FEATURES), "| leaky columns present:", leaked)
    _bad = feat.select_dtypes("number").replace([float("inf"), float("-inf")], float("nan")).isna().any().any()
    print("any infinite or missing numeric feature values:", bool(_bad))
    return (model_df,)


@app.cell
def _(guard, model_df, plt):
    # Step 6: chronological split (guard.split drops exact duplicates and freezes the final-exam rows)
    train, val, test = guard.split(model_df, target="is_loss", time_col="order_date")
    for _name, _p in [("train", train), ("val", val), ("test", test)]:
        print(
            _name, len(_p), "rows |", _p["order_date"].min().date(), "to", _p["order_date"].max().date(),
            "| loss share", round(float(_p["is_loss"].mean()), 3),
        )
    print(
        "orders in two splits:",
        len(set(train["order_id"]) & set(val["order_id"])),
        len(set(val["order_id"]) & set(test["order_id"])),
    )

    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    for _name, _p, _col in [("train", train, "tab:blue"), ("validation", val, "tab:orange"), ("final exam", test, "tab:green")]:
        _m = _p.groupby(_p["order_date"].dt.to_period("M")).size()
        _ax.bar(_m.index.to_timestamp(), _m.values, width=25, color=_col, label=f"{_name} ({len(_p):,} lines)")
    _ax.set_title("Lines per month, colored by split")
    _ax.legend()
    guard.fig(
        6,
        "split_timeline",
        _fig,
        "The chronological split: train on the earliest lines, tune on the next, and keep the latest period "
        "locked as the final exam. The loss share is about the same in each part, but the exam period is the busy season.",
    )
    train.to_pickle("ml_pipeline/data/train.pkl")
    val.to_pickle("ml_pipeline/data/val.pkl")
    test.to_pickle("ml_pipeline/data/test.pkl")
    return train, val


@app.cell
def _(FEATURES, train):
    # Step 7: the engineered features are row-wise (month, price per unit, discount flags); nothing learned from the target.
    print(train[["discount_over_20", "is_discounted", "list_price_per_unit", "month_of_year"]].describe().round(2).to_string())
    print("train correlation of discount_over_20 with loss:", round(float(train["discount_over_20"].corr(train["is_loss"])), 3))
    print("features used:", len(FEATURES))
    return


@app.cell
def _(CATEGORICAL, FEATURES, guard, make_preprocessor, plt, train, val):
    # Step 8: preprocessing, fit on TRAIN only
    prep = make_preprocessor()
    prep.fit(train[FEATURES])
    _xt = prep.transform(train[FEATURES])
    _xv = prep.transform(val[FEATURES])
    print("columns after preprocessing:", _xt.shape[1])
    print("train mean of numeric columns ~", round(float(_xt[:, :9].mean()), 3), "| val mean ~", round(float(_xv[:, :9].mean()), 3))
    for _c in CATEGORICAL:
        _unseen = sorted(set(val[_c]) - set(train[_c]))
        if _unseen:
            print("values in validation never seen in train for", _c, ":", _unseen)

    _fig, _axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    _axes[0].hist(train["sales"], bins=40)
    _axes[0].set_title("sales, raw")
    _axes[1].hist(_xt[:, 0], bins=40)
    _axes[1].set_title("sales, log then scaled")
    guard.fig(
        8,
        "log_scaling",
        _fig,
        "Sales has a long right tail. Taking the log and scaling (using training rows only) spreads it out so "
        "a few huge orders do not dominate the linear model.",
    )
    return


if __name__ == "__main__":
    app.run()
