import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard

    return guard, json, pd, plt


@app.cell
def _(pd):
    # Step 1: load the repo's cleaned data (read-only) and define the label we may predict
    df = pd.read_parquet("data/clean/lines.parquet")
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["is_loss"] = (df["profit"] < 0).astype(int)  # 1 = this line lost money
    print(df.shape)
    print("loss share:", round(float(df["is_loss"].mean()), 4))
    print("date range:", df["order_date"].min().date(), "to", df["order_date"].max().date())
    print("orders:", df["order_id"].nunique(), "| customers:", df["customer_id"].nunique(), "| products:", df["product_id"].nunique())
    print("lines per order (mean):", round(len(df) / df["order_id"].nunique(), 2))
    return (df,)


@app.cell
def _(df, guard, json):
    prof = guard.profile(df, target="is_loss", time_col="order_date")
    print(json.dumps(prof["shape"]))
    print(json.dumps(prof["target"]))
    print("duplicates:", prof["duplicates"])
    print("constant columns:", prof["constant_columns"])
    print("traits:", json.dumps(prof["traits"]))
    print("leakage suspects:", json.dumps(prof["leakage_suspects"], indent=1))
    print("missing total:", int(df.isna().sum().sum()))
    return


@app.cell
def _(df, guard, plt):
    # Step 2: required figures plus one that previews the repo's main question
    out = guard.eda_figures(df, target="is_loss", time_col="order_date")
    print([p.name for p in out])

    _order = ["0%", "1-20%", "21-40%", ">40%"]
    _g = df.groupby("discount_band")["is_loss"].agg(["mean", "size"]).loc[_order]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.bar(_g.index, _g["mean"] * 100)
    for _i, (_m, _n) in enumerate(zip(_g["mean"], _g["size"])):
        _ax.text(_i, _m * 100 + 1, f"{_m*100:.0f}%\n(n={_n})", ha="center", fontsize=9)
    _ax.set_ylim(0, 115)
    _ax.set_ylabel("share of lines that lose money (%)")
    _ax.set_title("Loss rate by discount band")
    guard.fig(
        2,
        "loss_by_discount",
        _fig,
        "Share of order lines that lost money in each discount band. Losses are rare at 0-20% off and "
        "nearly universal above 20%, so discount depth alone already separates most losses; any model has to beat this simple rule.",
    )
    print(_g.round(3).to_string())
    return


if __name__ == "__main__":
    app.run()
