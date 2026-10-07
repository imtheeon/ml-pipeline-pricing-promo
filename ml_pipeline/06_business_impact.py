"""Dollar view of the final exam. Descriptive only: uses the frozen exam predictions, changes no decision."""
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "ml_pipeline")
import guard
from preprocess import FEATURES

sel = json.load(open("ml_pipeline/selected.json"))
model = joblib.load("ml_pipeline/data/fitted_models.joblib")[sel["family"]]
raw = pd.read_parquet("data/clean/lines.parquet")
test = pd.read_pickle("ml_pipeline/data/test.pkl")
keys = ["order_id", "order_date", "sales", "quantity", "discount"]
raw["order_date"] = pd.to_datetime(raw["order_date"])
# join the true profit back by business keys (the split reset the row numbers); unique keys only
test = test.merge(raw[keys + ["profit"]].drop_duplicates(keys, keep=False), on=keys, how="left")
assert len(test) == 1998 and test["profit"].notna().all(), "profit join failed"
assert ((test["profit"] < 0).astype(int) == test["is_loss"]).all(), "joined profit disagrees with the label"
test["flag"] = model.predict_proba(test[FEATURES])[:, 1] >= sel["threshold"]
test["rule"] = test["discount"] > 0.2

def dollars(mask):
    loss = -test.loc[mask & (test["profit"] < 0), "profit"].sum()
    gain = test.loc[mask & (test["profit"] >= 0), "profit"].sum()
    return float(loss), float(gain)

total_loss = -test.loc[test["profit"] < 0, "profit"].sum()
m_loss, m_gain = dollars(test["flag"])
r_loss, r_gain = dollars(test["rule"])
out = {
    "exam_period": [str(test["order_date"].min().date()), str(test["order_date"].max().date())],
    "total_loss_dollars": float(total_loss),
    "model": {"loss_dollars_flagged": m_loss, "share_of_loss_flagged": m_loss / total_loss, "profit_dollars_on_false_alarms": m_gain},
    "rule": {"loss_dollars_flagged": r_loss, "share_of_loss_flagged": r_loss / total_loss, "profit_dollars_on_false_alarms": r_gain},
}
json.dump(out, open("ml_pipeline/business_impact.json", "w"), indent=1)
print(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(7, 4))
labels = ["All losses\nin exam period", "Flagged by\ndiscount rule", "Flagged by\nrandom forest"]
vals = [total_loss, r_loss, m_loss]
ax.bar(labels, vals, color=["#999999", "#7aa6c2", "#1f77b4"])
for i, v in enumerate(vals):
    ax.text(i, v + total_loss * 0.01, f"${v:,.0f}", ha="center")
ax.set_ylabel("dollars lost on money-losing lines")
ax.set_title("Exam period: loss dollars each method would flag for review")
guard.fig(17, "business_impact", fig,
          "Dollars lost on money-losing lines in the final exam period, and how much of that sits on lines each method flags. "
          "Flagging is not saving: it only points reviewers to the lines worth questioning.")
