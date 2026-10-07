"""Feature definitions and preprocessing for the pricing-promo loss model.

Every transformation lives in one object so training and prediction can never drift apart.
Always fit on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Dollar-like columns: heavily skewed, so they get log1p before scaling.
MONEY = ["sales", "gross_sales", "discount_usd", "list_price_per_unit"]
NUMERIC = ["quantity", "discount", "month_of_year", "discount_over_20", "is_discounted"]
CATEGORICAL = ["category", "sub_category", "segment", "ship_mode", "region", "state"]
FEATURES = MONEY + NUMERIC + CATEGORICAL

# Never used as features: built from the answer (profit) or not known when the line is priced.
LEAKY = ["profit", "cost", "margin_pct"]
NOT_KNOWN_AT_ORDER = ["ship_date"]
IDENTIFIERS = ["order_id", "customer_id", "product_id", "product_name", "city", "order_month"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features known when a line is priced. Nothing here is learned from the target."""
    out = pd.DataFrame(index=df.index)
    out["sales"] = df["sales"]
    out["gross_sales"] = df["gross_sales"]
    out["discount_usd"] = df["discount_usd"]
    out["list_price_per_unit"] = df["gross_sales"] / df["quantity"]
    out["quantity"] = df["quantity"]
    out["discount"] = df["discount"]
    out["month_of_year"] = pd.to_datetime(df["order_date"]).dt.month
    out["discount_over_20"] = (df["discount"] > 0.2).astype(int)
    out["is_discounted"] = (df["discount"] > 0).astype(int)
    for c in CATEGORICAL:
        out[c] = df[c]
    return out


def make_preprocessor() -> ColumnTransformer:
    money = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("money", money, MONEY),
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
