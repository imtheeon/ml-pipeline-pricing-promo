import pandas as pd
import pytest

from preprocess import CATEGORICAL, FEATURES, IDENTIFIERS, LEAKY, build_features, make_preprocessor

DATA = "data/clean/lines.parquet"


@pytest.fixture(scope="module")
def lines():
    return pd.read_parquet(DATA)


def test_leaky_and_identifier_columns_are_never_features():
    assert not set(FEATURES) & set(LEAKY)
    assert not set(FEATURES) & set(IDENTIFIERS)


def test_features_have_no_missing_values(lines):
    X = build_features(lines)
    assert list(X.columns) == FEATURES
    assert X.notna().all().all()


def test_preprocessor_handles_a_category_it_never_saw(lines):
    X = build_features(lines)
    prep = make_preprocessor().fit(X.iloc[:5000])
    new = X.iloc[5000:5010].copy()
    new["state"] = "Atlantis"
    Z = prep.transform(new)
    assert Z.shape[0] == 10 and pd.DataFrame(Z).notna().all().all()


def test_discount_flag_matches_the_rule(lines):
    X = build_features(lines)
    assert ((X["discount"] > 0.2).astype(int) == X["discount_over_20"]).all()
