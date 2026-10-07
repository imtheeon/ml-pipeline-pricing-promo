# ML pipeline example: which order lines lose money?

A learning project: a 16-step, gated ML pipeline (scikit-learn + XGBoost 3.4.1, CPU only, $0).

- **Goal:** flag order lines that lose money, especially the 1-20% discount lines a simple rule misses
- **Data:** Superstore order lines (9,994 rows, 2014-2017) from `imtheeon/pricing-promo-analysis` (`data/clean/lines.parquet`, not copied here)
- **Split:** chronological 60/20/20; final exam touched once
- **Result (final exam, PR-AUC):** dummy 0.181, discount>20% rule 0.735, logistic 0.941, random forest 0.953 (selected), XGBoost 0.955
- **Caveat:** the flag threshold chosen on validation gave 87.7% precision on the exam, below the 90% target

## Read first
- `PIPELINE.md` - the full log: steps, gates, decisions, monitoring plan
- `figures/` - every chart, with its explanation in PIPELINE.md

## Run
1. Put `lines.parquet` at `data/clean/lines.parquet`; keep these files in an `ml_pipeline/` folder next to `data/`
2. `pip install pandas scikit-learn xgboost matplotlib marimo pyarrow joblib`
3. Run `01_eda.py`, `02_prep.py`, `03_model.py`, `04_error_analysis.py`, `05_final_exam.py` in order from the folder above `ml_pipeline/`
