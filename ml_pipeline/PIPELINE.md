# ML Pipeline: Which order lines lose money? (pricing-promo-analysis repo)

Started 2026-10-07. Data: `data/clean/lines.parquet` from this repo (Superstore, 9,994 order lines, 2014-2017), read-only. Tools: scikit-learn plus XGBoost 3.4.1 (latest, checked with Context7 docs). Budget: $0, CPU only.

## Checklist

- [x] 1. Data inspection - done 2026-10-07: 9,994 lines x 24 columns, 0 missing, 1 exact duplicate (0.01%), 5,009 orders, 793 customers, 2014-01-03 to 2017-12-30; 18.7% of lines lose money
- [x] 2. EDA - done 2026-10-07: 6 figures; loss rate is 0% at no discount, 14% at 1-20%, 90% at 21-40%, 100% above 40%
- [x] 3. Define the prediction problem - done 2026-10-07, contract below, awaiting Gate A approval
- Gate A: approved 2026-10-07
- [x] 4. Data cleaning - done 2026-10-07: 0 missing, 0 non-positive sales, 0 bad quantities, discount 0-0.8, sales = gross*(1-discount) exactly; outliers (12-16% of money columns) are real big orders and are kept
- [x] 5. Data engineering - done 2026-10-07: one row per order line, 15 features known at pricing time; leaky columns (profit, cost, margin_pct) asserted absent
- [x] 6. Train / validation / test split - done 2026-10-07: guard.split chronological; 5,996 train / 1,998 validation / 1,998 exam (1 exact duplicate dropped); loss share 0.188 / 0.189 / 0.181; 1 order straddles validation/exam (negligible)
- [x] 7. Feature engineering - done 2026-10-07: month of year, list price per unit, discount over 20% flag, discounted flag; all row-wise, nothing learned from the target
- [x] 8. Preprocessing - done 2026-10-07: log1p+scale on money columns, scale on other numbers, one-hot on 6 categoricals (largest has 49 levels) = 86 columns, fit on train only; 3 states appear in validation but not in train (handled by ignoring unknowns)
- Gate B: approved 2026-10-07
- [x] 9. Baseline model - done 2026-10-07: on validation, dummy PR-AUC 0.189 (catches 0 of 378 losses); discount>20% rule PR-AUC 0.763, catches 276 of 378 (73%) at 97.5% precision
- [x] 10. Model training - done 2026-10-07: logistic, random forest and XGBoost 3.4.1 fit on train only (XGBoost kept 134 trees via early stopping on the last 15% of train)
- [x] 11. Hyperparameter tuning - done 2026-10-07: 10 settings, forward-chaining CV (4 folds) on train only; CV PR-AUC 0.934-0.946, differences mostly inside the fold noise (sd about 0.01); cv_results.csv
- [x] 12. Model evaluation - done 2026-10-07: validation PR-AUC logistic 0.949, forest 0.958, XGBoost 0.962; recall at 90% precision 0.80 / 0.83 / 0.86 vs rule 0.73; pre-declared rule selects random forest (min_leaf=3) because it is within 0.005 of XGBoost (gap 0.0048); selected.json
- Gate C: approved 2026-10-07
- [x] 13. Error analysis - done 2026-10-07: validation misses are all in the 1-20% discount band (63 of 102 losses there missed; 0 missed above 20%); worst sub-categories Accessories, Chairs, Storage; scores are well calibrated (lines scored 0.9+ lost money 99% of the time)
- [x] 14. Final test - done 2026-10-07: ONE pass; selected random forest PR-AUC 0.953 (validation 0.958); at the validation threshold it caught 300 of 362 losses (83%) at 87.7% precision, just under the 90% target; rule 0.735, logistic 0.941, XGBoost 0.955 (all within noise of each other); final_results.json
- [x] 15. Deployment - skipped 2026-10-07: user did not ask for deployment
- [x] 16. Monitoring + retraining plan - done 2026-10-07: see Monitoring plan below
- Gate D: report delivered 2026-10-07, awaiting the user's sign-off

## Step notes
- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: is_loss=0 is 81.3% of 9,994 rows, so a majority-class dummy scores 81.3%; the minority share is 18.7%.
- Figure 02_distributions.png: Histograms of 9 numeric features; 9 are strongly skewed (sales, quantity, discount, profit, gross_sales), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with is_loss: margin_pct (0.77); anything above 0.95 is a leakage suspect.
- Figure 02_temporal_coverage.png: order_date spans 2014-01-03 to 2017-12-30 over 48 months; a chronological split must hold out the latest period.
- Figure 02_loss_by_discount.png: Share of order lines that lost money in each discount band. Losses are rare at 0-20% off and nearly universal above 20%, so discount depth alone already separates most losses; any model has to beat this simple rule.

## Problem contract (step 3, awaiting Gate A approval)
- Target: `is_loss` = 1 when an order line's profit is below 0 (18.7% of lines); label comes from the `profit` column
- Prediction unit: one order line, scored at the moment it is priced
- Information available at prediction time (allowed features): discount, sales (net), gross_sales, discount_usd, quantity, unit_net_price, category, sub_category, segment, ship_mode, region, state, month of year of the order date
- EXCLUDED as leakage (the automatic checker did NOT flag these, I did by definition): `profit` (the target itself), `cost` (= sales - profit), `margin_pct` (= profit / sales)
- EXCLUDED as not known at order time: `ship_date`. EXCLUDED as identifiers / too specific: `order_id`, `customer_id`, `product_id`, `product_name`, `city`, raw `order_date` (year would only capture trend), `order_month`
- Objective: flag lines likely to lose money so low-margin discounts can be questioned
- Metric (proposed): PR-AUC for the loss class as the main score, plus recall of losses at precision >= 90% (catch losses while flags stay mostly right); all compared with the baselines below
- Baselines: (a) dummy that never flags a loss (81.3% accuracy, 0 losses caught); (b) the repo's own rule "discount above 20% = loss", which needs no fitting (EDA preview: roughly 72% recall at roughly 97% precision on all data; to be re-measured on validation); (c) logistic regression
- Real question: the rule misses the 14% of 1-20% discount lines that lose money; do models find those without raising false alarms?
- Split plan: chronological 60/20/20 by order_date (guard.split with time_col): train 2014-01-03 to 2016-11-04 (5,996 rows), validation 2016-11-04 to 2017-07-18 (1,999), final exam 2017-07-18 to 2017-12-30 (1,999). Loss share 0.189 / 0.189 / 0.181. One order straddles each cut (negligible, noted).
- Risks: the final exam period is the busy Q4 season; patterns may shift; XGBoost early stopping must use a slice of TRAIN (not validation) so validation stays clean
- Not pushed anywhere: all files live in the local clone's ml_pipeline/ folder; nothing is committed or pushed without the user's say-so

## Standing instruction and pre-declared rules (written 2026-10-07 BEFORE any modeling)
- User, at Gate A: "yes and only come back to me when you're done with all the gates or with all the steps." Treated as a standing instruction to run Gates B-C through to the wrap-up without stopping. Every gate still gets its artifacts, and every decision made on the user's behalf is listed in the final report.
- Model choice rule (fixed now): the model with the highest validation PR-AUC is selected; if two models are within 0.005 PR-AUC, the simpler one wins (logistic < random forest < XGBoost). Threshold for flagging: the lowest score on validation that keeps precision at or above 0.90.
- Final exam plan (fixed now): ONE pass over the locked exam rows computes every pre-declared predictor (dummy, discount>20% rule, logistic, random forest, XGBoost, all fit on train only). The claim is about the pre-selected model only; the others are shown for comparison, nothing is switched or tuned afterwards. That is one touch of the exam set, so no override is needed.
- guard.split drops exact duplicate rows (this data has 1), so the modeled table has 9,993 rows.
- Figure 04_outliers.png: Share of lines with unusually large dollar values or quantities. These are genuine big orders, not errors, so they are kept; a log transform tames them for the linear model.
- Figure 06_split_timeline.png: The chronological split: train on the earliest lines, tune on the next, and keep the latest period locked as the final exam. The loss share is about the same in each part, but the exam period is the busy season.
- Figure 08_log_scaling.png: Sales has a long right tail. Taking the log and scaling (using training rows only) spreads it out so a few huge orders do not dominate the linear model.

## Model rationale (written before any training, Gate B)
- traits: binary target, 9,993 rows (medium), minority 18.7% (not imbalanced), has a date column (chronological split), no groups
- baseline: never-loss dummy + "discount > 20%" rule + logistic regression (L2, log-scaled money features)
- candidates: XGBoost 3.4.1 (hist, depth 3-5, early stopping on the last 15% of train rows); random forest; regularized logistic
- ruled out: neural nets (tabular, ~10k rows); random split / plain k-fold (dated data, use forward-chaining CV); unregularized deep trees; one-hot on high-cardinality IDs (identifiers excluded; largest categorical has 49 levels)
- metric: PR-AUC for loss primary; recall at precision >= 0.90 secondary; accuracy not used for selection
- tuning: TimeSeriesSplit (4 folds) on TRAIN only; validation used once per model to compare
- Note on Gate B: recorded under the user's standing Gate A instruction (come back only when all gates are done); rationale in the Model rationale section.
- Figure 10_model_comparison.png: Each model scored on the validation period. The left bars rank every cut-off at once; the right bars show how many money-losing lines each model finds if we only accept flags that are right 90% of the time.
- Figure 12_evaluation.png: Curves show the trade-off: the further a line stays up and to the right, the more losses it finds with fewer wrong flags. The dashed line is the 90% precision bar we set in advance.
- Note on Gate C: recorded under the user's standing instruction. Artifacts: Figure 10_model_comparison.png, Figure 12_evaluation.png, runs.csv, cv_results.csv, selected.json. Selection followed the rule fixed before modeling (simpler model within 0.005 PR-AUC wins); XGBoost was nominally best but not by enough to override it.
- Figure 13_error_analysis.png: Where the selected model still misses losses. All 63 missed losses sit in the 1-20% discount band, the group the simple rule cannot see, so that is where any gain over the rule has to come from.
- Step 14 final test: pr_auc=0.9528 (touch 1, 2026-10-07)

## Monitoring plan (step 16)
- Each month, compare the share of discounted lines and the loss rate with the training period (18.7%); a move of more than 3 points means look closer
- Each quarter, score the new lines once actual profit is known: if recall at the flag threshold drops below 70% or precision below 80%, retrain
- Retrain on the newest data every 6 months, keeping the same chronological split and rules; re-pick the threshold on the newest validation slice (the exam showed 87.7% precision against a 90% target)
- Watch for new states, sub-categories or a discount policy change; the model has not seen them
- Limits: one store's data from 2014-2017; flags help review discounts, they do not set prices
- Figure 17_business_impact.png: Dollars lost on money-losing lines in the final exam period, and how much of that sits on lines each method flags. Flagging is not saving: it only points reviewers to the lines worth questioning.
