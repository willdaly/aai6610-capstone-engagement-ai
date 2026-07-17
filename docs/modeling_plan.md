# Modeling Plan — Preprocessing and Baseline Models

This file is the contract for the first modeling phase: a leakage-proof preprocessing
pipeline and a compared set of baseline classifiers, evaluated the way the proposal
promises (AUPRC and recall primary) plus a net-revenue lens that the EDA showed is
necessary. It builds directly on `docs/eda_findings.md`; every design decision below
cites the finding that motivates it. Read that document and `CLAUDE.md` before
implementing anything here.

## Objective

Primary task: binary classification of `TARGET_B` (respond / not respond), per the
approved proposal. The EDA's amount/frequency inversion (findings task 5: the
smallest-gift band responds 9.41% vs 3.42% for the largest) means a classifier chosen
on response probability alone systematically prefers the constituents who give least.
The resolution for this phase is: keep classification as the modeled task, and add
net revenue as an *evaluation and threshold-selection* lens, not a second model.
Net revenue for a contacted set = sum of `TARGET_D` over contacted responders minus
$0.68 per contact (the campaign's documented mail cost).

## Ground rules

- All data access through `model/src/load_data.py`. The test split is touched exactly
  once, by the final evaluation function, after every model and threshold choice is
  frozen. Everything else (CV, tuning, threshold selection, fairness checks during
  development) happens inside the training split, using the 15% validation carve-out
  (`make_split(df, validation=True)`) or cross-validation as specified per step.
- Every preprocessing step lives inside a scikit-learn `Pipeline` /
  `ColumnTransformer` that is fit on training data only. No statistic computed on the
  full dataset may enter a transform. This is the mechanical guarantee against
  leakage; tests must enforce it.
- Seeds: `random_state=2026` everywhere a seed exists (splits, CV shuffling, models).
- Every reported number comes from an actual run and lands in
  `docs/modeling_findings.md` with its frame (CV / validation / test) stated.
- Runtime budget: the full training script completes on a laptop CPU in minutes, not
  hours. If a grid blows the budget, shrink the grid and say so, don't cache-and-hope.

## Feature policy (decided, not to be relitigated mid-implementation)

The EDA inventory (findings task 1) assigns all 481 columns to eight groups. This
phase uses a curated feature set, group by group:

**In:**
- `demographics` (13): all. AGE and INCOME imputed per below.
- `giving_history` summaries only: `NGIFTALL`, `CARDGIFT`, `RAMNTALL`, `LASTGIFT`,
  `AVGGIFT`, `MINRAMNT`, `MAXRAMNT`, plus recency in days derived from `LASTDATE`
  (see date handling). The per-mailing `RDATE_*`/`RAMNT_*` pairs are OUT: their
  missingness is structural (findings task 1), and their information is already
  aggregated in the summaries. Multicollinearity among summaries is documented
  (r up to 0.91) and tolerated: it does not hurt the tree models, and the logistic
  baseline exists for calibration comparison, not coefficient reading.
- `promotion_history`: counts derived from the mailing columns (number of promotions
  received, number responded to if derivable), plus `RFA_2F` and `RFA_2A` as ordered
  categoricals. `RFA_2R` is constant (findings task 1, correction 5) and is OUT, as
  are the raw historical `RFA_3`..`RFA_24` codes.
- `geography`: `STATE` only, frequency-encoded (fit on train). Zip and area codes OUT.

**Out, with reasons the report can quote:**
- `census` (290 columns): excluded from the baseline feature set. Rationale: the EDA
  showed the dataset is mostly neighborhood description, and neighborhood-derived
  predictions are precisely where proxy discrimination risk concentrates. Baselines
  first establish what individual-level features achieve. A census-included variant
  is run as ONE explicit experiment at the end of this phase (same pipeline, census
  block added, same CV) so the report can state what predictive lift the
  neighborhood data buys and weigh it against the fairness cost. That comparison is
  a headline result for the ethics section, not a footnote.
- `interests_overlay` (33): vendor-appended lifestyle data; excluded on provenance
  grounds for the baseline, noted as future work.
- `id_admin`: all OUT (`CONTROLN` is the row ID; the rest are origin codes).
- Targets: `TARGET_B` is y; `TARGET_D` appears only in the net-revenue evaluation,
  never as a feature. `CONTROLN`, `TARGET_B`, `TARGET_D` being absent from X gets its
  own test.

## Preprocessing pipeline

1. **Blank normalization.** 65 of 74 string columns contain whitespace-only values
   (findings task 1). A transformer converts whitespace-only strings to NaN before
   anything else.
2. **Dates.** Gift and promotion dates are YYMM integers. Convert the ones used
   (`LASTDATE` for recency) to months-before-campaign relative to the campaign date
   (9706 per the dataset docs). Do not feed raw YYMM values as numerics.
3. **Imputation.** Numeric: median (fit on train). Categorical: explicit "missing"
   category. No missingness indicator flags for AGE/INCOME: the EDA tested this and
   missingness does not predict response (chi-square, findings task 3). Cite that,
   don't re-decide it.
4. **Encoding.** Low-cardinality categoricals (GENDER, HOMEOWNR, RFA_2F, RFA_2A,
   INCOME as ordered): one-hot or ordinal as appropriate, fit on train.
   `STATE`: frequency encoding fit on train.
5. **Scaling.** Standardize numerics for the logistic model only; trees get raw
   values. Implement as two pipeline variants sharing the transformers.
6. **Age floor.** Rows with AGE <= 5 are documented bad data (findings task 5, 15
   rows). Set AGE to NaN for AGE <= 5 rather than dropping rows.

## Models, in order

All trained on the same feature matrix, compared with stratified 5-fold CV on the
combined training portion (train + validation rows), scored on average precision
(AUPRC). Small, honest grids; tuning by `GridSearchCV` or `RandomizedSearchCV` with
the same CV object.

1. **Dummy baseline** (`DummyClassifier`, prior strategy). Exists to anchor AUPRC at
   the 5.08% positive rate. Every other model must beat it or the phase has failed.
2. **Logistic regression**, `class_weight="balanced"`, small C grid. The EDA predicts
   modest performance here (near-zero linear correlations, findings task 5); the
   point is a calibrated, interpretable floor.
3. **Random forest**, `class_weight="balanced_subsample"`, modest grid over depth and
   estimators.
4. **HistGradientBoostingClassifier** with class weighting via `sample_weight`, grid
   over learning rate and depth. Expected strongest; native NaN handling noted.

No imblearn, no SMOTE: the CLIMB benchmark the proposal cites found naive rebalancing
unreliable, and class weights are the course-consistent mechanism. No neural networks
in this repo (CLAUDE.md).

## Evaluation protocol

On CV during development; then, once per model family with tuned hyperparameters,
refit on the full training portion and evaluate ONCE on the held-out test split:

- **Primary:** average precision (AUPRC) with the 0.0508 prior as the stated floor,
  and recall at the chosen operating threshold.
- **Threshold selection:** on the validation carve-out, sweep thresholds and choose
  the one maximizing expected net revenue (TARGET_D of contacted responders minus
  $0.68 x contacts). Report precision/recall/contact-count at that threshold, and
  also at the naive 0.5 threshold to show why 0.5 is wrong here.
- **Net-revenue curve:** revenue vs. fraction-of-list-contacted for each model, with
  the contact-everyone and contact-no-one baselines as horizontal references. This
  figure is the business argument for the whole project.
- **Calibration:** reliability curve for the best model.
- **Fairness slices** (proposal commitment): at the chosen threshold, contact rates
  and recall by GENDER, AGE band, and INCOMe band, on validation. Report gaps
  plainly; no mitigation in this phase, but flag anything stark.
- **Census experiment:** best model's pipeline with census block added, same CV.
  Report the AUPRC delta in both directions of the trade-off.

## Deliverables

1. `model/src/features.py`: feature policy as code (column lists per group, derived
   features, the preprocessing `ColumnTransformer` builders).
2. `model/src/train.py`: runnable end to end (`python model/src/train.py`), trains
   all four models, writes `docs/modeling/results.csv` (model, params, CV AUPRC
   mean/std, validation metrics, test metrics where run), saves fitted pipelines to
   `model/artifacts/` (gitignored), writes all figures to `docs/figures/`.
3. `docs/modeling_findings.md`: findings organized as: setup, model comparison,
   threshold and revenue analysis, fairness slices, census experiment, limitations.
   Repo prose style.
4. Tests: X never contains `CONTROLN`/`TARGET_B`/`TARGET_D`; transformers fitted on
   train only (fit a pipeline on train, assert its learned statistics don't change
   when test data changes); blank normalization works; date conversion sane;
   AGE<=5 nulled; deterministic results for a fixed seed on a subsample.
5. Figures follow the existing EDA styling helpers and standards.

## Out of scope

Neural networks, SMOTE/resampling, the assistant, hyperparameter searches beyond the
small grids, deployment packaging, and any fairness mitigation (measurement only in
this phase). If the census experiment or fairness slices surface something that seems
to demand action, record it in findings for discussion, don't act.
