# EDA Plan — KDD Cup 1998 Learning Set

This file specifies the exploratory data analysis for the engagement prediction model.
It lives in `docs/` and is the contract for what the EDA produces. The findings feed
the capstone report's "Description of the dataset" and "Analysis of the dataset"
sections, so completeness matters more than cleverness: the grader wants to see the
process step by step, including the obvious moves (loading, shapes, dtypes, missing
values) before any analysis. Do not skip steps because they seem trivial.

## Ground rules

- All data access goes through `model/src/load_data.py` (`load_raw`, `make_split`).
  Never re-read the CSV directly and never re-implement the split.
- **Peeking rule.** Structural facts (shape, dtypes, missingness, cardinality) may use
  the full dataset. Anything involving a relationship with `TARGET_B` or `TARGET_D`
  (response rates by segment, distributions split by target, correlations with the
  target) uses the TRAINING split only, from `make_split(df)` with defaults. The test
  set stays untouched until final model evaluation. State which frame each figure and
  number comes from.
- Every number in the findings document comes from an actual run. No invented,
  remembered, or rounded-from-memory values.
- `CONTROLN` is a row ID. It appears in the ID/keys inventory and nowhere else.
- Deterministic and fast: the full EDA run finishes in a few minutes and produces the
  same outputs on every run.

## Feature landscape (orient before analyzing)

481 columns is too many to plot exhaustively. The EDA covers every column at the
inventory level, then goes deep on representative features per group. The map below
was provisional when this plan was written; task 1 verified it against the data on
2026-07-16 and the corrections are folded in here. Corrections are marked
**[corrected]** with what the data actually showed. The authoritative, executable form
of this map is the group assignment in `model/src/eda.py`, which fails loudly if any
of the 481 columns matches no group.

- **Identifiers / admin** (22 columns): `CONTROLN`, `NOEXCH` (string column: 35 `X`,
  7 blank, as described), origin and source codes (`ODATEDW`, `OSOURCE`, `TCODE`),
  the `REC*` donor flags, and the `MDMAUD*` major-donor codes.
  **[corrected]** zip and state fields are not admin. They belong with the other
  geographic columns, which are their own group below.
- **Geography** (10 columns): `STATE`, `ZIP`, `DOMAIN`, `CLUSTER`, `CLUSTER2`,
  `GEOCODE`, `GEOCODE2`, `MSA`, `ADI`, `DMA`.
  **[corrected]** this group did not exist in the provisional map. `MSA`, `ADI`, and
  `DMA` sit inside the census run by position but are area codes, not neighborhood
  statistics, so they are excluded from the census block explicitly.
- **Demographics** (13 columns): `AGE` (24.8% missing), `GENDER`, `INCOME` (22.3%
  missing), `HOMEOWNR`, `DOB`, `AGEFLAG`, the wealth ratings, and the `CHILD*` /
  `NUMCHLD` household composition fields.
  **[corrected]** the wealth ratings are missing far more often than the plan's
  headline figures suggest: `WEALTH1` is 46.9% missing and `WEALTH2` 45.9%. `NUMCHLD`
  is 87.0% missing. `AGE` and `INCOME` are not the worst-affected demographics.
- **Interests / purchased overlay** (33 columns): mail-order buying counts (`MB*`),
  publication subscriptions (`PUB*`, `MAG*`), `HIT`, and the lifestyle interest flags
  (`BIBLE`, `PETS`, `BOATS`, and so on).
  **[corrected]** this group did not exist in the provisional map, which would have
  swept these into demographics. They are vendor-appended overlay data, not fields the
  nonprofit collected, and that provenance difference matters for both modeling and
  the ethics discussion, so they are kept separate.
- **Census / neighborhood block** (290 columns): one contiguous run from `POP901`
  through `AC2` inclusive, minus `MSA`/`ADI`/`DMA`, plus seven percentage columns that
  sit outside the run. Treat as one group; characterize collectively (dtype,
  missingness, value ranges), pick 2-3 exemplars.
  **[corrected]** the run is not "roughly `POP901` through the `AC*`/`EC*`/`HC*`
  ranges". The `EC*` and `HC*` families are *inside* the run, not after it; the run
  ends at `AC2`, immediately before `ADATE_2`. Separately, `MALEMILI`, `MALEVET`,
  `VIETVETS`, `WWIIVETS`, `LOCALGOV`, `STATEGOV`, and `FEDGOV` are census percentages
  that sit *before* the run (positions 43-49, among the overlay columns) and the
  provisional map missed them. The group is defined by endpoints plus an explicit
  exclusion and extra list, not by name pattern.
- **Giving history (individual)** (57 columns): `NGIFTALL`, `RAMNTALL`, `LASTGIFT`,
  `AVGGIFT`, `MINRAMNT`, `MAXRAMNT`, gift dates, and the per-promotion `RDATE_*` /
  `RAMNT_*` families. Expect strong right skew; plan log-scale views. (Confirmed:
  `MAXRAMNT` skew is 99.8, `AVGGIFT` 26.0.)
- **Promotion history** (54 columns): the `ADATE_*`, `RFA_2` .. `RFA_24` code columns,
  plus the precomputed `RFA_2R`, `RFA_2F`, `RFA_2A` (recency / frequency / amount
  components of the most recent RFA code). Prefer the precomputed components for
  analysis.
  **[corrected]** `RFA_2R` is constant: it is `L` for all 95,412 rows and carries no
  information at all. Only `RFA_2F` and `RFA_2A` are usable, so task 5 charts those
  two and `RFA_2R` is reported as a near-constant column under task 6.
- **Targets** (2 columns): `TARGET_B` (binary, 5.08% positive), `TARGET_D` (amount,
  0.0 for non-responders).

## Tasks, in order

### 1. Inventory (full dataset)

A machine-generated column inventory covering all 481 columns: name, pandas dtype,
non-null count, % missing, cardinality (nunique), and an assigned group from the map
above. Save as `docs/eda/column_inventory.csv`. Summarize in the findings doc: counts
of columns per group, per dtype, and the top 20 columns by missingness. Verify the
group map; correct it in this file if the data disagrees.

**[corrected]** the inventory also carries `pct_blank`, the share of rows holding a
blank or whitespace-only string. Task 3 requires finding disguised missingness, and it
turned out to affect 65 of the 74 string columns, so an inventory reporting NaN-only
missingness would understate the problem enough to mislead a reader of the report.

### 2. Targets (training split)

- `TARGET_B`: counts and rate in the training split; confirm consistency with the
  full-dataset rate (stratification should guarantee it).
- `TARGET_D` among responders only: distribution, five-number summary, and how many
  responders gave exactly the modal amounts (donation amounts cluster on round
  numbers; show this). One figure: histogram of `TARGET_D` for responders.
- State the implication in the findings: 5.08% positives means accuracy is
  uninformative; AUPRC baseline (the positive rate itself) is the number to beat.

### 3. Missing data (full dataset for extent; training split for target relationships)

- Figure: horizontal bar of the 20 most-missing columns, with % labels.
- `AGE` and `INCOME` get individual treatment: distribution of the observed values,
  and (training split) response rate for missing vs. present. If missingness itself
  predicts response, say so; that decides whether imputation should carry an
  indicator flag.
- Check for disguised missingness in categorical codes (blank strings, single
  spaces), not just NaN. `NOEXCH` already demonstrates the pattern.
  **[corrected]** `NOEXCH` is the mildest case, not a representative one: 7 blank rows
  out of 95,412. The pattern affects 65 of the 74 string columns, and at the top of the
  range it is the dominant value rather than an edge case (`RECPGVG` 99.9% blank,
  `GEOCODE` 84.0% blank, `HOMEOWNR` 23.3% blank). The top-20 NaN missingness figure
  this task calls for therefore understates total missingness by a wide margin, because
  every column in that top 20 is an `RDATE_*`/`RAMNT_*` column and none of the blank-
  carrying string columns appear in it at all. Report both, and say plainly that the
  NaN figure alone is not the whole picture.

### 4. Distributions of key features (training split)

- Demographics: `AGE` histogram; `GENDER` and `INCOME` bar charts of category counts.
- Giving history: `LASTGIFT`, `AVGGIFT`, `NGIFTALL`, `RAMNTALL` on both raw and log
  scales (one combined figure, small multiples). Note outliers and whether they look
  like data errors or genuine large donors; do not remove anything yet.
- Census exemplars: the 2-3 chosen exemplar columns, briefly.

### 5. Relationships with response (training split)

- Response rate by `RFA_2F` (frequency) and by `RFA_2A` (amount category): two bar
  charts with the overall 5.08% rate drawn as a reference line.
- Response rate by `AGE` band (decade bins) and by `INCOME` category.
- Correlation heatmap of the numeric giving-history features plus `TARGET_B`
  (point-biserial via Pearson is fine at this stage; say that's what it is).
- Findings must state the strongest candidate signals in plain language, with the
  numbers.

### 6. Data quality worries for the report

A short section in the findings doc listing anything downstream modeling must handle:
high-cardinality categoricals, near-constant columns (report how many columns have
>99% a single value), date fields stored as numbers, the sparse-code columns, and
the class imbalance. Each with a one-line recommended handling, marked as
recommendation, not decision.

## Figures: standards and mechanics

- Save every figure to `docs/figures/` as PNG at 200 dpi with descriptive snake_case
  names (`response_rate_by_rfa2f.png`, not `fig5.png`).
- Course visualization standards apply: trustworthy (no truncated axes without a
  stated reason, no misleading framing), effortless (direct titles that state the
  takeaway, labeled axes with units, reference lines where they help), elegant (no
  chart junk, no pastel palettes, no default-blue-on-gray). Use a single consistent
  colorblind-safe palette across all figures; one accent color for the thing the
  reader should look at.
- Every figure gets a matching entry in the findings doc: what it shows, the frame it
  was computed on (full vs. training), and one paragraph of observation.

## Deliverables

1. `model/src/eda.py`: runnable script (`python model/src/eda.py`) that produces
   every table and figure above, deterministic, no interactivity. Organize as small
   functions per task so pieces are importable later.
2. `docs/eda/column_inventory.csv`: the full 481-column inventory.
3. `docs/figures/*.png`: the figures, exact set determined by the tasks above.
4. `docs/eda_findings.md`: the written findings, organized by the task numbers above,
   every claim carrying its number and its frame (full vs. training). This document
   is the raw material for the report's dataset and analysis sections; write it in
   the repo's prose style (see CLAUDE.md), no marketing language, no em-dashes.
5. Tests where they're cheap: the inventory has 481 rows; the figure directory
   contains the expected filenames after a run; the training frame used matches
   `make_split` defaults. Do not test matplotlib output pixel-for-pixel.

## Out of scope for this pass

No imputation, no encoding, no feature selection, no models, no assistant work.
Recommendations about those belong in the findings doc; implementations do not.
