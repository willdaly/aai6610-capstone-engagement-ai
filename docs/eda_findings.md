# EDA Findings — KDD Cup 1998 Learning Set

What the exploratory analysis specified in `docs/eda_plan.md` actually found. Organized
by the plan's task numbers. Every number here comes from a run of
`python model/src/eda.py`; nothing is estimated or carried over from documentation.

The dataset is a 1998 direct-mail campaign from a national veterans nonprofit, used as
a public proxy for constituent engagement. It is not Sturge-Weber Foundation data, and
this analysis is not affiliated with or endorsed by the Foundation.

**Frames.** Per the plan's peeking rule, structural facts (shape, dtypes, missingness,
cardinality) are computed on the full dataset of 95,412 rows. Anything involving
`TARGET_B` or `TARGET_D` is computed on the training split only (76,329 rows, from
`make_split(df)` with course defaults: 80/20 stratified, `random_state=2026`). The test
split of 19,083 rows is untouched and stays that way until final model evaluation.
Every claim below states its frame.

---

## 1. Inventory

**Frame: full dataset (95,412 rows, 481 columns).** Structural facts only.

The full inventory is `docs/eda/column_inventory.csv`, one row per column, 481 rows.
It carries the column name, pandas dtype, non-null count, % missing (NaN), % blank
(whitespace-only strings), cardinality, and the assigned group.

### Columns per group

| Group | Columns | What it holds |
| --- | ---: | --- |
| `census` | 290 | Neighborhood statistics from the 1990 census, appended by zip |
| `giving_history` | 57 | Per-gift dates and amounts, plus derived summaries |
| `promotion_history` | 54 | Mailings sent and the RFA code as of each |
| `interests_overlay` | 33 | Vendor-appended mail-order and lifestyle indicators |
| `id_admin` | 22 | Row ID, origin codes, donor flags |
| `demographics` | 13 | Age, gender, income, wealth, household composition |
| `geography` | 10 | State, zip, and area/cluster codes |
| `target` | 2 | `TARGET_B`, `TARGET_D` |
| **Total** | **481** | |

The single most important structural fact: 290 of 481 columns (60.3%) are census
neighborhood statistics, and only 13 (2.7%) describe the individual constituent. The
dataset is mostly about where people live, not who they are. Anything the model learns
from the census block is a statement about a neighborhood, which is exactly the kind of
inference that needs the fairness checks the proposal commits to.

### Columns per dtype

| dtype | Columns |
| --- | ---: |
| `int64` | 310 |
| `float64` | 97 |
| `str` | 74 |

No column loaded as `object`. `load_data.load_raw` declares `NOEXCH` as a string and
escalates any other mixed-type column to an error, so these dtypes are inference that
has been checked, not inference that was trusted.

### Verification of the group map

The plan's map was provisional and the data disagreed with it in five places. The
corrections are recorded in `docs/eda_plan.md`, marked **[corrected]**, and implemented
in `model/src/eda.py`, where the group assignment raises rather than bucketing an
unmatched column as "other". All 481 columns match exactly one group.

1. **The census run ends at `AC2`, and the `EC*`/`HC*` families are inside it, not
   after it.** The plan guessed "roughly `POP901` through the `AC*`/`EC*`/`HC*`
   ranges". `POP901` through `AC2` is one contiguous run of 286 columns ending
   immediately before `ADATE_2`. The group is defined by those endpoints.
2. **`MSA`, `ADI`, and `DMA` sit inside the run but are not neighborhood statistics.**
   They are geographic area codes (`MSA` ranges 0 to 9,360 across 298 levels). Grouping
   them with the percentages would invite treating an area code as a continuous
   quantity. They are excluded from the census block and assigned to geography.
3. **Seven census percentages sit outside the run.** `MALEMILI`, `MALEVET`,
   `VIETVETS`, `WWIIVETS`, `LOCALGOV`, `STATEGOV`, and `FEDGOV` are at positions 43-49,
   in the middle of the overlay columns, and are percentages on a 0-99 scale like the
   rest of the census group. The plan's map missed them. With those seven added and
   three area codes removed, the census group is 290 columns.
4. **Geography and purchased overlay are their own groups.** The plan folded zip and
   state into admin and had nowhere to put the 33 mail-order and lifestyle columns.
   Overlay data was appended by a vendor rather than collected by the nonprofit, and
   that provenance difference matters for both modeling and the ethics discussion.
5. **`RFA_2R` is constant.** The plan says to prefer the precomputed `RFA_2R`,
   `RFA_2F`, `RFA_2A` components. `RFA_2R` is `L` for all 95,412 rows and carries no
   information. Only `RFA_2F` and `RFA_2A` are usable, so task 5 charts those two and
   `RFA_2R` is reported under task 6 as a near-constant column.

### Top 20 columns by % missing (NaN)

| Column | % missing | Column | % missing |
| --- | ---: | --- | ---: |
| `RDATE_5` | 99.99 | `RDATE_23` | 91.76 |
| `RAMNT_5` | 99.99 | `RAMNT_23` | 91.76 |
| `RDATE_3` | 99.75 | `RDATE_20` | 91.73 |
| `RAMNT_3` | 99.75 | `RAMNT_20` | 91.73 |
| `RDATE_4` | 99.71 | `RDATE_7` | 90.68 |
| `RAMNT_4` | 99.71 | `RAMNT_7` | 90.68 |
| `RDATE_6` | 99.19 | `RDATE_17` | 90.15 |
| `RAMNT_6` | 99.19 | `RAMNT_17` | 90.15 |
| `RDATE_15` | 92.39 | `RDATE_21` | 90.03 |
| `RAMNT_15` | 92.39 | `RAMNT_21` | 90.03 |

Every column in the top 20 is an `RDATE_*`/`RAMNT_*` pair from the giving-history
group, and the pairs match exactly: `RDATE_5` and `RAMNT_5` are both 99.99% missing.
That is structure, not damage. These columns record the date and amount of the response
to one specific historical mailing, so a row is missing whenever that constituent did
not give to that mailing, which is nearly always. The missingness *is* the information,
and the honest reading is "did not respond", not "value unknown".

Two consequences. First, no imputation scheme belongs anywhere near these columns; task
6 records the handling recommendation. Second, this top-20 list is a poor summary of
where the dataset actually has gaps: `AGE` (24.80%) and `INCOME` (22.31%) do not appear
in it, and neither does any column whose missingness is genuine ignorance rather than a
non-event. Task 3 takes that up.

### Blank strings, which the NaN count misses entirely

65 of the 74 string columns contain blank or whitespace-only values, and none of them
appear in the NaN-based top 20 above, because `read_csv` reads a space as data. The
worst are `RECPGVG` (99.88% blank), `SOLP3` (99.81%), `MAJOR` (99.69%), `PLATES`
(99.41%), and `HOMEE` (99.07%). The plan cited `NOEXCH` as the example of the pattern;
`NOEXCH` is in fact the mildest case in the dataset, with 7 blank rows out of 95,412.
The inventory therefore reports `pct_blank` alongside `pct_missing`, and any statement
about missingness in the report has to account for both. Task 3 covers the substance.

---

## 2. Targets

**Frame: training split (76,329 rows).** Both targets are relationships with the
response, so the test split is not read here.

### TARGET_B: the class imbalance

| Quantity | Value |
| --- | --- |
| Positives (responded) | 3,874 |
| Total rows | 76,329 |
| Training response rate | 5.0754% |
| Full dataset response rate | 5.0759% |
| Difference | 0.0005 pp |
| Negatives per positive | 18.7 |

The stratified split did its job: the training rate matches the full-dataset rate to
four decimal places, so nothing about the class balance was distorted by splitting.

### What the imbalance means for evaluation

With 5.08% positives, a model that predicts "nobody responds" for every constituent is
94.92% accurate and completely useless: it finds no donors, which is the only thing the
model exists to do. Accuracy is therefore not reportable on its own, and CLAUDE.md's
rule against it is not a formality.

The number to beat is the AUPRC of a random-guessing baseline, which equals the
positive rate itself: **0.0508**. Any candidate model has to clear that, and clearing it
by a little is not the same as clearing it usefully. Recall matters alongside AUPRC
because the cost structure is asymmetric. Mailing a constituent who does not give wastes
roughly the price of a stamp; failing to mail one who would have given loses the whole
donation, which averages $15.52 here.

### TARGET_D: donation amounts among responders

Five-number summary for the 3,874 responders in the training split. Non-responders are
excluded because their `TARGET_D` is a structural 0.0, not a donation of nothing, and
averaging them in would report a mean of $0.79 that describes nobody.

| Statistic | Amount |
| --- | ---: |
| Minimum | $1.00 |
| 25th percentile | $10.00 |
| Median | $13.00 |
| 75th percentile | $20.00 |
| Maximum | $200.00 |
| Mean | $15.52 |
| Std. dev. | $12.41 |

Checked rather than assumed: 0 non-responders have a non-zero `TARGET_D`, and 0 have a
NaN. The `TARGET_B == 1` and `TARGET_D > 0` definitions of "responded" agree exactly, so
a later amount model can filter on either without a silent row loss.

The mean sits above the median ($15.52 against $13.00), the usual right skew of donation
amounts. The maximum of $200 is small enough that this campaign has no whale problem:
the largest single gift is 13 times the mean, not 1,000 times.

### Figure: `target_d_distribution_responders.png`

**Frame: training split, responders only (n=3,874).** Histogram of `TARGET_D` in $1
bins, with the median marked.

Donations pile up on round numbers, and the spikes dominate the shape. $10 is the mode
at 779 responders (20.1%), then $15 (473, 12.2%), $20 (466, 12.0%), $5 (395, 10.2%), and
$25 (313, 8.1%). Those five amounts alone account for **2,426 of 3,874 responders
(62.6%)**, and 2,637 (68.1%) gave an exact multiple of $5. The bins between the spikes
are not empty, but they are thin by comparison: $12 is the most common non-round amount
at 132 responders (3.4%).

This is a behavioral fact, not a data error, and it has a modeling consequence worth
recording now. `TARGET_D` is not a smooth continuous variable; it is closer to a choice
among a few round amounts with a scattering in between. A regression that assumes
continuity will put predictions in the gaps between the spikes, where almost no real
donor sits. The $1 bin width in the figure is deliberate: coarser bins would smooth the
spikes into a lognormal-looking hump and hide the entire point.

---

## 3. Missing data

**Frames: full dataset (95,412 rows) for how much is missing; training split (76,329
rows) for whether missingness predicts response.**

### Extent

| Category | Columns |
| --- | ---: |
| Have at least one NaN | 92 of 481 |
| Have at least one blank string | 65 of 481 |
| Have neither | 325 of 481 |

Two-thirds of the dataset (325 columns) is complete on both measures. The gaps are
concentrated, not spread thin, and the two kinds of gap barely overlap: the NaN columns
are numeric and the blank columns are strings.

### Figure: `missingness_top20.png`

**Frame: full dataset (n=95,412).** Horizontal bar of the 20 most-missing columns, with
percentage labels.

Discussed under task 1: the top 20 is entirely `RDATE_*`/`RAMNT_*` pairs, from 99.99%
(`RDATE_5`, `RAMNT_5`) down to 90.03% (`RDATE_21`, `RAMNT_21`), and each date column
matches its amount column exactly. Missing here means the constituent did not give to
that particular mailing. It is a recorded non-event, and reading these bars as "the data
is 99% broken" would be exactly wrong.

The figure's real use in the report is as a warning about what it does not show. Neither
`AGE` (24.80%) nor `INCOME` (22.31%) appears anywhere in it, and no blank-carrying
string column can appear in it at all, because `read_csv` reads a space as data rather
than as missing.

### AGE and INCOME

**Observed values, frame: full dataset.**

| Column | Missing | % missing | Observed range | Distinct |
| --- | ---: | ---: | --- | ---: |
| `AGE` | 23,665 | 24.80% | 1 to 98 | 96 |
| `INCOME` | 21,286 | 22.31% | 1 to 7 | 7 |

`AGE` has no zero-coded values, so the missingness is honest NaN rather than a sentinel
hiding in the range. `INCOME` is not a dollar amount: it is a 7-level ordinal bracket,
which limits how much can be said about a constituent's actual means. The distributions
of the observed values are plotted under task 4, which covers demographics as a group.

**AGE missingness is `DOB` missingness.** 23,661 rows have `DOB = 0` and `AGE` missing;
4 rows have `AGE` missing with a real `DOB`; 0 rows have `DOB = 0` and an `AGE`. `AGE` is
derived from `DOB`, and `DOB = 0` is the vendor's way of writing "unknown" in a numeric
field. The two columns carry one fact, not two, and imputing `AGE` while leaving `DOB` as
a raw number would put the same made-up value into the model twice.

### Does missingness predict response?

**Frame: training split (n=76,329).** Overall training response rate: 5.075%.

| Column | Group | n | Response rate | Difference | chi-square | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AGE` | missing | 18,915 | 4.880% | -0.260 pp | 1.945 | 0.1631 |
| `AGE` | present | 57,414 | 5.140% | | | |
| `INCOME` | missing | 17,027 | 5.227% | +0.195 pp | 1.005 | 0.3160 |
| `INCOME` | present | 59,302 | 5.032% | | | |

**No.** Neither difference is significant at the 0.05 level, and they point in opposite
directions: constituents with a missing `AGE` respond slightly *less* often, those with a
missing `INCOME` slightly *more*. Both gaps are under a third of a percentage point on a
5% base rate. The plan made the indicator-flag recommendation conditional on this test,
and the test came back null, so the honest answer is that response-rate evidence does not
support adding missingness flags for these two columns.

That is a weaker claim than "the flags are useless", and the distinction matters. A
chi-square on the marginal rate cannot see an interaction, so missingness could still
carry signal conditional on other features. The recommendation in task 6 is therefore to
try the flags and let cross-validation decide, not to assume they help and not to rule
them out on this evidence.

### Figure: `response_rate_by_missingness_age_income.png`

**Frame: training split (n=76,329).** Response rate for missing versus present, with the
overall 5.08% rate as a dashed reference line.

All four bars sit within a quarter of a percentage point of the reference line, which is
the finding. The y axis starts at zero and runs to 8% rather than zooming into the
4.8-5.3% range: a zoomed axis would turn four near-identical bars into two dramatic
contrasts and manufacture an effect the chi-square says is not there. This is the
"trustworthy axes" standard doing actual work rather than being a style note.

### Disguised missingness

**Frame: full dataset.** The blank-string pattern affects 65 of 74 string columns. Worst
12:

| Column | % blank | Group | Column | % blank | Group |
| --- | ---: | --- | --- | ---: | --- |
| `RECPGVG` | 99.88 | id_admin | `CHILD03` | 98.80 | demographics |
| `SOLP3` | 99.81 | id_admin | `MAILCODE` | 98.53 | id_admin |
| `MAJOR` | 99.69 | id_admin | `PVASTATE` | 98.47 | id_admin |
| `PLATES` | 99.41 | interests_overlay | `KIDSTUFF` | 98.39 | interests_overlay |
| `HOMEE` | 99.07 | interests_overlay | `CHILD07` | 98.36 | demographics |
| `CARDS` | 98.91 | interests_overlay | `RECSWEEP` | 98.31 | id_admin |

These are not all the same thing, and the report should not treat them as one problem.
Most are flag columns where blank is a real value meaning "no": `RECSWEEP` blank means
"not a sweepstakes donor", and `PLATES` blank means "no collector-plate interest
recorded". For those, blank is informative and imputing it would be a mistake. Others are
genuine unknowns: `HOMEOWNR` is 23.3% blank alongside `H` and `U` codes, and a blank
there means the vendor did not know, not that the person owns no home. `GEOCODE` at 84.0%
blank is a third case, a mostly-unpopulated column.

Telling these apart needs the data dictionary, not inference from the frequencies, and
that is a task 6 handling recommendation rather than something this pass decides.

**Numeric columns hide missingness too**, by coding it as 0 rather than NaN:

| Column | Rows coded 0 | % | Real range otherwise |
| --- | ---: | ---: | --- |
| `DOB` | 23,661 | 24.80% | 1 to 9710 (YYMM) |
| `FISTDATE` | 2 | 0.00% | 4912 to 9603 (YYMM) |

A YYMM date of 0 is not a date. `DOB = 0` is the `AGE` missingness described above,
wearing a different costume. `FISTDATE = 0` affects 2 rows and is negligible, but it is
the same defect and is listed so nobody rediscovers it later as a mysterious outlier.
Any downstream code that converts these fields to dates or treats them as numbers has to
handle the zeros first, because a model given `DOB = 0` will read it as a birth date in
1900, not as a missing value.

---

## 4. Distributions of key features

**Frame: training split (76,329 rows)** throughout. These are structural facts the
peeking rule would permit on the full dataset, but they are computed on training anyway
so that every distribution described here matches the rows the models will actually see.

### Demographics

**`AGE`, observed values only (n=57,414; 18,915 missing).**

| Statistic | Value |
| --- | ---: |
| Minimum | 1 |
| 25th percentile | 48 |
| Median | 62 |
| 75th percentile | 75 |
| Maximum | 98 |
| Mean | 61.6 |

Figure: `age_distribution.png`. The constituent base is old. Half are over 62, a quarter
are over 75, and the modal two-year bin is 76-78 with 2,556 people. For the Foundation
scenario this is the sharpest limitation of the proxy: a 1998 veterans-charity mailing
list skews decades older than the families of children with a rare congenital disorder,
so the age-response relationship found here should not be read as transferable. The
model's method transfers; this particular coefficient does not.

40 rows (0.070% of observed) have an `AGE` under 20, of which 15 are 5 or under and 9
are exactly 1. A direct-mail donor file with 9 one-year-old donors has bad data, not
infant philanthropists. The count is too small to matter for a model and too obviously
wrong to leave unmentioned; it is carried into task 6 and excluded from the age bands in
task 5.

**`GENDER` and `INCOME`.** Figure: `gender_income_counts.png`.

| `GENDER` | Count | % |
| --- | ---: | ---: |
| F | 41,014 | 53.73 |
| M | 31,282 | 40.98 |
| (blank) | 2,360 | 3.09 |
| U | 1,381 | 1.81 |
| J | 290 | 0.38 |
| C | 1 | 0.00 |
| A | 1 | 0.00 |

`F` and `M` cover 94.71%. The rest is a blank (unknown), `U` (explicitly "unknown", so
the column has two different ways to say the same thing), `J` (joint account, a household
rather than a person), and two codes appearing once each. `C` and `A` at one row apiece
are the signature of data entry error. Any encoding of this column has to decide what to
do with a category of size 1, and a naive one-hot encoding would hand the model a feature
that fires for exactly one training row.

`INCOME` is a 7-level ordinal bracket, not a dollar amount. The largest category is
missing (17,027, 22.31%), bigger than any real bracket; the largest observed bracket is 5
(12,272, 16.08%). The distribution across the observed brackets is fairly flat, ranging
from 5,986 (bracket 7) to 12,272 (bracket 5).

### Giving history

| Column | Min | Median | Mean | Max | Skew | Zeros |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LASTGIFT` | 0.00 | 15.00 | 17.32 | 1,000.00 | 17.3 | 304 |
| `AVGGIFT` | 1.29 | 11.67 | 13.33 | 1,000.00 | 27.5 | 0 |
| `NGIFTALL` | 1.00 | 7.00 | 9.60 | 237.00 | 2.1 | 0 |
| `RAMNTALL` | 13.00 | 78.00 | 104.41 | 9,485.00 | 13.8 | 0 |

Figure: `giving_history_distributions.png`, small multiples with the raw scale on top and
the log scale below.

The plan predicted strong right skew and log-scale views would be needed. Both hold, and
the raw row of the figure shows why: at raw scale every panel collapses into a single bar
at the left with a flat line running to the maximum, which is a true picture that
communicates nothing. On the log scale the bulk separates into a roughly lognormal hump
for the three dollar-valued columns. `NGIFTALL` is the exception and stays spiky on both
scales, because it is a discrete count with 87 distinct values and real mass on the small
integers (7,918 constituents have given exactly once, 6,203 exactly twice).

**Outliers: real donors, not errors.** The largest values are `MAXRAMNT` $5,000 then
$1,000 and $1,000; `LASTGIFT` $1,000, $1,000, $563; `RAMNTALL` $9,485, $5,675, $3,985.
These are plausible: a $9,485 lifetime total across a giving history spanning to 1975 is
an ordinary major donor, not a decimal point in the wrong place. Nothing is removed, per
the plan, and nothing should be: the recommendation in task 6 is to transform rather than
trim.

Two genuine oddities, both small. `LASTGIFT` has 304 zeros, a "most recent gift" of
nothing, and one row of exactly $0.01, which is what stretches the `LASTGIFT` log panel
across two empty decades. `MINRAMNT` has 459 zeros. These look like placeholder values
rather than gifts, and they are listed in task 6.

### Census exemplars

Three of the 290 census columns, chosen to span the group's three scales rather than for
interest. Figure: `census_exemplars.png`.

| Column | Min | Median | Max | Rows = 0 | What it is |
| --- | ---: | ---: | ---: | ---: | --- |
| `POP901` | 0 | 1,562 | 98,701 | 623 | Neighborhood population (count) |
| `MALEVET` | 0 | 31 | 99 | 2,617 | Male veterans in neighborhood (%) |
| `HV1` | 0 | 737 | 6,000 | 875 | Median home value (dollars, hundreds) |

The group is not homogeneous despite loading as 290 near-identical numeric columns. It
mixes unbounded counts (`POP901` to 98,701), percentages capped at 99, and dollar amounts
in hundreds. Any scaling decision has to be made per sub-kind, not for "the census block"
as one thing.

Every exemplar has a zero spike, and the zeros are suspicious in the same way as `DOB = 0`.
A neighborhood with a population of 0 and a median home value of $0 does not exist; these
are almost certainly unmatched zip codes recorded as zero rather than as missing. That
matters more here than elsewhere because it is 290 columns' worth of the same problem,
and NaN-based missingness reporting cannot see any of it. `MALEVET = 0` is the ambiguous
case: a neighborhood genuinely can have no male veterans, so zero is not automatically
missing there. Task 6 records this.

---

## 5. Relationships with response

**Frame: training split (76,329 rows).** Overall training response rate: 5.075%. All
figures in this section carry 95% Wilson intervals, and every bar is labeled with its
segment size.

### Response by RFA_2F (gift frequency)

Figure: `response_rate_by_rfa2f.png`.

| `RFA_2F` | n | Responders | Rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| 1 (fewest gifts) | 38,151 | 1,445 | 3.788% | [3.60%, 3.98%] |
| 2 | 16,497 | 845 | 5.122% | [4.80%, 5.47%] |
| 3 | 12,165 | 788 | 6.478% | [6.05%, 6.93%] |
| 4 (most gifts) | 9,516 | 796 | 8.365% | [7.83%, 8.94%] |

Monotonic and clean: every step up in frequency raises the response rate, the top band
responds **2.21x** as often as the bottom, and no two confidence intervals overlap. This
is the strongest single signal in the EDA.

### Response by RFA_2A (gift amount)

Figure: `response_rate_by_rfa2a.png`.

| `RFA_2A` | n | Responders | Rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| D (smallest gifts) | 5,952 | 560 | 9.409% | [8.69%, 10.18%] |
| E | 17,255 | 1,120 | 6.491% | [6.13%, 6.87%] |
| F | 37,643 | 1,664 | 4.420% | [4.22%, 4.63%] |
| G (largest gifts) | 15,479 | 530 | 3.424% | [3.15%, 3.72%] |

Monotonic in the **opposite** direction, and stronger: the smallest-gift band responds
**2.75x** as often as the largest, intervals again non-overlapping. Constituents who have
historically given small amounts are substantially more likely to give again.

This is the most useful finding in the EDA and the most easily misread. It does not mean
small donors are worth more; it means they respond more often. The two targets pull
against each other: `RFA_2A = D` maximizes the chance of a response, and the same band by
construction gives the least when it does respond. Any outreach policy built on `TARGET_B`
alone will systematically prefer the constituents who give least. This is precisely why
the campaign has two targets, and why an expected-value framing (`TARGET_B` probability
times `TARGET_D` amount) is the honest way to combine them. That is a modeling decision
and out of scope for this pass, but the EDA is where the tension becomes visible, so it
is recorded here.

### Response by AGE band and INCOME

Figures: `response_rate_by_age_band.png`, `response_rate_by_income.png`.

| AGE band | n | Rate | 95% CI |
| --- | ---: | ---: | --- |
| 20-29 | 917 | 4.580% | [3.41%, 6.13%] |
| 30-39 | 4,957 | 4.075% | [3.56%, 4.66%] |
| 40-49 | 9,589 | 4.828% | [4.42%, 5.28%] |
| 50-59 | 10,539 | 5.190% | [4.78%, 5.63%] |
| 60-69 | 9,996 | 5.512% | [5.08%, 5.98%] |
| 70-79 | 11,988 | 5.831% | [5.43%, 6.26%] |
| 80-99 | 9,388 | 4.708% | [4.30%, 5.16%] |

Response climbs from the 30s through the 70s, peaks at 5.831% in the 70-79 band, then
falls back in the 80s. Spread is 1.43x, well short of the RFA signals. The bands are
decade bins as the plan specified, with two exclusions: 18,915 rows where `AGE` is missing
and the 40 rows with an `AGE` under 20. The under-20 rows would have shown a 12.5%
response rate from 5 responders out of 40, which would have been the tallest bar on the
chart and pure noise resting on data that is wrong anyway.

| `INCOME` | n | Rate | 95% CI |
| --- | ---: | ---: | --- |
| 1 (lowest) | 7,212 | 4.146% | [3.71%, 4.63%] |
| 2 | 10,486 | 4.873% | [4.48%, 5.30%] |
| 3 | 6,877 | 5.046% | [4.55%, 5.59%] |
| 4 | 10,192 | 4.974% | [4.57%, 5.41%] |
| 5 | 12,272 | 5.166% | [4.79%, 5.57%] |
| 6 | 6,277 | 5.401% | [4.87%, 5.99%] |
| 7 (highest) | 5,986 | 5.797% | [5.23%, 6.42%] |

A gentle rise, 1.40x from bracket 1 to bracket 7, and not strictly monotonic (bracket 4
sits below bracket 3). Every band is within 0.93 percentage points of the overall rate,
and most of the confidence intervals overlap. `INCOME` is a weak signal on its own.

Both of these are demographic segments, and both need the fairness treatment the proposal
commits to. The direction matters ethically: response rises with income, so a model
trained to maximize response will preferentially target wealthier constituents. For
outreach efficiency that is defensible. If a score built this way were ever allowed to
influence who gets *services*, it would systematically disadvantage the poorest families,
which is the exact failure the project's ethics constraints exist to prevent. The
constraint is not decorative, and this table is the reason.

### Correlations with response

Figure: `correlation_giving_history.png`. Pearson correlations among the numeric
giving-history features plus `TARGET_B`. Because `TARGET_B` is binary, its row is a
point-biserial correlation, computed as Pearson.

| Feature | r with `TARGET_B` |
| --- | ---: |
| `CARDGIFT` | +0.0528 |
| `NGIFTALL` | +0.0498 |
| `RAMNTALL` | +0.0137 |
| `TIMELAG` | -0.0115 |
| `MAXRAMNT` | -0.0173 |
| `MINRAMNT` | -0.0323 |
| `AVGGIFT` | -0.0345 |
| `LASTGIFT` | -0.0371 |

Every one is under 0.06 in absolute value. The strongest is `CARDGIFT` at +0.0528. Read
literally, no giving-history feature has a meaningful linear association with response.

That reading would be a mistake, and the figure is included partly to make the mistake
visible. `RFA_2F` sorts response from 3.8% to 8.4% and `RFA_2A` from 9.4% to 3.4%, both
monotonically, and those two columns are built from the same giving history these
correlations are computed on. A correlation near zero against a target that is 95% zeros
does not mean no relationship; it means the relationship is not linear in the raw
feature. This is the concrete argument for the plan's model lineup: logistic regression
on raw features is a baseline that should be expected to struggle, and the tree-based
models are there because the signal is in the bands, not the slopes.

The off-diagonal is the more actionable part. `NGIFTALL` and `CARDGIFT` correlate at
**0.91**, and `AVGGIFT` correlates at 0.79 with `LASTGIFT`, 0.76 with `MINRAMNT`, and 0.75
with `MAXRAMNT`. The giving-history block carries far fewer independent facts than its
column count suggests, which matters for the logistic regression baseline's coefficient
stability. Task 6 records it.

### Strongest candidate signals, in plain language

1. **`RFA_2A`, past gift amount band.** 9.409% response for the smallest-gift band
   against 3.424% for the largest, 2.75x, monotonic across four bands, non-overlapping
   intervals. Smaller past gifts predict more frequent response.
2. **`RFA_2F`, past gift frequency.** 3.788% to 8.365%, 2.21x, monotonic,
   non-overlapping intervals. More past gifts predict response.
3. **`AGE`**, 1.43x across decade bands, peaking at 70-79 (5.831%).
4. **`INCOME`**, 1.40x, rising with bracket but not strictly monotonic and with wide
   interval overlap.

The two RFA components are the signals worth building on, and both are derived from
giving behavior rather than from who the constituent is. That is a good property for this
project: the strongest predictors are things constituents *did*, not demographics they
were born into. Neither raw giving-history correlations nor the 290-column census block
produced anything comparable at this stage.
