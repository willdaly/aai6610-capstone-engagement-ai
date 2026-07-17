# Modeling Findings — TARGET_B Response Classifier

What the modeling phase specified in `docs/modeling_plan.md` actually found. Organized
as the plan's deliverable 3 asks: setup, model comparison, threshold and revenue
analysis, fairness slices, census experiment, limitations. Every number here comes from
a run of `python model/src/train.py` and is recorded in `docs/modeling/results.csv`;
nothing is estimated or carried over from documentation.

The dataset is a 1998 direct-mail campaign from a national veterans nonprofit, used as
a public proxy for constituent engagement. It is not Sturge-Weber Foundation data, and
this analysis is not affiliated with or endorsed by the Foundation.

**The short version.** The boosting model reaches an AUPRC of **0.0881 on the held-out
test split against a 0.0508 random-guessing floor, a lift of 1.73x**, and at its frozen
operating threshold it earns **$2,774.74 against $2,562.06 for mailing everyone, a gain
of $212.68 (+8.3%)**. Three results matter more than that headline:

1. Mailing the whole list is already profitable, so the model is trimming a thin margin
   rather than finding hidden donors. The naive 0.5 threshold is not merely suboptimal,
   it is *worse than mailing everyone*.
2. Adding the 290 census columns changes AUPRC by **-0.0014**, less than the
   fold-to-fold noise. The neighbourhood data buys nothing measurable, so excluding it
   costs nothing.
3. At the chosen threshold the model contacts the lowest income bracket **61.8%** of the
   time against **91.1%** for bracket 6, a **29.3 pp gap**, while that bracket responds
   at an above-average rate.

---

## 1. Setup

### Frames

Four frames, from `load_data.make_split` with the course defaults (80/20 stratified,
`random_state=2026`, 15% validation carved from the training portion). Which stage may
see which frame is the plan's central ground rule, so it is stated here and enforced in
code.

| Frame | Rows | Positive | Used for |
| --- | ---: | ---: | --- |
| `train_full` | 76,329 | 5.08% | Model comparison by 5-fold CV; the final refit |
| `train_inner` | 64,879 | 5.08% | The models used for threshold, calibration, fairness |
| `validation` | 11,450 | 5.07% | Threshold selection, calibration, fairness slices |
| `test` | 19,083 | 5.08% | Read exactly once, at the end |

`train_inner` and `validation` exist because a threshold chosen from a model that was
fitted on the rows it is scoring is chosen on training data. The models used for
everything in sections 3 and 4 are fitted on `train_inner` and never see a validation
row.

The test split was read once, by `evaluate_on_test`, after the model family, its
hyperparameters and the operating threshold were all fixed. This is asserted rather than
asserted-to: `TestFrameDiscipline` in `model/tests/test_train.py` parses `train.py` and
fails if any function other than `evaluate_on_test` names a test frame, if `main` calls
it more than once, or if any fit, tune or threshold call appears after it.

### Feature policy as built

The plan's policy, implemented in `model/src/features.py`. X starts as 478 raw columns
(481 minus `CONTROLN`, `TARGET_B`, `TARGET_D`), the policy selects and derives **29
columns**, and encoding produces a **50-column model matrix**.

| Group | In | What |
| --- | ---: | --- |
| `demographics` | 13 | All. `AGE`, `INCOME`, `WEALTH1/2`, `NUMCHLD` numeric; `GENDER`, `HOMEOWNR`, `AGEFLAG`, `CHILD03/07/12/18` one-hot; `DOB` as a duration |
| `giving_history` | 7 | `NGIFTALL`, `CARDGIFT`, `RAMNTALL`, `LASTGIFT`, `AVGGIFT`, `MINRAMNT`, `MAXRAMNT` |
| `promotion_history` | 6 | `NUMPROM`, `CARDPROM`, `NUMPRM12`, `CARDPM12`, `RFA_2F`, `RFA_2A` |
| `geography` | 1 | `STATE`, frequency-encoded on train |
| derived | 3 | `MONTHS_SINCE_LASTGIFT`, `MONTHS_SINCE_DOB`, `N_PROMO_RESPONSES` |

Out, with the plan's reasons: `census` (290, section 5 below), `interests_overlay` (33,
vendor provenance), `id_admin` (22), `ZIP` and the area codes, `RFA_2R` (constant), the
raw `RFA_3..RFA_24` codes, and the per-mailing `RDATE_*`/`RAMNT_*` pairs.

### Two things checked against the data rather than assumed

**The campaign date is 9706, and promotion 2 is the campaign itself.** `ADATE_2` is 9706
on 95,399 of 95,412 rows, which confirms the date the plan cites. It also means the
response to promotion 2 *is* `TARGET_B`, and the file ships no `RDATE_2` or `RAMNT_2`
column for exactly that reason. The `RAMNT_*` family starts at 3 and is entirely
historical, so `N_PROMO_RESPONSES` (the count of past mailings a constituent gave to)
is safe. `features._promo_response_columns` raises rather than counting a `RAMNT_2` if
one ever appears, because that column would be the target under another name.

**`DOB` is a date, not a number.** The plan says all 13 demographics are in, and also
that raw YYMM values must not reach a model. `DOB` is both a demographic and a YYMM
integer, so it is converted to months-before-campaign like `LASTDATE`, with the 0
sentinel (23,661 rows) nulled first. Left raw, a `DOB` of 0 reads as a birth date in
1900, and the gap from 9612 to 9701 reads as 89 months rather than one.

### Deviations from the plan, and why

Three, all small, all recorded here rather than left to be discovered in the diff.

- **Recency is in months, not days.** The plan asks for "recency in days derived from
  `LASTDATE`". `LASTDATE` is a YYMM field; days are not recoverable from it. Months are
  the finest honest unit.
- **The logistic C grid has five points, not four.** The first run chose C=0.01, the
  edge of the grid, and an optimum at a boundary is a grid that stopped too early. Adding
  0.001 answered the question (section 2).
- **Gradient boosting uses `class_weight="balanced"` rather than manual
  `sample_weight`.** The plan specifies weighting "via sample_weight", which is what this
  estimator needed before it had a `class_weight` parameter. This scikit-learn has one
  and converts it to balanced sample weights internally. Same decision, cost-sensitive
  weighting rather than resampling, with the weights computed inside each CV fold from
  that fold's own class rate instead of once from the whole training portion.

No grid was shrunk. The full script runs in **3m16s** on a laptop CPU, inside the plan's
"minutes, not hours" budget.

---

## 2. Model comparison

**Frame: training portion (n=76,329), stratified 5-fold CV, scored on average
precision.** Figure: `cv_auprc_by_model.png`.

| Model | CV AUPRC | SD | vs floor | Best params | Fits | Secs |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `dummy` | 0.0508 | 0.0000 | 1.00x | (none) | 5 | 5.4 |
| `logistic` | 0.0800 | 0.0058 | 1.57x | C=0.001 | 25 | 18.6 |
| `random_forest` | 0.0813 | 0.0056 | 1.60x | depth=8, n=500 | 30 | 108.2 |
| `hist_gradient_boosting` | **0.0838** | 0.0040 | **1.65x** | lr=0.1, depth=3 | 30 | 24.4 |

Every model clears the floor, so the phase has not failed, and the boosting model is
strongest, as the plan expected.

The dummy is worth a sentence of its own. It scores 0.050754 with a fold-to-fold SD of
0.000025, reproducing the training positive rate to four decimals. It is routed through
the same `GridSearchCV` code path as everything else precisely so that the floor is
measured rather than asserted, and the agreement says the CV scaffolding is measuring
what it claims to.

The three real models are separated by 0.0038 AUPRC, and the fold-to-fold SDs are 0.004
to 0.006. **The model ranking is inside its own noise.** Boosting wins, and the report
should not pretend the win is decisive.

### Every model chose the most-regularized setting its grid offered

`docs/modeling/cv_grid.csv` records all 19 grid points, not just the winners, which is
what makes this answerable. Logistic picked the smallest C, random forest the shallowest
depth, boosting the shallowest depth. Three models at three grid edges is a pattern, not
a coincidence, and the grid says the three are not the same story.

**The logistic C curve is flat**, so its boundary is immaterial:

| C | 0.001 | 0.01 | 0.1 | 1.0 | 10.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CV AUPRC | 0.0800 | 0.0799 | 0.0795 | 0.0793 | 0.0793 |

The whole range spans 0.0007 across four orders of magnitude of penalty, against a fold
SD of 0.0058. Nothing is being left on the table at C=0.001.

**The random forest depth curve is not flat**, and this one is a real limitation:

| max_depth | 8 | 16 | None |
| --- | ---: | ---: | ---: |
| CV AUPRC (n=500) | 0.0813 | 0.0713 | 0.0679 |

Depth 8 beats unlimited depth by 0.0134, more than twice the fold SD, and the trend is
steep and monotone toward shallower trees. Depth 8 is the shallowest the grid offered,
so the optimum may well sit below it and this grid did not find out. Boosting shows a
milder version of the same thing (depth 3 at 0.0838, depth 6 at 0.0810, unlimited at
0.0820), and its learning rates are indistinguishable at depth 3 (0.08377 at 0.05,
0.08379 at 0.1).

The plan puts hyperparameter searches beyond the small grids out of scope, so this was
recorded rather than chased. It is the first thing a next phase should pick up, and the
direction is consistent with what the EDA predicted: the signal is weak, every model
overfits it quickly, and the ones that do best are the ones allowed to do least.

---

## 3. Threshold and revenue analysis

**Frame: validation carve-out (n=11,450), model fitted on `train_inner` (n=64,879).**
Figure: `net_revenue_curve.png`.

### The economics that drive everything in this section

A contact costs $0.68. A responder gives $15.52 on average (findings task 2). So a
contact pays for itself whenever the response probability clears **0.68 / 15.52 =
4.38%**. The overall response rate is 5.08%.

**The base rate is above break-even.** Mailing the entire list is already profitable, by
a thin margin, and that single fact reframes the whole exercise. The model is not finding
donors who would otherwise be missed. It is trimming the part of the list that falls
below 4.38%, and there is not much of it.

### Threshold selection

| Rule | Threshold | Contacts | % list | Precision | Recall | Net revenue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max net revenue | 0.3836 | 9,734 | 85.0% | 5.58% | 93.46% | **$1,291.88** |
| Naive 0.5 | 0.5000 | 4,360 | 38.1% | 7.61% | 57.14% | $907.20 |
| Contact everyone | - | 11,450 | 100.0% | - | 100.00% | $1,111.00 |
| Contact no one | - | 0 | 0.0% | - | 0.00% | $0.00 |

The plan asks for the naive 0.5 threshold "to show why 0.5 is wrong here", and it is
wrong more thoroughly than that framing suggests. At 0.5 the campaign mails 38% of the
list, finds 57% of the responders, and earns **$907.20, which is $203.80 less than
mailing everyone and asking no questions**. A 0.5 threshold on a 5%-positive target with
class-weighted training is not a neutral default. It is an active loss.

### The peak of a revenue curve is a biased number, and the dummy measures the bias

The dummy scores every constituent identically, so its ranking is arbitrary and its
revenue curve is a random walk: up about $15 at each responder, down $0.68 at every
other contact. The maximum of a random walk is not zero, it is a positive excursion. So:

| | Net revenue | Gain over mailing everyone |
| --- | ---: | ---: |
| Contact everyone | $1,111.00 | - |
| Dummy, at the best point on its curve | $1,230.20 | **+$119.20 (pure noise)** |
| Boosting, at the best point on its curve | $1,291.88 | +$180.88 |

**The dummy knows nothing and still "beats" mailing everyone by $119.** That is what
picking a maximum out of 11,450 correlated candidates is worth to a model with no
information. Against that floor the boosting model's real edge on this frame is roughly
$62, not $181.

This is not an argument that the model is worthless, and it is not a reason to abandon
the plan's threshold rule. Selecting a threshold by maximizing validation revenue is
still the right way to pick an operating point; the resulting *revenue estimate* is
simply optimistic, because the same rows chose it and scored it. The plan already
handles this by freezing the threshold and evaluating once on test, where no maximum is
picked. Section 6 reports that number, and it is the one to quote.

The figure draws the dummy as this noise reference rather than as a fourth competitor,
for the same reason.

### Calibration

Figure: `calibration_best_model.png`. **Frame: validation carve-out, 10 quantile bins.**

The boosting model's probabilities are not calibrated and are not close. Predicted
values span 0.33 to 0.65 while the observed response rate in those same bins runs 2% to
10%. At the top bin a predicted 0.65 corresponds to an observed 0.10.

The cause is not a defect, it is `class_weight="balanced"` doing what it was asked to.
Weighting the 5.08% positive class up by roughly 9.8x moves the decision boundary, and
the predicted probabilities move with it. The reliability curve is monotone, which is the
part that matters here: the ranking is sound, and AUPRC and threshold selection both
depend only on the ranking.

The consequence is a caveat the report must carry. **These scores are not probabilities
and must never be read as "this family has a 38% chance of responding".** They are
ranking positions with a probability-shaped scale. Any use that needs a real
probability, for example an expected-value calculation multiplying the score by a
predicted gift amount, needs a calibration step first (`CalibratedClassifierCV` on a
held-out frame), which the plan puts out of scope for this phase.

---

## 4. Fairness slices

**Frame: validation carve-out (n=11,450), at the chosen threshold 0.3836.** Figure:
`fairness_slices.png`. Full table: `docs/modeling/fairness_slices.csv`.

Measurement only, per the plan: no mitigation in this phase, and anything stark gets
flagged rather than fixed. Segments partition the frame exactly, missing values included,
and `fairness_slices` raises if they ever do not. This is deliberate. An early version of
this table silently dropped the 2,873 validation rows with a missing `AGE`, a quarter of
the frame, which would have made every AGE row below a claim about a population the
campaign is not actually mailing.

Contact rate is the share of a segment that gets mailed. Recall is the share of that
segment's real responders the campaign reaches.

### INCOME: the stark one

| INCOME | n | Responders | Response rate | Contacted | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| (missing) | 2,547 | 131 | 5.14% | 89.4% | 96.2% |
| 1 (lowest) | 1,044 | 54 | **5.17%** | **61.8%** | **85.2%** |
| 2 | 1,636 | 71 | 4.34% | 78.4% | 88.7% |
| 3 | 1,024 | 55 | 5.37% | 87.3% | 92.7% |
| 4 | 1,494 | 63 | 4.22% | 88.8% | 92.1% |
| 5 | 1,857 | 93 | 5.01% | 88.7% | 98.9% |
| 6 | 941 | 51 | 5.42% | **91.1%** | 92.2% |
| 7 (highest) | 907 | 63 | 6.95% | 88.5% | 95.2% |

**Contact-rate gap: 29.3 pp**, bracket 1 against bracket 6. This is the starkest gap in
the phase and it needs to be in the report.

The reason it is stark rather than merely unequal is the middle column. Bracket 1
responds at **5.17%, above the overall validation rate of 5.07%**, and above brackets 2,
4 and 5, all of which get contacted 78% to 89% of the time. The model is not
under-contacting bracket 1 because bracket 1 does not respond. Its recall there is also
the worst of any income segment at 85.2%, against 98.9% for bracket 5, so the model is
both mailing that group least and missing most of the responders it does have.

The EDA predicted the direction and this quantifies the size. What the EDA could not
show is that the gap is larger than the response-rate differences justify.

For the assigned scenario, the governance line in `CLAUDE.md` is the whole point:
predictive scores inform outreach efficiency only, and must never gate any family's
access to services. A 29.3 pp contact-rate gap by income means one group gets less mail.
Under the constraint that is a defensible efficiency decision about postage. Without it,
the same score applied to service allocation would systematically disadvantage the
poorest families, and this table is what that failure would look like on the way in.

### AGE band

| AGE band | n | Responders | Response rate | Contacted | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| (missing) | 2,873 | 142 | 4.94% | 83.2% | 93.7% |
| 20-29 | 134 | 5 | 3.73% | 78.4% | 80.0% |
| 30-39 | 713 | 27 | 3.79% | 79.4% | 96.3% |
| 40-49 | 1,485 | 57 | 3.84% | 84.2% | 94.7% |
| 50-59 | 1,620 | 96 | 5.93% | 84.4% | 92.7% |
| 60-69 | 1,526 | 99 | 6.49% | 86.6% | 88.9% |
| 70-79 | 1,766 | 93 | 5.27% | 90.3% | 95.7% |
| 80-99 | 1,333 | 62 | 4.65% | 85.4% | 96.8% |

**Contact-rate gap: 12.0 pp**, 70-79 against 20-29. The pattern tracks the EDA's
age-response relationship (response climbs through the 70s) and is modest. The 2,873
rows with a missing `AGE` are contacted at 83.2%, near the bottom of the range, which is
worth noting: not knowing someone's age costs them mail.

For the Foundation scenario this is the slice where the proxy is least transferable. A
1998 veterans mailing list skews decades older than the families of children with a rare
congenital disorder, and the 20-39 bands here (the ones closest to the Foundation's
actual constituency) are both the smallest and the least contacted.

### GENDER

| GENDER | n | Responders | Response rate | Contacted | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| (missing) | 320 | 24 | 7.50% | 89.7% | 87.5% |
| F | 6,131 | 282 | 4.60% | 85.5% | 95.0% |
| J | 50 | 3 | 6.00% | 92.0% | 100.0% |
| M | 4,712 | 258 | 5.48% | 84.0% | 92.2% |
| U | 237 | 14 | 5.91% | 85.7% | 92.9% |

**Contact-rate gap: 5.7 pp** across segments of 100 rows or more. The smallest gap of
the three, and F against M is 1.5 pp. `J` (joint accounts, 50 rows) is below the size
where any number here is evidence: three responders, and its 100% recall moves in
33-point steps.

---

## 5. Census experiment

**Frame: training portion (n=76,329), the same 5-fold CV, `hist_gradient_boosting` at
its tuned parameters.** One run, as the plan specifies.

The only difference between the two rows is the 290 census columns, which take the model
matrix from 50 columns to 340. The hyperparameters are not re-tuned, because a
difference in score could then be the neighbourhood data or could be the new
hyperparameters, and the experiment exists to answer exactly one question.

| Feature set | CV AUPRC | SD | vs floor | Secs |
| --- | ---: | ---: | ---: | ---: |
| Baseline (no census) | **0.0838** | 0.0040 | 1.65x | 24.4 |
| Baseline + census (290 columns) | 0.0824 | 0.0039 | 1.62x | 14.4 |
| **Delta** | **-0.0014** | | **-1.6%** | |

**Adding the census block makes the model slightly worse, and the delta is smaller than
the fold-to-fold SD in either direction.** The honest statement is not "the census
hurts"; it is that 290 columns of neighbourhood description buy nothing this experiment
can measure, and the sign of the change is not resolvable at this sample size.

The plan asked for this delta "in both directions of the trade-off" so the report could
weigh predictive lift against fairness cost. **There is no trade-off to weigh.** The
exclusion is free.

That is a stronger result for the ethics section than a small positive delta would have
been, and it is worth being precise about why. The dataset is 60.3% census columns and
2.7% individual demographics (findings task 1). The obvious worry going in was that
most of the file's predictive power lives in the neighbourhood block, and that excluding
it on proxy-discrimination grounds would mean accepting a materially worse model. It
does not. Everything the model can find, it finds in what constituents *did*, which is
what the EDA predicted: the two strongest signals it found were `RFA_2A` and `RFA_2F`,
both derived from giving behavior rather than from who someone is or where they live.

Two limits on this claim, both real. It is one model at one setting, and a model tuned
for 340 columns might use them better than one tuned for 50. And "no measurable lift on
this proxy dataset" is not "no lift anywhere"; a different campaign with a different
target could differ. The claim here is bounded to what was run.

---

## 6. Test evaluation

**Frame: held-out test split (n=19,083). Read exactly once.** The model family, its
hyperparameters and the operating threshold were all fixed before this ran, and nothing
measured here fed back into any of them.

`hist_gradient_boosting`, learning rate 0.1, max depth 3, refitted on the full training
portion (n=76,329), evaluated at the threshold 0.3836 frozen from validation.

| Metric | Value |
| --- | ---: |
| **AUPRC** | **0.0881** |
| AUPRC floor (test positive rate) | 0.0508 |
| **Lift over floor** | **1.73x** |
| Contacts at the frozen threshold | 15,882 of 19,083 (83.2%) |
| Precision | 5.59% |
| **Recall** | **91.64%** |
| Net revenue | **$2,774.74** |
| Net revenue, mailing everyone | $2,562.06 |
| **Gain** | **+$212.68 (+8.3%)** |

The test AUPRC of 0.0881 is slightly above the CV estimate of 0.0838, comfortably inside
the 0.0040 fold-to-fold SD. Nothing needs explaining there.

**The +8.3% is the number to quote**, and it is worth being clear about why it is worth
more than the $180.88 from section 3. No maximum is picked here. The threshold arrived
fixed from a different frame, and these rows had no say in it, so +8.3% is an unbiased
estimate of what that threshold is worth on constituents the model has never seen. The
validation peak was the best point on a noisy curve chosen by the same rows that scored
it, and the dummy showed that $119 of that peak was available to a model that knew
nothing.

One wrinkle to state rather than hide. The threshold was chosen using a model fitted on
`train_inner` (64,879 rows), and applied to a model refitted on `train_full` (76,329
rows). The two are the same configuration but not the same fit, and the refitted model's
probability scale can shift slightly. This is what the plan specifies, and it is what a
real campaign would do (choose an operating point on held-out data, then ship a model
trained on everything available). The effect appears small: the frozen threshold
contacts 83.2% of test against the 85.0% it selected on validation.

---

## 7. Limitations

In roughly the order they would change a decision.

**The margin is thin, and the base rate does most of the work.** At $0.68 a contact and
a $15.52 mean gift, break-even is a 4.38% response rate and the list responds at 5.08%.
Mailing everyone earns $2,562.06 on the test split without any model at all. The model
adds $212.68 to that. It is a real gain, measured honestly on held-out data, but the
campaign's profitability is a property of the economics, not of the model, and a report
that leads with "1.73x lift over random" without that context is misleading by omission.

**The model ranking is inside its own noise.** Logistic 0.0800, random forest 0.0813,
boosting 0.0838, with fold SDs of 0.004 to 0.006. Boosting is the best of the three and
the evidence does not support a stronger statement than that.

**Three grids stopped at their edges, and one of them matters.** Random forest depth
runs 0.0813 at depth 8, 0.0713 at 16, 0.0679 unlimited, and depth 8 is the shallowest
offered. That trend has not bottomed out. Wider searches are out of scope for this phase
(plan), so this is a known gap, not a resolved question. The logistic C boundary is
flat and does not matter.

**The scores are not probabilities.** Class weighting inflates them by roughly an order
of magnitude (predicted 0.65 against an observed 0.10). Rankings are sound; any use
needing a real probability needs calibration first.

**A 29.3 pp income contact gap is measured and unmitigated.** By design in this phase.
It is larger than the response-rate differences justify, and bracket 1 responds above
average while being contacted least and having the worst recall of any income segment.

**`TARGET_B` alone prefers the constituents who give least.** The EDA's amount/frequency
inversion (the smallest-gift band responds 9.41%, the largest 3.42%) is not solved by
this phase, only worked around: net revenue enters at threshold selection, not in the
objective the model optimizes. A model that ranked by expected value (response
probability times predicted gift) would be a different and probably better answer. The
plan scopes that out, and section 3's calibration finding is the obstacle to doing it
naively: multiplying an uncalibrated score by a predicted amount multiplies the
miscalibration through.

**The proxy is old and the population is wrong.** A 1998 veterans direct-mail file
skews decades older than the families of children with a rare congenital disorder. The
method transfers. The coefficients do not, and neither does the 4.38% break-even, which
is a fact about 1998 postage and this campaign's gift sizes. The 20-39 age bands, the
ones closest to the Foundation's actual constituency, are the smallest and least
contacted segments in section 4.

**`INCOME` and `AGE` are 22.3% and 24.8% missing and are median-imputed.** The EDA
tested whether missingness predicts response and it does not (p=0.16 and p=0.32), which
is why there are no missingness flags. But a chi-square on the marginal rate cannot see
an interaction, and the fairness table shows the imputed groups behave distinctly:
missing-`INCOME` rows are contacted 89.4% of the time, near the top of the range, and
missing-`AGE` rows 83.2%, near the bottom.

**Nothing here is a fairness mitigation, and nothing here should gate services.**
Measurement only, per the plan. The governance constraint in `CLAUDE.md` is not
decorative: these scores exist to decide who receives mail, and the income gap in
section 4 is a concrete picture of what would go wrong if they were ever allowed to
decide who receives help.

---

## What this phase did not do

No calibration step, no expected-value model combining `TARGET_B` with `TARGET_D`, no
fairness mitigation, no wider hyperparameter searches, no resampling, no neural
networks, no assistant work. All out of scope per the plan.

The findings that most affect the next phase, in order:

1. Mailing everyone already pays. Any further work has to beat $2,562.06 on test, not
   $0, and the honest headline is +8.3%.
2. The census block buys nothing measurable (-0.0014 AUPRC), so the fairness-motivated
   exclusion is free.
3. The random forest depth grid stopped at its edge, and the trend toward shallower
   trees had not flattened.
4. The income contact gap is 29.3 pp and unmitigated.
5. The scores need calibration before any expected-value framing can use them.
