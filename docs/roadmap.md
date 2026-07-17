# Strategic AI Roadmap

Three horizons for adopting the two builds in this repo: the engagement prediction model
and the family navigation assistant. Every claim traces to a number in
`docs/modeling_findings.md`, `docs/eda_findings.md`, or `docs/assistant_findings.md`. The
work used a public proxy dataset (a national veterans nonprofit's direct-mail file), not
the Foundation's own data, so the numbers illustrate what the method buys rather than
forecast results for the Foundation.

## Short term (0 to 6 months): pilot with humans in the loop

**Recommendation.** Use the model to rank constituents for one real outreach campaign,
mail the top fraction, and A/B test it against current practice, with staff reviewing the
generated list before anything is sent. Separately, put the assistant in front of staff
only, as a drafting aid for answering families, not as a public-facing bot.

**What it builds on.** On the held-out test split the best model reaches AUPRC 0.0881
against a 0.0508 random-guessing floor (1.73x), and at the chosen threshold it reaches
91.64% recall while contacting 83.2% of the list (`modeling_findings.md`). The honest
framing matters: mailing everyone already pays, because the 5.08% base rate clears the
$0.68 / $15.52 break-even of 4.38%, so the model's contribution is trimming the least
responsive roughly 17% for a net-revenue gain of about +8.3% ($212.68 on 19,083 rows), not
a dramatic lift. The assistant hits 100% on its two behavior guarantees (medical refusal,
out-of-scope escalation) and hit@3 of 0.80 and 0.91 on the two retrieval categories
(`assistant_findings.md`), which is adequate for a staff member who verifies before
replying.

**Effort and feasibility.** Low. The pipeline exists, is deterministic (seed 2026), and
runs end to end in about 3m37s on a laptop. The assistant runs offline with no API key. No
new infrastructure.

**Risks.** The three models are separated by 0.0038 AUPRC, inside their own fold-to-fold
SD of 0.004 to 0.006, so the model ranking is within noise. The scores are not calibrated
(see medium term), so rank with them, do not report them as probabilities. The proxy
dataset is not the Foundation's population, so transfer is unproven. The assistant still
misses some navigation queries (a "how do I contact" question retrieves the privacy-policy
page, per the findings).

**Governance.** Scores inform outreach efficiency only and never gate any family's access
to services. A human reviews every mailing list. Keeping the assistant staff-internal means
a wrong retrieval is caught before it reaches a family. No personal data is collected.

## Medium term (6 to 18 months): calibration, expected value, better retrieval

**Recommendation.** Calibrate the model's probabilities (Platt or isotonic on the
validation carve-out) and rank by expected value, calibrated probability times predicted
gift, rather than by raw score. Add embedding retrieval to the assistant to lift its hit@3
ceiling, keeping BM25 as a fallback. Expand the evaluation set with questions staff
actually field.

**What it builds on.** The best model's probabilities are not calibrated: predictions span
0.33 to 0.65 while observed response is 2 to 10%, inflated by roughly an order of magnitude
by class weighting, though the ranking stays sound (`modeling_findings.md`). Calibration is
the prerequisite for expected-value ranking, which is worth doing precisely because
response rate and gift size invert: the smallest-gift RFA band responds 2.75x as often as
the largest (`eda_findings.md`), so ranking on response alone favors the constituents who
give least. On the assistant, category-1 hit@3 of 0.80 and the documented BM25 misses
(short factual questions about types, research, and books) are what embeddings would target.

**Effort and feasibility.** Moderate. Calibration is a small scikit-learn addition.
Embeddings add a dependency and an index build, which the assistant plan deliberately
deferred; introduce them behind the existing retrieval interface so the change is
measurable.

**Risks.** Embeddings add infrastructure, cost, and a semantic-drift failure mode; hit@3
must be re-measured and the grounded-or-escalated behavior preserved. Calibration can shift
the operating threshold, so the net-revenue choice is re-run. Any expanded eval set must
keep categories 3 and 4 at 100%, which the existing behavior tests already enforce.

**Governance.** Expected-value ranking is still outreach only. Embedding models run over
the public corpus, never family data. The medical-refusal and escalation guarantees stay
as regression tests so a retrieval change cannot quietly weaken them.

## Long term (18+ months): Foundation-owned data, privacy by design

**Recommendation.** If the Foundation wants prediction on its own constituents, build a
Foundation-owned engagement dataset under privacy by design (de-identified, consented,
minimal, access-controlled) to replace the proxy. Carry "a score never gates access to
services" forward as standing written policy, audited, not a per-project reminder.

**What it builds on.** The census experiment is the encouraging signal: adding 290
neighborhood columns changed AUPRC by -0.0014 (-1.6%), smaller than the fold SD, so
dropping them is effectively free (`modeling_findings.md`), which means a privacy-minimal
feature set costs little accuracy. The fairness slices show why governance is needed: the
INCOME contact-rate gap is 29.3 pp (bracket 1 at 61.8% versus bracket 6 at 91.1%), and the
lowest bracket also has the worst recall.

**Effort and feasibility.** High and multi-year. It needs data infrastructure, consent
flows, and legal review, and is contingent on organizational will and funding.

**Risks.** A rare-disease population is small and identifiable, so re-identification risk is
real, and a patient-advocacy organization holding predictive scores on families is
sensitive. The central danger is scope creep, outreach ranking drifting toward anything that
touches services.

**Governance.** Data minimization, purpose limitation, de-identification, and retention
limits, made easier by the census-free result. Fairness slices become an ongoing check, not
a one-time measurement. The standing rule, audited: scores inform outreach efficiency and
never access to care.
