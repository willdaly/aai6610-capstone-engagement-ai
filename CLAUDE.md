# CLAUDE.md — AAI6610 Capstone: Engagement Prediction + Family Navigation Assistant

## What this project is

Capstone for AAI6610 Applied Machine Learning (Northeastern University, Summer 2026,
instructor Mimoza Dimodugno). The assigned scenario is the Sturge-Weber Foundation, a
patient advocacy nonprofit for a rare neurological disorder. This is an academic project;
it is NOT affiliated with or endorsed by the Foundation, and the Foundation provides no
data. Keep that disclaimer in the README and any public-facing text.

Two builds plus a strategy document:

1. **Engagement prediction model** (brief Area 1): predict which constituents will
   respond to an outreach campaign, using a public proxy dataset.
2. **Family navigation assistant prototype** (brief Areas 2 + 8): a retrieval-grounded
   Q&A assistant over the Foundation's public website content, with escalation to staff
   for anything out of scope.
3. **Strategic AI roadmap**: short/medium/long-term adoption recommendations,
   feasibility, risks, data gaps.

The approved proposal (docs/) is the source of truth for scope. Do not expand scope
without the owner's say-so.

## Dataset (prediction model)

KDD Cup 1998 direct-mail dataset, UCI ML Repository, CC BY 4.0.
Citation: Parsa, I. (1998). KDD Cup 1998 [Data set]. https://doi.org/10.24432/C5401H

- Learning set: `cup98LRN.txt`, 95,412 rows x 481 fields (verified 2026-07-16;
  uncompressed size 117,167,952 bytes matches official docs).
- Targets: `TARGET_B` (binary response, 5.08% positive = 4,843 of 95,412) and
  `TARGET_D` (donation amount; responders: mean $15.62, median $13.00, max $200).
- Known missingness: AGE 24.8%, INCOME 22.3%. Handle explicitly; never silently drop.
- Download without touching UCI (mirror verified byte-identical):
  `curl -sL -o data/cup98lrn.txt.gz https://raw.githubusercontent.com/facebookresearch/metamulti/main/codes/cup98lrn.txt.gz && gunzip data/cup98lrn.txt.gz`
- Do NOT commit the data file. Add `data/` to .gitignore; commit a download script instead.
- In prose and reports, refer to the sponsor as "a national veterans nonprofit"
  (original dataset terms ask educational users not to name the sponsor org).

## Pipeline conventions (course-wide, do not deviate silently)

- 80/20 stratified train/test split, `random_state=2026`.
- Where a validation set is needed, carve 15% from the training portion.
- Never fabricate metrics; every number reported must come from an actual run.
- Evaluation for the imbalanced target: area under the precision-recall curve (AUPRC)
  and recall are primary; accuracy alone is never reported without them.
- Models to compare: logistic regression baseline, balanced random forest,
  cost-sensitive gradient boosting. Tune via cross-validation.
- Prefer scikit-learn. Avoid TensorFlow in this repo: the owner's machine is Apple
  Silicon and tensorflow-cpu has no wheels for it. If a Keras model ever becomes
  necessary, it gets trained elsewhere and only the artifact lands here.

## Assistant prototype conventions

- Retrieval-augmented generation over chunked public website / educational content from
  sturge-weber.org. Corpus lives in `assistant/corpus/` with a scraper script and a
  recorded scrape date; respect robots.txt and keep the corpus to public pages.
- Answers must ground in retrieved passages; out-of-scope questions escalate to a
  "contact the Foundation" response rather than a guess. This is a support tool for
  families of patients, including children: tone is plain, calm, and careful, and the
  assistant never gives medical advice, only directions to Foundation resources.
- Evaluate faithfulness and answer relevance (Ragas-style reference-free metrics) on a
  small hand-written question set committed to the repo.

## Ethics and governance constraints (graded, 50 rubric points across two criteria)

- Predictive scores inform outreach efficiency only. They must never gate any family's
  access to services, and the code/docs should make that impossible to misread.
- De-identified public data only. No collection of personal data in the prototype.
- Include fairness checks of model behavior across demographic segments in the
  evaluation notebook, and document model limitations in plain language.

## Repo structure

```
model/        # download script, EDA, training pipeline, evaluation
assistant/    # scraper, corpus, RAG prototype, eval question set
docs/         # proposal, roadmap drafts, figures
data/         # gitignored; created by the download script
```

## Engineering practice

- Versioned pipeline, tests for data loading and preprocessing, documented data
  dependencies (per Sculley et al., 2015, which the proposal cites).
- Notebooks ship clean top-to-bottom with outputs cleared; correctness verified by
  running component scripts.
- Plots: no pastel palettes, no chart junk; truthful axes (course visualization
  standards: trustworthy, effortless, elegant).

## Writing style for any prose in this repo

Direct, lightly hedged, no marketing language. No em-dashes (use commas, parentheses,
or sentence breaks). Avoid "delve," "leverage," "sharp," "clean," and similar filler.
Concrete examples over abstract framing.
