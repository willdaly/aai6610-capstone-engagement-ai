# AAI6610 Capstone: Engagement Prediction + Family Navigation Assistant

Coursework for AAI6610 Applied Machine Learning (Northeastern University, Summer 2026).
The assigned scenario is the Sturge-Weber Foundation, a patient advocacy nonprofit for a
rare neurological disorder. The project has three parts: a model that predicts which
constituents respond to an outreach campaign, trained on a public direct-mail dataset
used as a proxy; a retrieval-grounded Q&A prototype over the Foundation's public website
content that escalates anything out of scope to staff; and a strategic roadmap covering
adoption, feasibility, risks, and data gaps. Predictive scores here are for outreach
efficiency only and must never gate any family's access to services.

**This is an academic project. It is not affiliated with or endorsed by the Sturge-Weber
Foundation, and the Foundation provides no data.** The prototype is not a source of
medical advice.

## Setup

Requires Python 3.11 or newer (developed on 3.14.3, macOS Apple Silicon). Create and
activate a virtual environment first, since a system Python installed via Homebrew will
refuse to install into itself:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Then, from a fresh clone, three commands take you to passing tests:

```bash
bash model/download_data.sh        # ~117 MB into data/, verifies size, safe to re-run
pip install -r requirements.txt
pytest                             # 33 tests, ~9s
```

## Data

KDD Cup 1998 direct-mail dataset, UCI ML Repository, CC BY 4.0. The learning set is
95,412 rows by 481 fields. The response target `TARGET_B` is 5.08% positive (4,843
rows), which is why the evaluation uses AUPRC and recall rather than accuracy.

The download script pulls from a GitHub mirror verified byte-identical to the UCI copy,
so a run does not depend on UCI being up. The data is gitignored and never committed.
The dataset sponsor was a national veterans nonprofit; the original terms ask
educational users not to name it, so the reports refer to it that way.

Citation: Parsa, I. (1998). KDD Cup 1998 [Data set]. <https://doi.org/10.24432/C5401H>

## Layout

```text
model/        # download script, data loading, training pipeline, evaluation
assistant/    # scraper, corpus, RAG prototype, eval question set (not started)
docs/         # proposal, roadmap drafts, figures
data/         # gitignored; created by model/download_data.sh
```

`CLAUDE.md` holds the conventions that apply across the repo (split settings, metrics,
ethics constraints, writing style). It is authoritative. Each directory's README says
what belongs in it.
