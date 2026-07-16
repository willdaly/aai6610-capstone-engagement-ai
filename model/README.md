# model/

Engagement prediction model (brief Area 1): predict which constituents respond to an
outreach campaign, using the KDD Cup 1998 direct-mail dataset as a public proxy.

What belongs here:

- `download_data.sh`: fetches the learning set into `data/`. Idempotent, verifies size.
- `src/`: importable pipeline code (data loading, preprocessing, training, evaluation).
- `tests/`: pytest tests for data loading and preprocessing.
- EDA and evaluation notebooks, shipped with outputs cleared.

What does not belong here: the dataset itself (`data/` is gitignored), and anything
about the family navigation assistant (that goes in `assistant/`).

Conventions that apply to everything in this directory, from CLAUDE.md:

- 80/20 stratified train/test split on `TARGET_B`, `random_state=2026`. Where a
  validation set is needed, carve 15% from the training portion.
- AUPRC and recall are the primary metrics for the imbalanced target. Accuracy is never
  reported on its own.
- scikit-learn is the default. No TensorFlow in this repo.
- Predictive scores inform outreach efficiency only. They must never gate any family's
  access to services.
