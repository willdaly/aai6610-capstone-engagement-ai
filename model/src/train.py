"""Train and compare the TARGET_B classifiers. Implements docs/modeling_plan.md.

Run it as a script:

    python model/src/train.py

It writes docs/modeling/results.csv, saves fitted pipelines to model/artifacts/
(gitignored), writes figures to docs/figures/, and prints the numbers that
docs/modeling_findings.md quotes. Deterministic: every seed is 2026.

The frames, and which one each stage is allowed to see:

    train_full (76,329)  train + validation rows. Model comparison by 5-fold CV, and
                         the final refit before the test evaluation.
    train_inner (64,879) train_full minus the validation carve-out. The models used
                         for threshold selection, calibration and fairness are fitted
                         here, so the validation rows are genuinely held out from them.
    validation (11,450)  Threshold selection, calibration, fairness slices.
    test (19,083)        Touched exactly once, by evaluate_on_test, called last from
                         main after every choice is frozen. Nothing else in this
                         module may read it.

That last rule is the plan's central ground rule, so it is a single function with the
word "test" in its name, called from one place, rather than a flag threaded through
the code where it could fire early by accident.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

# Non-interactive backend, set before pyplot is imported. Same reason as eda.py: this
# is a script, and Agg makes the PNG output identical whether or not a display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from eda import ACCENT, BASE, REFERENCE, apply_style, save_figure, style_axes  # noqa: E402
from features import build_pipeline, make_xy  # noqa: E402
from load_data import RANDOM_STATE, REPO_ROOT, load_raw, make_split  # noqa: E402

MODELING_DIR = REPO_ROOT / "docs" / "modeling"
RESULTS_PATH = MODELING_DIR / "results.csv"
ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"

# The campaign's documented mail cost per contact (plan, objective section). Net
# revenue for a contacted set is the TARGET_D of the contacted responders minus this
# per contact, including the contacts that do not respond.
MAIL_COST = 0.68

# AUPRC of a random baseline equals the positive rate (findings task 2). Every model
# is measured against this and the phase has failed if one does not clear it.
AUPRC_FLOOR = 0.0508

# Stratified because the target is 5.08% positive: unstratified folds would let the
# rate drift between folds and make the fold-to-fold spread partly an artifact of the
# splitting. Shuffled and seeded so the folds are the same on every run.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
CV_SCORING = "average_precision"

# Every figure this script writes. eda.py has its own list and writes into the same
# directory, so both lists together are the declared contents of docs/figures/ and a
# PNG in neither is a stray. Kept as a module constant for the same reason the EDA
# keeps one: the tests assert the directory against it without restating it.
EXPECTED_FIGURES = (
    "cv_auprc_by_model.png",
    "net_revenue_curve.png",
    "calibration_best_model.png",
    "fairness_slices.png",
)


@dataclass
class ModelSpec:
    """One model family: how to build it, what to tune, how to preprocess for it.

    Args:
        name: Row key in results.csv and label on every figure.
        estimator: An unfitted sklearn classifier.
        grid: Parameter grid, keyed by the pipeline's step-prefixed names.
        scale_numeric: Standardize numerics. True for the logistic model only.
        needs_sample_weight: Pass class-balancing weights at fit time, for estimators
            with no class_weight parameter.
        note: Anything the findings doc has to say about this row, e.g. a grid that
            was shrunk to stay inside the runtime budget.
    """

    name: str
    estimator: object
    grid: dict = field(default_factory=dict)
    scale_numeric: bool = False
    needs_sample_weight: bool = False
    note: str = ""


def model_specs() -> list[ModelSpec]:
    """The plan's model lineup, in the plan's order.

    No imblearn and no SMOTE: the CLIMB benchmark the proposal cites found naive
    rebalancing unreliable, and class weights are the course-consistent mechanism
    (plan, models section). No neural networks (CLAUDE.md).
    """
    return [
        ModelSpec(
            name="dummy",
            # Anchors AUPRC at the positive rate. It exists to be beaten: a model that
            # cannot clear a constant predictor has found nothing.
            estimator=DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        ),
        ModelSpec(
            name="logistic",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            # Regularization strength only. The EDA predicts modest performance here
            # (every linear correlation with TARGET_B is under 0.06, findings task 5),
            # so this model is the calibrated, interpretable floor rather than a
            # contender, and a wide grid would buy nothing.
            grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            scale_numeric=True,
        ),
    ]


def fit_weights(y: pd.Series) -> np.ndarray:
    """Class-balancing sample weights, the sklearn "balanced" formula.

    For estimators that have no class_weight parameter. Weight for a class is
    n_samples / (n_classes * n_class_samples), so at a 5.08% positive rate a responder
    counts about 9.8x a non-responder.
    """
    counts = y.value_counts()
    total = len(y)
    weights = {label: total / (len(counts) * count) for label, count in counts.items()}
    return y.map(weights).to_numpy()


def run_cv(spec: ModelSpec, X: pd.DataFrame, y: pd.Series) -> dict:
    """Tune one model by 5-fold CV on the training portion. Frame: train_full.

    Uses GridSearchCV even for a model with an empty grid, so every row of
    results.csv comes from the same code path and the dummy's CV AUPRC is measured the
    same way as everything else rather than asserted from the positive rate.

    Returns:
        A results row: name, best params, CV AUPRC mean/std, fit seconds, and the
        refitted best estimator under "estimator".
    """
    pipeline = build_pipeline(
        spec.estimator,
        scale_numeric=spec.scale_numeric,
        columns=list(X.columns),
    )

    fit_params = {}
    if spec.needs_sample_weight:
        fit_params["model__sample_weight"] = fit_weights(y)

    search = GridSearchCV(
        pipeline,
        param_grid=spec.grid,
        scoring=CV_SCORING,
        cv=CV,
        refit=True,
        n_jobs=-1,
    )

    started = time.perf_counter()
    search.fit(X, y, **fit_params)
    elapsed = time.perf_counter() - started

    best = search.cv_results_["params"].index(search.best_params_)
    return {
        "model": spec.name,
        "params": json.dumps(search.best_params_, sort_keys=True),
        "cv_auprc_mean": float(search.cv_results_["mean_test_score"][best]),
        "cv_auprc_std": float(search.cv_results_["std_test_score"][best]),
        "cv_fit_seconds": round(elapsed, 1),
        "grid_size": len(search.cv_results_["params"]),
        "note": spec.note,
        "estimator": search.best_estimator_,
    }


def report_cv(rows: list[dict]) -> None:
    """Print the CV comparison against the floor. Frame: train_full, 5-fold CV."""
    print("=" * 78)
    print("MODEL COMPARISON  Frame: training portion (n=76,329), stratified 5-fold CV.")
    print("=" * 78)
    print(f"\nAUPRC floor (random baseline = positive rate): {AUPRC_FLOOR:.4f}\n")
    print(f"  {'model':<12} {'CV AUPRC':>10} {'std':>8} {'vs floor':>10} {'fits':>6} {'secs':>7}")
    for row in rows:
        lift = row["cv_auprc_mean"] / AUPRC_FLOOR
        print(
            f"  {row['model']:<12} {row['cv_auprc_mean']:>10.4f} "
            f"{row['cv_auprc_std']:>8.4f} {lift:>9.2f}x "
            f"{row['grid_size'] * CV.n_splits:>6} {row['cv_fit_seconds']:>7.1f}"
        )
        print(f"  {'':<12} params: {row['params']}")
    print()


def write_results(rows: list[dict]) -> pd.DataFrame:
    """Write docs/modeling/results.csv, one row per model.

    The fitted estimator is dropped: results.csv is a table of numbers for the
    findings doc, and the pipelines go to model/artifacts/ instead.
    """
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "estimator"} for row in rows])
    frame.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)} ({len(frame)} rows)\n")
    return frame


def save_artifact(estimator, name: str) -> Path:
    """Persist a fitted pipeline to model/artifacts/ (gitignored)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(estimator, path)
    return path


def net_revenue(y_true: np.ndarray, amounts: np.ndarray, contacted: np.ndarray) -> float:
    """Net revenue for a contacted set, in dollars.

    Sum of TARGET_D over the contacted constituents who responded, minus the mail cost
    for every contact including the ones who did not respond. TARGET_D is 0.0 for
    non-responders (findings task 2, checked not assumed), so the first term is just
    the amounts of the contacted rows.
    """
    return float(amounts[contacted].sum() - MAIL_COST * contacted.sum())


def figure_cv_comparison(rows: list[dict]) -> None:
    """CV AUPRC per model against the floor. Frame: train_full, 5-fold CV."""
    fig, ax = plt.subplots(figsize=(9, 5))

    names = [row["model"] for row in rows]
    means = np.array([row["cv_auprc_mean"] for row in rows])
    stds = np.array([row["cv_auprc_std"] for row in rows])

    best = int(means.argmax())
    colors = [ACCENT if i == best else BASE for i in range(len(means))]
    ax.bar(
        names,
        means,
        color=colors,
        width=0.6,
        yerr=stds,
        capsize=4,
        error_kw={"ecolor": REFERENCE, "elinewidth": 1.2},
    )
    for index, (mean, std) in enumerate(zip(means, stds)):
        ax.text(index, mean + std + 0.002, f"{mean:.4f}", ha="center", fontsize=9)

    ax.axhline(AUPRC_FLOOR, color=REFERENCE, linestyle="--", linewidth=1.2)
    ax.text(
        len(names) - 0.4,
        AUPRC_FLOOR + 0.0015,
        f"random baseline {AUPRC_FLOOR:.4f}",
        ha="right",
        fontsize=8,
        color=REFERENCE,
    )

    style_axes(
        ax,
        title="Cross-validated AUPRC against the 5.08% random-guessing floor",
        xlabel="",
        ylabel="AUPRC (average precision)",
        frame="training portion (n=76,329), stratified 5-fold CV, error bars 1 SD across folds",
    )
    # Zero-based: the models sit close together and close to the floor, and a zoomed
    # axis would turn a difference of a few thousandths into an apparent chasm.
    ax.set_ylim(0, max(means + stds) * 1.35)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "cv_auprc_by_model.png")


def main() -> None:
    """Run the phase in plan order. The test split is read once, at the end."""
    apply_style()

    df = load_raw()
    train_full, test = make_split(df)
    train_inner, validation, test_check = make_split(df, validation=True)
    assert test.index.equals(test_check.index), "the two split calls must agree on test"

    X_train_full, y_train_full, _amounts_train_full = make_xy(train_full)

    print(
        f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.\n"
        f"  train_full  {len(train_full):>6,} rows ({train_full['TARGET_B'].mean() * 100:.2f}% positive)\n"
        f"  train_inner {len(train_inner):>6,} rows ({train_inner['TARGET_B'].mean() * 100:.2f}% positive)\n"
        f"  validation  {len(validation):>6,} rows ({validation['TARGET_B'].mean() * 100:.2f}% positive)\n"
        f"  test        {len(test):>6,} rows (untouched until the final evaluation)\n"
        f"  X carries {X_train_full.shape[1]} raw columns before the feature policy.\n"
    )

    rows = []
    for spec in model_specs():
        print(f"Fitting {spec.name} ...", flush=True)
        row = run_cv(spec, X_train_full, y_train_full)
        save_artifact(row["estimator"], row["model"])
        rows.append(row)
    print()

    report_cv(rows)
    figure_cv_comparison(rows)
    write_results(rows)


if __name__ == "__main__":
    main()
