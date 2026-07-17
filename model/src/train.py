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
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from eda import (  # noqa: E402
    ACCENT,
    AGE_BAND_EDGES,
    BASE,
    REFERENCE,
    SECONDARY,
    apply_style,
    save_figure,
    style_axes,
)
from features import build_pipeline, make_xy  # noqa: E402
from load_data import RANDOM_STATE, REPO_ROOT, load_raw, make_split  # noqa: E402

MODELING_DIR = REPO_ROOT / "docs" / "modeling"
RESULTS_PATH = MODELING_DIR / "results.csv"
# Every grid point, not just the winner. results.csv answers "which model won"; this
# answers "by how much, and did the grid stop too early", which is the question a
# best-params-only table cannot. It costs nothing: the search already computed it.
GRID_PATH = MODELING_DIR / "cv_grid.csv"
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
        note: Anything the findings doc has to say about this row, e.g. a grid that
            was shrunk to stay inside the runtime budget.
    """

    name: str
    estimator: object
    grid: dict = field(default_factory=dict)
    scale_numeric: bool = False
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
            #
            # 0.001 is here because the first run picked 0.01, the edge of the grid,
            # and an optimum at a boundary is a grid that stopped too early rather than
            # an answer. Heavy regularization is also what the EDA predicts: the giving
            # summaries correlate up to r=0.91 (findings task 5), and correlated
            # predictors are what a penalty exists to control.
            grid={"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            scale_numeric=True,
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                # Single-threaded on purpose: GridSearchCV parallelizes across folds
                # with n_jobs=-1, and a forest that also grabs every core would
                # oversubscribe the machine and run slower than either alone.
                n_jobs=1,
            ),
            # Depth and estimators, as the plan specifies. Unlimited depth is in the
            # grid as the honest default rather than because it is expected to win: on
            # a 5%-positive target a fully grown forest memorizes the majority class
            # cheaply, and the CV score is what says whether it did.
            grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [8, 16, None],
            },
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            # The plan calls for class weighting here "via sample_weight", which is the
            # mechanism this estimator needed when it had no class_weight parameter.
            # This scikit-learn has one, and it does the same thing (it converts the
            # setting to balanced sample weights internally) with two advantages: the
            # weights are computed inside each CV fold from that fold's own class rate
            # rather than once from the whole training portion, and there is no fit
            # param to thread through GridSearchCV and get subtly wrong. Same decision
            # the plan made, cost-sensitive weighting rather than resampling, expressed
            # in the estimator's own vocabulary.
            #
            # Native NaN handling is worth noting: the median imputation upstream is
            # not doing this model any favours. It is there to keep every model on one
            # feature matrix, so the comparison is about the models rather than about
            # their preprocessing.
            estimator=HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            grid={
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 6, None],
            },
        ),
    ]


def fit_spec(spec: ModelSpec, params: dict, X: pd.DataFrame, y: pd.Series):
    """Build a fresh pipeline for one spec at fixed params and fit it to X, y.

    Always a new pipeline, never a refit of an estimator that is already fitted
    somewhere else. The phase fits each model on two different frames (train_inner for
    the validation analyses, train_full for the final test evaluation), and refitting
    one shared object in place would mean the second fit silently destroys the first.
    Whichever frame ran last would win, which is a bug that changes numbers without
    changing any visible code.

    Args:
        spec: The model family.
        params: Pipeline params, e.g. the CV-chosen {"model__C": 0.001}.
        X: Features, from make_xy.
        y: TARGET_B.
    """
    pipeline = build_pipeline(
        spec.estimator,
        scale_numeric=spec.scale_numeric,
        columns=list(X.columns),
    )
    pipeline.set_params(**params)
    return pipeline.fit(X, y)


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

    search = GridSearchCV(
        pipeline,
        param_grid=spec.grid,
        scoring=CV_SCORING,
        cv=CV,
        refit=True,
        n_jobs=-1,
    )

    started = time.perf_counter()
    search.fit(X, y)
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
        "cv_results": pd.DataFrame(
            {
                "model": spec.name,
                "params": [json.dumps(p, sort_keys=True) for p in search.cv_results_["params"]],
                "cv_auprc_mean": search.cv_results_["mean_test_score"],
                "cv_auprc_std": search.cv_results_["std_test_score"],
                "rank": search.cv_results_["rank_test_score"],
            }
        ),
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
    """Write docs/modeling/results.csv (one row per model) and cv_grid.csv (every point).

    The fitted estimator is dropped: results.csv is a table of numbers for the
    findings doc, and the pipelines go to model/artifacts/ instead.
    """
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    dropped = {"estimator", "cv_results"}
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in dropped} for row in rows])
    frame.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)} ({len(frame)} rows)")

    grid = pd.concat([row["cv_results"] for row in rows], ignore_index=True)
    grid.to_csv(GRID_PATH, index=False)
    print(f"Wrote {GRID_PATH.relative_to(REPO_ROOT)} ({len(grid)} rows)\n")
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
    # In the gap between the first two bars. Right-aligned at the axis edge collided
    # with the last bar, and every bar except the dummy's clears the line, so the gap
    # after the dummy is the one place on this axis that is reliably empty.
    ax.text(
        0.5,
        AUPRC_FLOOR + 0.002,
        f"random baseline {AUPRC_FLOOR:.4f}",
        ha="center",
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


# ---------------------------------------------------------------------------
# Threshold selection and the net-revenue lens (validation carve-out)
# ---------------------------------------------------------------------------
# The EDA found that response rate and gift size pull in opposite directions: the
# smallest-gift band responds 9.41% and the largest 3.42% (findings task 5). So a
# model chosen and thresholded on response probability alone systematically prefers
# the constituents who give least. The plan's resolution is to keep classification as
# the modeled task and bring the amounts back at threshold selection, through net
# revenue. This section is where that happens.


def revenue_curve(y: np.ndarray, amounts: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Net revenue as a function of how far down the ranked list you mail.

    Sorts by score descending and walks down the list, so row k of the result answers
    "if we contacted the top k constituents this model ranked, what would we have
    made". That is the quantity the campaign actually cares about, and it is what the
    net-revenue figure plots.

    Ties are broken by original row order (a stable sort). That matters for a constant
    scorer like the dummy, where every row ties and the "ranking" is arbitrary: the
    curve is then a straight line plus noise, and picking its maximum would be reading
    the noise. choose_threshold is only ever called for a real model for that reason.

    Args:
        y: True TARGET_B, 0/1.
        amounts: TARGET_D, aligned to y. 0.0 for non-responders.
        proba: Predicted probability of response.

    Returns:
        One row per k from 1 to n: contacts, contact_fraction, threshold (the score of
        the last constituent contacted), net_revenue, precision, recall.
    """
    order = np.argsort(-proba, kind="stable")
    amounts_sorted = amounts[order]
    y_sorted = y[order]
    contacts = np.arange(1, len(y) + 1)

    revenue = np.cumsum(amounts_sorted) - MAIL_COST * contacts
    responders = np.cumsum(y_sorted)

    return pd.DataFrame(
        {
            "contacts": contacts,
            "contact_fraction": contacts / len(y),
            "threshold": proba[order],
            "net_revenue": revenue,
            "precision": responders / contacts,
            "recall": responders / y.sum(),
        }
    )


def metrics_at_threshold(
    y: np.ndarray, amounts: np.ndarray, proba: np.ndarray, threshold: float
) -> dict:
    """Contact set, revenue and classification metrics at one probability threshold.

    Contacted means proba >= threshold, which is the rule a campaign would actually
    run. It is not identical to "the top k rows" when scores tie at the threshold, so
    the contact count is recomputed here rather than carried over from the curve.
    """
    contacted = proba >= threshold
    n_contacted = int(contacted.sum())
    hits = int(y[contacted].sum())
    responders = int(y.sum())
    return {
        "threshold": float(threshold),
        "contacts": n_contacted,
        "contact_fraction": n_contacted / len(y),
        "net_revenue": net_revenue(y, amounts, contacted),
        # Precision with nobody contacted and recall with nobody to find are both
        # undefined rather than zero. Zero would read as "the model got them all
        # wrong", which is a different claim from "the question does not apply here".
        "precision": hits / n_contacted if n_contacted else np.nan,
        "recall": hits / responders if responders else np.nan,
    }


def choose_threshold(y: np.ndarray, amounts: np.ndarray, proba: np.ndarray) -> dict:
    """The threshold maximizing net revenue on the validation carve-out.

    This is the plan's threshold rule, and it is selected on validation rows that the
    model scoring them was not fitted on. The chosen number is then frozen and applied
    to the test split once, at the end.
    """
    curve = revenue_curve(y, amounts, proba)
    best = curve.loc[curve["net_revenue"].idxmax()]
    return metrics_at_threshold(y, amounts, proba, float(best["threshold"]))


def report_thresholds(
    name: str, chosen: dict, naive: dict, everyone: float, dummy_peak: float, n: int
) -> None:
    """Print the chosen threshold against the naive 0.5, mailing everyone, and noise."""
    print("=" * 78)
    print(f"THRESHOLD SELECTION  Frame: validation carve-out (n={n:,}). Model: {name}.")
    print("=" * 78)
    print(
        f"\n  {'rule':<28} {'threshold':>10} {'contacts':>9} {'% list':>8} "
        f"{'precision':>10} {'recall':>8} {'net revenue':>13}"
    )
    for label, row in (("max net revenue", chosen), ("naive 0.5", naive)):
        print(
            f"  {label:<28} {row['threshold']:>10.4f} {row['contacts']:>9,} "
            f"{row['contact_fraction'] * 100:>7.1f}% {row['precision'] * 100:>9.2f}% "
            f"{row['recall'] * 100:>7.2f}% {row['net_revenue']:>13,.2f}"
        )
    print(
        f"  {'contact everyone':<28} {'-':>10} {n:>9,} {'100.0%':>8} {'-':>10} "
        f"{'100.00%':>8} {everyone:>13,.2f}"
    )
    print(
        f"  {'contact no one':<28} {'-':>10} {0:>9,} {'0.0%':>8} {'-':>10} "
        f"{'0.00%':>8} {0.0:>13,.2f}"
    )

    # How much of the chosen threshold's apparent advantage is the act of choosing.
    # The dummy ranks at random, so the best point on its revenue curve is what picking
    # a maximum out of 11,450 correlated candidates is worth to a model that knows
    # nothing. Any gain smaller than this is not a gain.
    print(
        f"\n  Gain over mailing everyone: ${chosen['net_revenue'] - everyone:,.2f}\n"
        f"  Same gain for the dummy, which ranks at random: ${dummy_peak - everyone:,.2f}\n"
        f"  The dummy's is pure selection noise, so the model's real edge on this frame\n"
        f"  is about ${chosen['net_revenue'] - dummy_peak:,.2f}. The test evaluation at the frozen\n"
        f"  threshold is the number that settles this; it picks no maximum.\n"
    )


def figure_net_revenue(curves: dict[str, pd.DataFrame], everyone: float) -> None:
    """Net revenue vs fraction of list contacted, per model. Frame: validation.

    The plan calls this the business argument for the whole project. It is, but not in
    the direction the plan anticipated, and the dummy curve is why.

    The dummy scores every constituent identically, so its "ranking" is arbitrary and
    its curve is a random walk: revenue drifts up by about $15 at each responder and
    down $0.68 at every other contact. The maximum of a random walk is not zero, it is
    a positive excursion, so the dummy peaks *above* mailing everyone while knowing
    nothing at all. That number is the noise floor for the peak statistic, and every
    other model's peak carries the same bias. Reading a model's peak as its business
    value without subtracting that floor would overstate the case badly, so the dummy
    is drawn as the reference it is rather than as a fourth competitor.

    The unbiased number is the one the test evaluation produces at the single frozen
    threshold, which is exactly why the plan freezes it first.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "logistic": SECONDARY,
        "random_forest": BASE,
        "hist_gradient_boosting": ACCENT,
    }

    for name, curve in curves.items():
        peak = curve.loc[curve["net_revenue"].idxmax()]
        is_dummy = name == "dummy"
        label = f"{name}: peaks ${peak['net_revenue']:,.0f} at {peak['contact_fraction'] * 100:.0f}%"
        if is_dummy:
            label += " (knows nothing: this is the noise floor)"
        ax.plot(
            curve["contact_fraction"] * 100,
            curve["net_revenue"],
            color=REFERENCE if is_dummy else palette.get(name, BASE),
            linewidth=1.8 if name == "hist_gradient_boosting" else 1.1,
            linestyle=":" if is_dummy else "-",
            label=label,
        )

    # Contact everyone is the right-hand endpoint every curve converges to: mail the
    # whole list and the ranking stops mattering. Drawn as a reference line because the
    # question the figure asks is how far above it a model can get, and the answer is
    # "less far than the peaks suggest".
    ax.axhline(everyone, color=REFERENCE, linestyle="--", linewidth=1.2)
    # Below the line and hard left. Every curve converges on this line at the right-hand
    # edge, so a label there sits on top of four of them; at the left the curves are
    # still climbing out of $0 and the space under the line is empty.
    ax.text(
        1,
        everyone - 75,
        f"contact everyone: ${everyone:,.0f}",
        fontsize=8,
        ha="left",
        color=REFERENCE,
    )
    ax.axhline(0.0, color=REFERENCE, linestyle="-", linewidth=1.0)
    ax.text(99, 25, "contact no one: $0", fontsize=8, ha="right", color=REFERENCE)

    style_axes(
        ax,
        title=(
            "Mailing everyone already pays, so the models only trim a thin margin.\n"
            "The dummy knows nothing and still peaks above it: that gap is selection noise"
        ),
        xlabel="Constituents contacted, ranked by predicted response (% of list)",
        ylabel="Net revenue (US dollars)",
        frame="validation carve-out (n=11,450), models fitted on train_inner (n=64,879)",
    )
    ax.set_xlim(0, 100)
    # Upper left is the one region no curve enters: they all start at $0 and climb.
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, "net_revenue_curve.png")


def figure_calibration(name: str, y: np.ndarray, proba: np.ndarray) -> None:
    """Reliability curve for the best model. Frame: validation carve-out."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Quantile bins, not uniform width: the scores bunch up, and uniform bins would put
    # almost every constituent in the first bin and leave the rest of the curve resting
    # on a handful of rows.
    observed, predicted = calibration_curve(y, proba, n_bins=10, strategy="quantile")

    ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1.2)
    ax.text(0.62, 0.66, "perfect calibration", fontsize=8, color=REFERENCE, rotation=39)
    ax.plot(predicted, observed, color=ACCENT, marker="o", linewidth=1.8, markersize=5)

    style_axes(
        ax,
        title=f"{name} probabilities are not calibrated:\nclass weighting inflates them by roughly an order of magnitude",
        xlabel="Mean predicted probability of response",
        ylabel="Observed response rate",
        frame="validation carve-out (n=11,450), 10 quantile bins",
    )
    # Both axes on the same 0-1 scale, so the distance from the diagonal is readable as
    # a distance rather than as an artefact of two different zooms.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, "calibration_best_model.png")


# ---------------------------------------------------------------------------
# Fairness slices (validation carve-out)
# ---------------------------------------------------------------------------
# A proposal commitment, and 50 rubric points. Measurement only in this phase: the
# plan puts mitigation out of scope and says to record anything stark rather than act
# on it. The framing that matters is in CLAUDE.md: these scores inform outreach
# efficiency only and must never gate any family's access to services. A contact-rate
# gap means one group gets less mail. It must never come to mean one group gets less
# help.

FAIRNESS_MIN_SEGMENT = 100


MISSING_SEGMENT = "(missing)"


def age_band(values: pd.Series) -> pd.Series:
    """Decade band labels, matching the EDA's bands so the two documents line up.

    Returns plain strings like "70-79", not Interval objects: an Interval renders as
    "[70.0, 80.0)", which is unreadable as a tick label and wrong as a description of
    an integer age.

    Rows with no band become MISSING_SEGMENT rather than dropping out. AGE is 24.8%
    missing (findings task 3) and CLAUDE.md's rule is to handle that explicitly and
    never silently drop it. A fairness table that quietly omits a quarter of the list
    is exactly the kind of silence that rule exists to prevent, and a missing AGE is
    itself a group the campaign either mails or does not.
    """
    bands = pd.cut(values, bins=AGE_BAND_EDGES, right=False)
    return pd.Series(
        [
            MISSING_SEGMENT if pd.isna(b) else f"{int(b.left)}-{int(b.right) - 1}"
            for b in bands
        ],
        index=values.index,
        dtype=object,
    )


def fairness_slices(frame: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Contact rate and recall by demographic segment. Frame: validation carve-out.

    Contact rate is the share of a segment the campaign would mail at the chosen
    threshold. Recall is the share of that segment's actual responders it would reach.
    The two answer different questions and both belong here: a segment can be mailed
    rarely and still have its responders found, or mailed heavily and still have them
    missed.

    Segments below FAIRNESS_MIN_SEGMENT rows are reported with their size but should
    not be read as evidence; on a 5%-positive target a 40-row segment has about two
    responders and its recall moves in 50-point steps.
    """
    y = frame["TARGET_B"].to_numpy()
    contacted = proba >= threshold

    segments = {
        "GENDER": frame["GENDER"]
        .str.strip()
        .replace("", None)
        .fillna(MISSING_SEGMENT)
        .astype(object),
        "AGE band": age_band(frame["AGE"]),
        "INCOME": frame["INCOME"].map(
            lambda v: MISSING_SEGMENT if pd.isna(v) else f"{v:g}"
        ),
    }

    rows = []
    for name, segment in segments.items():
        # Every row of the frame lands in exactly one segment of every attribute. This
        # is the mechanical form of CLAUDE.md's rule against silently dropping the
        # missing: if a level ever fails to account for its rows, the table is a
        # fairness claim about a population that is not the one being mailed.
        if segment.isna().any():
            raise ValueError(
                f"{name} left {int(segment.isna().sum())} rows unassigned. Every row "
                f"needs a segment, including the missing ones."
            )

        for level in sorted(segment.unique(), key=str):
            mask = (segment == level).to_numpy()
            responders = int(y[mask].sum())
            rows.append(
                {
                    "attribute": name,
                    "segment": str(level),
                    "n": int(mask.sum()),
                    "responders": responders,
                    "contact_rate": float(contacted[mask].mean()),
                    "recall": float(y[mask & contacted].sum() / responders) if responders else np.nan,
                    "response_rate": float(y[mask].mean()),
                }
            )

    table = pd.DataFrame(rows)
    for name, group in table.groupby("attribute"):
        if int(group["n"].sum()) != len(frame):
            raise ValueError(
                f"{name} segments cover {int(group['n'].sum()):,} rows but the frame "
                f"has {len(frame):,}. Segments must partition the frame."
            )
    return table


def report_fairness(slices: pd.DataFrame) -> None:
    """Print the slices and the gap between the extremes of each attribute."""
    print("=" * 78)
    print("FAIRNESS SLICES  Frame: validation carve-out (n=11,450), at the chosen threshold.")
    print("=" * 78)
    print("\nMeasurement only, per the plan. Scores inform outreach efficiency and must")
    print("never gate access to services (CLAUDE.md).\n")

    for attribute, group in slices.groupby("attribute", sort=False):
        print(f"{attribute}. Frame: validation carve-out.")
        print(f"  {'segment':<12} {'n':>6} {'responders':>11} {'response':>9} {'contacted':>10} {'recall':>8}")
        for row in group.itertuples():
            small = " *" if row.n < FAIRNESS_MIN_SEGMENT else ""
            recall = f"{row.recall * 100:>7.1f}%" if not np.isnan(row.recall) else f"{'-':>8}"
            print(
                f"  {row.segment:<12} {row.n:>6,} {row.responders:>11,} "
                f"{row.response_rate * 100:>8.2f}% {row.contact_rate * 100:>9.1f}% {recall}{small}"
            )
        big = group[group["n"] >= FAIRNESS_MIN_SEGMENT]
        if len(big) > 1:
            spread = big["contact_rate"].max() - big["contact_rate"].min()
            print(
                f"  Contact-rate gap across segments with n>={FAIRNESS_MIN_SEGMENT}: "
                f"{spread * 100:.1f} pp "
                f"({big.loc[big['contact_rate'].idxmax(), 'segment']} highest, "
                f"{big.loc[big['contact_rate'].idxmin(), 'segment']} lowest)"
            )
        print(f"  * segment smaller than {FAIRNESS_MIN_SEGMENT} rows, not evidence\n")


def figure_fairness(slices: pd.DataFrame, threshold: float) -> None:
    """Contact rate and recall per segment, one panel per attribute. Frame: validation."""
    attributes = list(dict.fromkeys(slices["attribute"]))
    # Panels sized by how many segments each attribute has, so the eight INCOME bars
    # are not squeezed to the width of the five GENDER ones and the tick labels have
    # room to sit flat instead of overlapping.
    widths = [int((slices["attribute"] == a).sum()) for a in attributes]
    fig, axes = plt.subplots(
        1, len(attributes), figsize=(16, 5), width_ratios=widths
    )

    for ax, attribute in zip(axes, attributes):
        group = slices[slices["attribute"] == attribute]
        positions = np.arange(len(group))
        width = 0.38

        ax.bar(
            positions - width / 2,
            group["contact_rate"] * 100,
            width=width,
            color=BASE,
            label="contacted",
        )
        ax.bar(
            positions + width / 2,
            group["recall"] * 100,
            width=width,
            color=ACCENT,
            label="recall (responders reached)",
        )

        # Small segments get their n called out on the bar rather than in a caption
        # nobody reads: a 40-row segment's recall is not a finding and the figure
        # should say so where the bar is.
        labels = []
        for row in group.itertuples():
            mark = "\n(n<100)" if row.n < FAIRNESS_MIN_SEGMENT else ""
            labels.append(f"{row.segment}{mark}")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(attribute, fontsize=11, pad=6)
        ax.set_ylabel("% of segment" if attribute == attributes[0] else "", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)

    # One legend for the figure, not one per panel. Every segment here is contacted 60%
    # of the time or more, so the bars are tall and an in-panel legend sits on top of
    # them; the strip beside the title is the only reliably empty space.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.99),
        ncol=2,
        fontsize=9,
    )

    fig.suptitle(
        "Who the campaign would mail, and whose responders it would find, at the chosen "
        f"threshold ({threshold:.3f})\n"
        "Frame: validation carve-out (n=11,450). Measurement only: these scores inform "
        "outreach efficiency and never access to services.",
        fontsize=11.5,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, "fairness_slices.png")


# ---------------------------------------------------------------------------
# The census experiment (training portion, same CV)
# ---------------------------------------------------------------------------


def census_experiment(spec: ModelSpec, params: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    """The best model's pipeline with the 290 census columns added. Frame: train_full.

    Exactly one run, as the plan specifies: the same model at the same tuned
    hyperparameters, scored by the same CV object on the same rows, with the only
    difference being the census block. Re-tuning here would confound the question,
    because a difference in score could then be the neighbourhood data or could be the
    new hyperparameters.

    This is a headline result for the ethics section rather than a footnote. The
    baseline excludes the census because neighbourhood-derived prediction is where
    proxy discrimination concentrates (plan, feature policy), and the dataset is 60.3%
    census columns against 2.7% individual demographics (findings task 1). What the
    report needs is the price of that exclusion in predictive terms, stated honestly in
    both directions: what accuracy the exclusion costs, and what it buys.
    """
    pipeline = build_pipeline(
        spec.estimator,
        scale_numeric=spec.scale_numeric,
        include_census=True,
        columns=list(X.columns),
    )
    pipeline.set_params(**params)

    started = time.perf_counter()
    scores = cross_val_score(pipeline, X, y, scoring=CV_SCORING, cv=CV, n_jobs=-1)
    elapsed = time.perf_counter() - started

    return {
        "cv_auprc_mean": float(scores.mean()),
        "cv_auprc_std": float(scores.std()),
        "cv_fit_seconds": round(elapsed, 1),
    }


def report_census(name: str, baseline: dict, with_census: dict) -> None:
    """Print the census delta in both directions. Frame: train_full, same 5-fold CV."""
    delta = with_census["cv_auprc_mean"] - baseline["cv_auprc_mean"]
    print("=" * 78)
    print(f"CENSUS EXPERIMENT  Frame: training portion (n=76,329), same 5-fold CV. Model: {name}.")
    print("=" * 78)
    print(f"\n  {'feature set':<28} {'CV AUPRC':>10} {'std':>8} {'vs floor':>10} {'secs':>7}")
    for label, row in (
        ("baseline (no census)", baseline),
        ("baseline + census (290 cols)", with_census),
    ):
        print(
            f"  {label:<28} {row['cv_auprc_mean']:>10.4f} {row['cv_auprc_std']:>8.4f} "
            f"{row['cv_auprc_mean'] / AUPRC_FLOOR:>9.2f}x {row['cv_fit_seconds']:>7.1f}"
        )
    print(
        f"\n  Delta from adding the census block: {delta:+.4f} AUPRC "
        f"({delta / baseline['cv_auprc_mean'] * 100:+.1f}%)"
    )
    print(
        f"  Fold-to-fold SD of the baseline is {baseline['cv_auprc_std']:.4f}, so a delta\n"
        f"  smaller than that is not a difference this experiment can resolve.\n"
    )


# ---------------------------------------------------------------------------
# The test split. Read once, here, and nowhere else.
# ---------------------------------------------------------------------------


def evaluate_on_test(
    spec: ModelSpec,
    params: dict,
    threshold: float,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    test: pd.DataFrame,
) -> dict:
    """Evaluate the frozen configuration on the held-out test split. Once.

    This is the only function in the module that reads the test frame, and main calls
    it last, after the model family, its hyperparameters and the operating threshold
    are all fixed. Nothing it measures can feed back into any of those choices, which
    is what makes the number it returns worth reporting.

    The model is refitted on the full training portion here rather than reused from the
    validation stage: the threshold was chosen with a model fitted on train_inner, but
    the campaign would ship a model fitted on everything it has. That mismatch is
    deliberate and worth stating in the findings, because the refitted model's
    probability scale can shift slightly under a threshold chosen from the other one.

    Args:
        spec: The winning model family.
        params: Its CV-chosen hyperparameters.
        threshold: The operating threshold chosen on validation. Frozen.
        X_train_full: Features for the whole training portion.
        y_train_full: TARGET_B for the whole training portion.
        test: The raw held-out frame.

    Returns:
        Test metrics: AUPRC, and the contact set at the frozen threshold.
    """
    X_test, y_test, amounts_test = make_xy(test)
    estimator = fit_spec(spec, params, X_train_full, y_train_full)
    proba = estimator.predict_proba(X_test)[:, 1]

    y_array = y_test.to_numpy()
    amounts_array = amounts_test.to_numpy()

    result = metrics_at_threshold(y_array, amounts_array, proba, threshold)
    result["auprc"] = float(average_precision_score(y_array, proba))
    result["everyone"] = net_revenue(
        y_array, amounts_array, np.ones(len(y_array), dtype=bool)
    )
    result["n"] = len(y_array)
    result["positive_rate"] = float(y_array.mean())

    print("=" * 78)
    print(f"TEST EVALUATION  Frame: held-out test split (n={len(y_array):,}). Model: {spec.name}.")
    print("=" * 78)
    print("\n  Read once, after the model, its parameters and the threshold were frozen.")
    print(f"  Fitted on the full training portion (n={len(y_train_full):,}).\n")
    print(f"  AUPRC                        {result['auprc']:.4f}")
    print(f"  AUPRC floor (positive rate)  {result['positive_rate']:.4f}")
    print(f"  Lift over floor              {result['auprc'] / result['positive_rate']:.2f}x")
    print(f"\n  At the frozen threshold {threshold:.4f}:")
    print(f"    contacts       {result['contacts']:,} of {result['n']:,} ({result['contact_fraction'] * 100:.1f}%)")
    print(f"    precision      {result['precision'] * 100:.2f}%")
    print(f"    recall         {result['recall'] * 100:.2f}%")
    print(f"    net revenue    ${result['net_revenue']:,.2f}")
    print(f"    contact all    ${result['everyone']:,.2f}")
    print(
        f"    gain           ${result['net_revenue'] - result['everyone']:,.2f} "
        f"({(result['net_revenue'] / result['everyone'] - 1) * 100:+.1f}% over mailing everyone)"
    )
    print()
    return result


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

    # Threshold, revenue, calibration and fairness all need models that never saw the
    # validation rows, so they are refitted on train_inner with the CV-chosen params
    # rather than reusing the artifacts above (which were refitted on train_full, and
    # so have seen every validation row).
    X_inner, y_inner, _amounts_inner = make_xy(train_inner)
    X_val, y_val, amounts_val = make_xy(validation)
    y_val_array = y_val.to_numpy()
    amounts_val_array = amounts_val.to_numpy()

    print("Refitting on train_inner for the validation-frame analyses ...", flush=True)
    curves = {}
    probabilities = {}
    for row, spec in zip(rows, model_specs()):
        estimator = fit_spec(spec, json.loads(row["params"]), X_inner, y_inner)
        proba = estimator.predict_proba(X_val)[:, 1]
        probabilities[spec.name] = proba
        curves[spec.name] = revenue_curve(y_val_array, amounts_val_array, proba)
    print()

    everyone = net_revenue(
        y_val_array, amounts_val_array, np.ones(len(y_val_array), dtype=bool)
    )
    figure_net_revenue(curves, everyone)

    # The best model by CV AUPRC, chosen on the training portion. Everything downstream
    # is about this model, and the choice is made here, before anything sees test.
    best_name = max(rows, key=lambda r: r["cv_auprc_mean"])["model"]
    best_proba = probabilities[best_name]

    chosen = choose_threshold(y_val_array, amounts_val_array, best_proba)
    naive = metrics_at_threshold(y_val_array, amounts_val_array, best_proba, 0.5)
    dummy_peak = float(curves["dummy"]["net_revenue"].max())
    report_thresholds(
        best_name, chosen, naive, everyone, dummy_peak, len(y_val_array)
    )

    figure_calibration(best_name, y_val_array, best_proba)

    slices = fairness_slices(validation, best_proba, chosen["threshold"])
    report_fairness(slices)
    figure_fairness(slices, chosen["threshold"])
    slices.to_csv(MODELING_DIR / "fairness_slices.csv", index=False)

    for row in rows:
        peak = curves[row["model"]]["net_revenue"].max()
        row["val_net_revenue_peak"] = round(float(peak), 2)
    best_row = next(r for r in rows if r["model"] == best_name)
    best_row.update(
        {
            "val_threshold": round(chosen["threshold"], 6),
            "val_contacts": chosen["contacts"],
            "val_contact_fraction": round(chosen["contact_fraction"], 4),
            "val_precision": round(chosen["precision"], 4),
            "val_recall": round(chosen["recall"], 4),
            "val_net_revenue": round(chosen["net_revenue"], 2),
        }
    )

    # The census experiment, on the training portion. Still no test rows.
    best_spec = next(s for s in model_specs() if s.name == best_name)
    best_params = json.loads(best_row["params"])
    with_census = census_experiment(best_spec, best_params, X_train_full, y_train_full)
    report_census(best_name, best_row, with_census)
    rows.append(
        {
            "model": f"{best_name}_with_census",
            "params": best_row["params"],
            "cv_auprc_mean": with_census["cv_auprc_mean"],
            "cv_auprc_std": with_census["cv_auprc_std"],
            "cv_fit_seconds": with_census["cv_fit_seconds"],
            "grid_size": 1,
            "note": (
                "census experiment: baseline pipeline plus the 290 census columns, same "
                "params, same CV. Not a candidate for deployment."
            ),
            "cv_results": pd.DataFrame(
                {
                    "model": [f"{best_name}_with_census"],
                    "params": [best_row["params"]],
                    "cv_auprc_mean": [with_census["cv_auprc_mean"]],
                    "cv_auprc_std": [with_census["cv_auprc_std"]],
                    "rank": [1],
                }
            ),
        }
    )

    # Everything above this line is frozen: the model family, its hyperparameters, and
    # the operating threshold. The test split has not been read.
    test_result = evaluate_on_test(
        best_spec, best_params, chosen["threshold"], X_train_full, y_train_full, test
    )
    best_row.update(
        {
            "test_auprc": round(test_result["auprc"], 4),
            "test_threshold": round(test_result["threshold"], 6),
            "test_contacts": test_result["contacts"],
            "test_contact_fraction": round(test_result["contact_fraction"], 4),
            "test_precision": round(test_result["precision"], 4),
            "test_recall": round(test_result["recall"], 4),
            "test_net_revenue": round(test_result["net_revenue"], 2),
            "test_net_revenue_contact_all": round(test_result["everyone"], 2),
        }
    )

    write_results(rows)


if __name__ == "__main__":
    main()
