"""Feature policy and preprocessing for the TARGET_B classifier.

Implements the "Feature policy" and "Preprocessing pipeline" sections of
docs/modeling_plan.md. That plan is the contract: the column lists below are the
policy written as code, and the reason each group is in or out is recorded in the
plan, not re-argued here. Where a constant traces back to an EDA number, the comment
cites the task in docs/eda_findings.md that produced it.

The module has three layers, and the split between them is the leakage guarantee:

1. ``make_xy``: pulls y (TARGET_B) and the amount vector (TARGET_D) out of the raw
   frame and hands back an X that cannot contain CONTROLN, TARGET_B, or TARGET_D.
   TARGET_D is returned as its own object because it is an evaluation input for the
   net-revenue lens and never a feature.
2. ``FeatureFrameBuilder``: stateless row-wise work (column selection, blank
   normalization, the AGE floor, YYMM date conversion, derived counts). It learns
   nothing from the data it sees, so it cannot leak regardless of what it is fit on.
3. ``build_preprocessor``: everything that learns a statistic (imputation medians,
   one-hot categories, STATE frequencies, scaler means). All of it lives inside a
   ColumnTransformer that only ever sees training rows, because it is a step of the
   same Pipeline as the estimator.

All data access goes through load_data. This module never reads the CSV itself.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from eda import CENSUS_OUTSIDE_BLOCK, resolve_census_block
from load_data import TARGET_AMOUNT, TARGET_BINARY

# ---------------------------------------------------------------------------
# Columns that may never appear in a feature matrix
# ---------------------------------------------------------------------------
# CONTROLN is the row ID: it carries no signal and would let a model memorize rows.
# TARGET_B is y. TARGET_D is the label of the amount the constituent gave, so it is
# the answer to the question in a different costume; it enters the net-revenue
# evaluation and nothing else. This tuple exists so the exclusion is one named fact
# that a test can assert against rather than three separate drops scattered around.
ROW_ID = "CONTROLN"
FORBIDDEN_IN_X = (ROW_ID, TARGET_BINARY, TARGET_AMOUNT)

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
# The campaign being predicted is the 97NK mailing. Its promotion date is ADATE_2,
# which is 9706 for 95,399 of 95,412 rows, so the whole file is a snapshot taken at
# June 1997 and every date in it is a duration before that point.
CAMPAIGN_YYMM = 9706

# Every date in the file falls between 1975 and 1997 (findings task 6: the widest
# range is MINRDATE at 7506 to 9702). So the two-digit year is always 19xx and the
# month arithmetic below never has to straddle a century.
MIN_PLAUSIBLE_YY = 0
MAX_PLAUSIBLE_YY = 99

# ---------------------------------------------------------------------------
# Feature policy: demographics (13 of 13, per the plan)
# ---------------------------------------------------------------------------
# AGE and INCOME carry the dataset's headline missingness (24.8% and 22.3%). WEALTH1,
# WEALTH2 and NUMCHLD are worse (46.9%, 45.9%, 87.0%) but the plan says all 13
# demographics are in, and median imputation is the plan's answer for numerics.
# INCOME is a 7-level ordinal bracket already coded 1..7 (findings task 3), so it is
# already ordinally encoded and passes through the numeric path. Same for WEALTH1/2.
DEMOGRAPHIC_NUMERIC = ("AGE", "INCOME", "WEALTH1", "WEALTH2", "NUMCHLD")

DEMOGRAPHIC_CATEGORICAL = (
    "AGEFLAG",
    "GENDER",
    "HOMEOWNR",
    "CHILD03",
    "CHILD07",
    "CHILD12",
    "CHILD18",
)

# DOB is the thirteenth demographic. It is a YYMM integer, so the plan's rule against
# feeding raw YYMM values to a model applies to it exactly as it does to LASTDATE: the
# gap from 9612 to 9701 is one month but reads as 89. It is therefore converted to a
# duration rather than dropped or passed through raw. DOB=0 is the vendor's "unknown"
# and is nulled first (findings task 3: 23,661 rows, and they are exactly the rows
# where AGE is missing).
DEMOGRAPHIC_DATE = ("DOB",)

# ---------------------------------------------------------------------------
# Feature policy: giving history (summaries only)
# ---------------------------------------------------------------------------
# The per-mailing RDATE_*/RAMNT_* pairs are out as raw features: their missingness is
# structural, and these summaries already aggregate the same history (plan, feature
# policy). Multicollinearity here runs up to r=0.91 (findings task 5) and is tolerated
# per the plan.
GIVING_NUMERIC = (
    "NGIFTALL",
    "CARDGIFT",
    "RAMNTALL",
    "LASTGIFT",
    "AVGGIFT",
    "MINRAMNT",
    "MAXRAMNT",
)

# Recency. The plan asks for "recency in days derived from LASTDATE"; the field is
# YYMM, so days are not recoverable and the unit is months. Stated in findings.
GIVING_DATE = ("LASTDATE",)

# ---------------------------------------------------------------------------
# Feature policy: promotion history
# ---------------------------------------------------------------------------
# Counts of mailings received, which is the plan's "number of promotions received".
PROMOTION_NUMERIC = ("NUMPROM", "CARDPROM", "NUMPRM12", "CARDPM12")

# RFA_2F is gift frequency banded 1..4, already ordered, so it is already an ordinal
# encoding and needs no transform. The strongest signal in the EDA after RFA_2A
# (findings task 5: 3.79% response at band 1 to 8.37% at band 4).
PROMOTION_ORDINAL_NUMERIC = ("RFA_2F",)

# RFA_2A is gift amount banded D<E<F<G, smallest to largest. Encoded ordinally with
# that order stated explicitly rather than left to alphabetical luck. Response runs
# 9.41% at D down to 3.42% at G (findings task 5), monotonic, so an ordinal encoding
# is a fair representation and not a convenience.
RFA_2A_CATEGORIES = ["D", "E", "F", "G"]
PROMOTION_ORDINAL_CATEGORICAL = ("RFA_2A",)

# RFA_2R is constant ('L' for all 95,412 rows, findings task 1 correction 5) and the
# raw RFA_3..RFA_24 history codes are out (plan, feature policy).

# ---------------------------------------------------------------------------
# Feature policy: geography
# ---------------------------------------------------------------------------
# STATE only, frequency-encoded, fit on train. ZIP (19,938 levels) and the area codes
# are out (plan, feature policy; findings task 6).
GEOGRAPHY_FREQUENCY = ("STATE",)

# ---------------------------------------------------------------------------
# Derived feature names
# ---------------------------------------------------------------------------
MONTHS_SINCE_LASTGIFT = "MONTHS_SINCE_LASTGIFT"
MONTHS_SINCE_DOB = "MONTHS_SINCE_DOB"
N_PROMO_RESPONSES = "N_PROMO_RESPONSES"

DERIVED_NUMERIC = (MONTHS_SINCE_LASTGIFT, MONTHS_SINCE_DOB, N_PROMO_RESPONSES)

# The plan's "number responded to if derivable" for promotion history. It is
# derivable: RAMNT_n is non-null exactly when the constituent gave to mailing n.
#
# The index is what makes this safe, and it is worth being explicit about. Promotion 2
# IS the campaign being predicted (ADATE_2 = 9706 = the 97NK mailing), and the file
# ships no RDATE_2 or RAMNT_2 column precisely because the response to promotion 2 is
# TARGET_B. The RAMNT_* family therefore starts at 3 and is entirely historical. A
# RAMNT_2 appearing in a future version of this file would be the target leaking in,
# so _promo_response_columns raises rather than counting it.
PROMO_RESPONSE_PATTERN = re.compile(r"RAMNT_(\d+)$")
CAMPAIGN_PROMOTION_INDEX = 2

# ---------------------------------------------------------------------------
# The AGE floor
# ---------------------------------------------------------------------------
# Findings task 4: 15 rows in the training split have AGE <= 5, and 9 are exactly 1. A
# one-year-old with a decade of giving history is bad data. The plan's decision is to
# null the value rather than drop the row: a 5%-positive target cannot spare rows, and
# the other 480 fields on those rows are fine.
AGE_FLOOR = 5


def yymm_to_months_before(values: pd.Series, reference_yymm: int = CAMPAIGN_YYMM) -> pd.Series:
    """Convert YYMM integers to whole months before a reference YYMM.

    A YYMM integer is not a number and must not be treated as one: 9701 minus 9612 is
    89 in integer arithmetic and one month in reality. This turns the field into a
    duration, which is the quantity a model can actually use (findings task 6).

    Values that are not a real YYMM become NaN rather than a wrong date. That covers
    the 0 sentinel both DOB (23,661 rows) and FISTDATE (2 rows) use for "unknown": a 0
    left alone would parse as a month zero of 1900 and hand the model a fabricated
    date, which is the specific failure the findings warn about.

    Every date in this file is between 1975 and 1997, so the two-digit year is always
    19xx and the arithmetic is century-free.

    Args:
        values: YYMM integers, e.g. 9702 for February 1997. May contain NaN.
        reference_yymm: The point to measure back from. Defaults to the campaign date.

    Returns:
        Float months before the reference. Positive means earlier than the reference.
        NaN where the input is missing or is not a valid YYMM.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    year = numeric // 100
    month = numeric % 100

    valid = (
        numeric.notna()
        & month.between(1, 12)
        & year.between(MIN_PLAUSIBLE_YY, MAX_PLAUSIBLE_YY)
    )

    reference_months = (reference_yymm // 100) * 12 + (reference_yymm % 100)
    months = reference_months - (year * 12 + month)
    return months.where(valid).astype("float64")


def normalize_blanks(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn whitespace-only strings into NaN, leaving everything else alone.

    65 of the 74 string columns carry blanks (findings task 1), and read_csv reads a
    space as data, so without this step a NaN-based imputer sees a fully populated
    column and a one-hot encoder invents a ' ' category. Non-string columns pass
    through untouched.

    Returns a new frame; the input is not modified.
    """
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(
            series
        ):
            stripped = series.str.strip()
            out[column] = stripped.where(stripped != "", other=None)
    return out


def census_columns(columns: list[str]) -> list[str]:
    """The 290 census columns, resolved against the real column order.

    Delegates to the EDA's group map rather than restating it, so the census
    experiment and the EDA report cannot drift apart about what "census" means.
    """
    block = resolve_census_block(columns)
    resolved = [c for c in columns if c in block or c in CENSUS_OUTSIDE_BLOCK]
    return resolved


def _promo_response_columns(columns: list[str]) -> list[str]:
    """RAMNT_* columns for historical promotions only.

    Raises:
        ValueError: If a RAMNT column for the campaign promotion itself is present.
            That column would be the target under another name (see the module notes
            on CAMPAIGN_PROMOTION_INDEX), and counting it would leak.
    """
    matched = []
    for column in columns:
        match = PROMO_RESPONSE_PATTERN.fullmatch(column)
        if match is None:
            continue
        if int(match.group(1)) <= CAMPAIGN_PROMOTION_INDEX:
            raise ValueError(
                f"{column!r} records the response to promotion "
                f"{match.group(1)}, but promotion {CAMPAIGN_PROMOTION_INDEX} is the "
                "campaign being predicted. Counting it would leak TARGET_B into X."
            )
        matched.append(column)
    return matched


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a raw frame into features, label, and the amount vector.

    The amount vector is returned separately, and deliberately not as a column of X,
    because TARGET_D is an evaluation input for the net-revenue lens and never a
    feature. Handing it back as its own object is what makes "TARGET_D is not in X" a
    property of the code rather than a habit of the caller.

    Args:
        df: A raw frame from load_raw, or any split of one.

    Returns:
        (X, y, amounts). X is every column except CONTROLN, TARGET_B and TARGET_D.
        y is TARGET_B as int. amounts is TARGET_D, aligned to X by index.

    Raises:
        ValueError: If a target column is absent.
    """
    missing = [c for c in (TARGET_BINARY, TARGET_AMOUNT) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) absent from the frame: {missing}")

    X = df.drop(columns=[c for c in FORBIDDEN_IN_X if c in df.columns])
    y = df[TARGET_BINARY].astype(int)
    amounts = df[TARGET_AMOUNT].astype(float)
    return X, y, amounts


class FeatureFrameBuilder(BaseEstimator, TransformerMixin):
    """Apply the feature policy: select columns, normalize, derive.

    Everything here is row-wise and stateless. No quantity computed from one row
    affects another, and nothing is remembered from fit, so this step cannot leak
    information from one split into another no matter what it is fit on. Every fitted
    statistic lives downstream in the ColumnTransformer.

    fit does one thing besides bookkeeping: it checks that the required raw columns
    are present, so a policy referring to a column the file does not have fails at fit
    with a name rather than at transform with a KeyError.

    Args:
        include_census: Add the 290 census columns to the numeric block. False for the
            baseline; True for the single census experiment the plan specifies. The
            plan's rationale for the default is in its feature policy section:
            neighborhood-derived prediction is where proxy discrimination risk
            concentrates, so the baseline establishes what individual-level features
            achieve on their own first.
    """

    def __init__(self, include_census: bool = False):
        self.include_census = include_census

    def _required_columns(self, columns: list[str]) -> list[str]:
        required = [
            *DEMOGRAPHIC_NUMERIC,
            *DEMOGRAPHIC_CATEGORICAL,
            *DEMOGRAPHIC_DATE,
            *GIVING_NUMERIC,
            *GIVING_DATE,
            *PROMOTION_NUMERIC,
            *PROMOTION_ORDINAL_NUMERIC,
            *PROMOTION_ORDINAL_CATEGORICAL,
            *GEOGRAPHY_FREQUENCY,
        ]
        if self.include_census:
            required.extend(census_columns(columns))
        return required

    def fit(self, X: pd.DataFrame, y=None):
        columns = list(X.columns)
        required = self._required_columns(columns)

        absent = [c for c in required if c not in columns]
        if absent:
            raise ValueError(
                f"The feature policy needs {len(absent)} column(s) the frame does not "
                f"have: {absent[:10]}"
            )

        forbidden = [c for c in FORBIDDEN_IN_X if c in columns]
        if forbidden:
            raise ValueError(
                f"{forbidden} must not reach a feature matrix. Build X with "
                "features.make_xy, which drops them."
            )

        # Resolved at fit so transform cannot silently count a different set of
        # promotion columns than the pipeline was fitted on.
        self.promo_response_columns_ = _promo_response_columns(columns)
        self.census_columns_ = census_columns(columns) if self.include_census else []
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        n_promo_responses = X[self.promo_response_columns_].notna().sum(axis=1)

        selected = [
            *DEMOGRAPHIC_NUMERIC,
            *DEMOGRAPHIC_CATEGORICAL,
            *GIVING_NUMERIC,
            *PROMOTION_NUMERIC,
            *PROMOTION_ORDINAL_NUMERIC,
            *PROMOTION_ORDINAL_CATEGORICAL,
            *GEOGRAPHY_FREQUENCY,
            *self.census_columns_,
        ]
        out = normalize_blanks(X[selected])

        # AGE <= 5 is bad data, not a young donor (findings task 4). Nulled here so the
        # downstream median imputer treats it as the unknown it is. Ordering matters:
        # this has to happen before imputation, and imputation is downstream, so it
        # cannot be done anywhere but here.
        out["AGE"] = out["AGE"].where(~(out["AGE"] <= AGE_FLOOR))

        out[MONTHS_SINCE_LASTGIFT] = yymm_to_months_before(X[GIVING_DATE[0]])
        out[MONTHS_SINCE_DOB] = yymm_to_months_before(X[DEMOGRAPHIC_DATE[0]])
        out[N_PROMO_RESPONSES] = n_promo_responses.astype("float64")

        # sklearn's imputers and encoders want numpy-friendly object arrays, not the
        # pandas string dtype, whose pd.NA does not compare the way they expect.
        for column in DEMOGRAPHIC_CATEGORICAL + PROMOTION_ORDINAL_CATEGORICAL + GEOGRAPHY_FREQUENCY:
            out[column] = out[column].astype(object).where(out[column].notna(), np.nan)

        return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it occurs in the training data.

    Used for STATE (57 levels), where one-hot is defensible but frequency encoding
    keeps the block to one column and avoids handing the model 57 mostly-empty
    indicators. The frequency table is learned at fit, so it is learned from the
    training rows only; that is the whole point of it living inside the Pipeline.

    Categories not seen during fit encode as 0.0, which is the honest value: a state
    that never appeared in training has a training frequency of zero. NaN encodes as
    0.0 for the same reason.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.frequencies_ = {
            column: frame[column].value_counts(normalize=True, dropna=True)
            for column in frame.columns
        }
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X) -> np.ndarray:
        frame = pd.DataFrame(X)
        encoded = {
            column: frame[column].map(self.frequencies_[column]).fillna(0.0).astype(float)
            for column in frame.columns
        }
        return pd.DataFrame(encoded, index=frame.index).to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{name}_freq" for name in names], dtype=object)


def numeric_feature_names(include_census: bool, columns: list[str] | None = None) -> list[str]:
    """The numeric block, in the order build_preprocessor assembles it."""
    names = [
        *DEMOGRAPHIC_NUMERIC,
        *GIVING_NUMERIC,
        *PROMOTION_NUMERIC,
        *PROMOTION_ORDINAL_NUMERIC,
        *DERIVED_NUMERIC,
    ]
    if include_census:
        if columns is None:
            raise ValueError("include_census=True needs the frame's columns to resolve.")
        names.extend(census_columns(columns))
    return names


def build_preprocessor(
    include_census: bool = False,
    scale_numeric: bool = False,
    columns: list[str] | None = None,
) -> ColumnTransformer:
    """Assemble the fitted half of the preprocessing: impute, encode, optionally scale.

    Every statistic this learns (medians, one-hot categories, STATE frequencies,
    scaler means and variances) is learned at fit time from whatever rows it is given.
    It is only safe because it is a step of a Pipeline whose fit sees the training
    rows and nothing else. Fitting this on a full frame would be the leak, and
    model/tests/test_features.py asserts that the pipeline's learned statistics do not
    move when the test rows change.

    Args:
        include_census: Add the 290 census columns to the numeric block.
        scale_numeric: Standardize the numeric, ordinal, and frequency blocks. True for
            the logistic model, which needs comparable scales for its penalty to mean
            anything. False for the trees, which are indifferent to monotone rescaling
            (plan, preprocessing step 5).
        columns: The raw frame's columns. Required when include_census is True.

    Returns:
        A ColumnTransformer expecting the output of FeatureFrameBuilder.
    """
    numeric = numeric_feature_names(include_census, columns)

    # Median rather than mean: every giving-history column is heavily right-skewed,
    # up to a skew of 27.5 for AVGGIFT (findings task 4), and a mean imputed into a
    # distribution like that lands where almost no real constituent sits.
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # Explicit "missing" category rather than an imputed code, per the plan: for these
    # columns the vendor not knowing is itself a fact about the row. No missingness
    # indicator flags for AGE/INCOME: the EDA tested whether missingness predicts
    # response and it does not (chi-square, findings task 3, p=0.16 and p=0.32). The
    # plan says cite that rather than re-decide it.
    #
    # min_frequency handles the size-1 category problem the EDA flagged: GENDER has a
    # 'C' and an 'A' appearing once each (findings task 4), and a one-hot column that
    # fires for a single training row is noise the model can memorize. Rare levels fold
    # into one "infrequent" column, and the threshold is applied to the training counts
    # because the encoder is fitted inside the Pipeline.
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    # RFA_2A's order is stated, not inferred. It happens to be alphabetical, but that
    # is luck, and relying on it would break silently if the codes were ever relabeled.
    ordinal_steps: list[tuple[str, object]] = [
        (
            "encode",
            OrdinalEncoder(
                categories=[RFA_2A_CATEGORIES],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            ),
        ),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        ordinal_steps.append(("scale", StandardScaler()))
    ordinal_pipeline = Pipeline(ordinal_steps)

    frequency_steps: list[tuple[str, object]] = [("encode", FrequencyEncoder())]
    if scale_numeric:
        frequency_steps.append(("scale", StandardScaler()))
    frequency_pipeline = Pipeline(frequency_steps)

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, list(DEMOGRAPHIC_CATEGORICAL)),
            ("ordinal", ordinal_pipeline, list(PROMOTION_ORDINAL_CATEGORICAL)),
            ("frequency", frequency_pipeline, list(GEOGRAPHY_FREQUENCY)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator,
    include_census: bool = False,
    scale_numeric: bool = False,
    columns: list[str] | None = None,
) -> Pipeline:
    """The full path from a raw frame to a fitted estimator.

    One Pipeline object end to end is the mechanical guarantee the plan asks for: the
    caller cannot fit a transformer on anything the estimator was not fitted on,
    because there is only one fit call and it takes one X.

    Args:
        estimator: Any sklearn classifier.
        include_census: Add the census block. Baseline is False.
        scale_numeric: Standardize numerics. True for logistic regression only.
        columns: The raw frame's columns. Required when include_census is True.

    Returns:
        Pipeline of (frame -> preprocess -> model), expecting X from make_xy.
    """
    return Pipeline(
        [
            ("frame", FeatureFrameBuilder(include_census=include_census)),
            (
                "preprocess",
                build_preprocessor(
                    include_census=include_census,
                    scale_numeric=scale_numeric,
                    columns=columns,
                ),
            ),
            ("model", estimator),
        ]
    )
