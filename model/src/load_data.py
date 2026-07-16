"""Load the KDD Cup 1998 learning set and split it per the course conventions.

The dataset is a direct-mail campaign from a national veterans nonprofit, used here as
a public proxy for constituent engagement. It is not Sturge-Weber Foundation data.

Citation: Parsa, I. (1998). KDD Cup 1998 [Data set]. https://doi.org/10.24432/C5401H

Run ``bash model/download_data.sh`` before importing anything here. That script writes
``data/cup98lrn.txt`` and verifies its size.

Split conventions come from CLAUDE.md and are course-wide. Do not change the constants
in this module without changing them there too, since every reported metric depends on
the split being identical across runs and across team members.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Repo root is two levels up from this file: model/src/load_data.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = REPO_ROOT / "data" / "cup98lrn.txt"

# Verified against the official documentation on 2026-07-16.
EXPECTED_SHAPE = (95412, 481)

TARGET_BINARY = "TARGET_B"
TARGET_AMOUNT = "TARGET_D"

# Course-wide split convention (CLAUDE.md). Not tunable.
RANDOM_STATE = 2026
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Columns whose inferred type depends on which chunk pandas happens to look at, so
# they must be declared. NOEXCH holds '0' and '1' for almost every row, but 35 rows
# hold 'X' and 7 hold a single space, which makes it a categorical column, not a
# numeric one. Declaring it as string keeps those 42 rows readable instead of coercing
# them to NaN. Any other mixed column is a bug: load_raw turns DtypeWarning into an
# error rather than letting inference quietly pick a type.
DTYPE_OVERRIDES: dict[str, str] = {"NOEXCH": "str"}


def load_raw(path: Path | str | None = None, validate: bool = True) -> pd.DataFrame:
    """Read the learning set into a DataFrame with types declared, not guessed.

    Expected shape: (95412, 481). Every column of the raw file is preserved, including
    both targets, so callers can drop what they do not need rather than discovering a
    column was silently missing.

    Missing values are left as NaN and are not imputed or dropped here. The two columns
    that matter most for that are AGE (23,665 missing, 24.8%) and INCOME (21,286
    missing, 22.3%). Handle them explicitly downstream.

    Args:
        path: Location of cup98lrn.txt. Defaults to ``data/cup98lrn.txt`` in the repo.
        validate: If True, raise when the shape is not (95412, 481) or a target column
            is absent. Turn this off only to inspect a deliberately truncated file.

    Returns:
        DataFrame of shape (95412, 481). TARGET_B is int64 (0/1), TARGET_D is float64
        (0.0 for non-responders), NOEXCH is a string column.

    Raises:
        FileNotFoundError: If the file is absent, with the download command to fix it.
        ValueError: If validate is True and the shape or target columns are wrong.
        pandas.errors.DtypeWarning: Escalated to an error if a column outside
            DTYPE_OVERRIDES has mixed types.
    """
    data_path = Path(path) if path is not None else RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run: bash model/download_data.sh"
        )

    # An undeclared mixed-type column would otherwise be a warning printed into the
    # void, and the resulting silent coercion is exactly the kind of data bug that is
    # invisible until it shows up in a metric. Fail instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.DtypeWarning)
        df = pd.read_csv(data_path, dtype=DTYPE_OVERRIDES)

    if validate:
        _validate_raw(df, data_path)

    return df


def _validate_raw(df: pd.DataFrame, data_path: Path) -> None:
    """Check the loaded frame against the documented facts. Raises ValueError."""
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{data_path} loaded as {df.shape}, expected {EXPECTED_SHAPE}. "
            "The file is truncated or the wrong dataset. Re-run model/download_data.sh."
        )

    missing = [c for c in (TARGET_BINARY, TARGET_AMOUNT) if c not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) absent from {data_path}: {missing}")


def make_split(
    df: pd.DataFrame,
    validation: bool = False,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, ...]:
    """Split into train/test, optionally carving a validation set out of train.

    Implements the CLAUDE.md convention: 80/20 stratified on TARGET_B with
    random_state=2026, and where a validation set is needed, 15% of the *training*
    portion (not of the whole dataset). The test set is identical whether or not
    validation is requested, so results stay comparable across both call styles.

    Stratifying matters here: TARGET_B is 5.08% positive (4,843 of 95,412), so an
    unstratified split would leave the class rate drifting between runs.

    Expected shapes, given the full (95412, 481) frame:

        validation=False -> train (76329, 481), test (19083, 481)
        validation=True  -> train (64879, 481), val (11450, 481), test (19083, 481)

    Each part holds ~5.08% positives. Rows are not modified, only partitioned, so the
    parts always sum back to len(df) and the original index is preserved.

    Args:
        df: Frame to split. Must contain TARGET_B.
        validation: If True, return a validation set carved from the training portion.
        random_state: Seed. Defaults to the course-wide 2026. Override only for
            experiments about split sensitivity, never for reported metrics.

    Returns:
        (train, test) if validation is False, else (train, val, test).

    Raises:
        ValueError: If TARGET_B is absent or has fewer than 2 classes to stratify on.
    """
    if TARGET_BINARY not in df.columns:
        raise ValueError(
            f"{TARGET_BINARY} is required to stratify the split but is not in the frame."
        )

    if df[TARGET_BINARY].nunique() < 2:
        raise ValueError(
            f"{TARGET_BINARY} has a single class; a stratified split is not defined."
        )

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_BINARY],
        random_state=random_state,
    )

    if not validation:
        return train, test

    train, val = train_test_split(
        train,
        test_size=VALIDATION_SIZE,
        stratify=train[TARGET_BINARY],
        random_state=random_state,
    )

    return train, val, test
