"""Shared fixtures for the model tests."""

import pytest

from load_data import RAW_DATA_PATH, load_raw


@pytest.fixture(scope="session")
def raw_df():
    """The full learning set, loaded once for the whole session.

    Expected shape (95412, 481). The file is ~117 MB, so reading it per test would
    dominate the runtime. Tests must treat this frame as read-only.

    Skips rather than fails when the dataset is absent: a missing download is a setup
    problem, not a defect in the code under test.
    """
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"{RAW_DATA_PATH} not found. Run: bash model/download_data.sh")
    return load_raw()
