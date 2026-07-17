"""Tests for the EDA script.

Scope is what docs/eda_plan.md asks for: the inventory covers all 481 columns, the
figure directory holds the expected files after a run, and the frame the EDA analyses
is the one make_split produces with the course defaults. Plus the pure helpers, which
are cheap to test and are where an arithmetic error would be invisible in a PNG.

Deliberately not tested: matplotlib output pixel-for-pixel (the plan rules it out, and
it would break on every matplotlib point release without indicating a real defect).

These tests read the committed artifacts rather than regenerating them. Regenerating
would take about 20 seconds per test and would write into docs/ as a side effect, which
a test run should not do. The clean-state regeneration is verified by running
`python model/src/eda.py` from an empty docs/figures and docs/eda, not from pytest.
"""

import pandas as pd
import pytest

from eda import (
    CENSUS_BLOCK_EXCLUSIONS,
    CENSUS_OUTSIDE_BLOCK,
    EXPECTED_FIGURES,
    FIGURES_DIR,
    INVENTORY_PATH,
    assign_group,
    blank_share,
    near_constant_columns,
    resolve_census_block,
    wilson_interval,
)
from load_data import make_split
from train import EXPECTED_FIGURES as MODELING_FIGURES

EXPECTED_COLUMN_COUNT = 481

# Documented in load_data.make_split and stamped on every training-split figure.
EXPECTED_TRAIN_ROWS = 76329
EXPECTED_TEST_ROWS = 19083


@pytest.fixture(scope="module")
def inventory():
    """The committed inventory. Skips if the EDA has not been run."""
    if not INVENTORY_PATH.exists():
        pytest.skip(f"{INVENTORY_PATH} not found. Run: python model/src/eda.py")
    return pd.read_csv(INVENTORY_PATH)


class TestInventory:
    def test_has_one_row_per_column(self, inventory):
        assert len(inventory) == EXPECTED_COLUMN_COUNT

    def test_column_names_are_unique(self, inventory):
        assert inventory["column"].is_unique

    def test_carries_every_field_the_plan_asks_for(self, inventory):
        expected = {
            "column",
            "dtype",
            "non_null",
            "pct_missing",
            "pct_blank",
            "nunique",
            "group",
        }
        assert set(inventory.columns) == expected

    def test_every_column_has_a_group(self, inventory):
        assert inventory["group"].notna().all()

    def test_group_counts_sum_to_the_column_count(self, inventory):
        assert inventory["group"].value_counts().sum() == EXPECTED_COLUMN_COUNT

    def test_census_group_is_290_columns(self, inventory):
        """286 in the POP901..AC2 run, minus 3 area codes, plus 7 outside it.

        This is the plan correction task 1 made. If the number moves, either the map
        changed or the file did, and both need a look.
        """
        assert int((inventory["group"] == "census").sum()) == 290

    def test_controln_is_the_row_id(self, inventory):
        """The plan's ground rule: CONTROLN appears in the inventory and nowhere else."""
        row = inventory.loc[inventory["column"] == "CONTROLN"].iloc[0]
        assert row["group"] == "id_admin"
        assert row["nunique"] == EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS

    def test_non_null_and_pct_missing_agree(self, inventory):
        total = EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS
        implied = (1 - inventory["non_null"] / total) * 100
        assert (implied - inventory["pct_missing"]).abs().max() < 0.01


class TestGroupMap:
    def test_every_real_column_is_assigned_exactly_one_group(self, raw_df):
        """The map must cover the frame with no column falling through."""
        census_block = resolve_census_block(list(raw_df.columns))
        groups = [assign_group(c, census_block) for c in raw_df.columns]
        assert len(groups) == EXPECTED_COLUMN_COUNT
        assert all(isinstance(g, str) and g for g in groups)

    def test_unknown_column_raises_rather_than_bucketing_as_other(self):
        with pytest.raises(ValueError, match="matches no group"):
            assign_group("NOT_A_REAL_COLUMN", frozenset())

    def test_rfa_2f_and_2a_are_promotion_not_swept_up_by_the_rfa_prefix(self):
        assert assign_group("RFA_2F", frozenset()) == "promotion_history"
        assert assign_group("RFA_2A", frozenset()) == "promotion_history"
        assert assign_group("RFA_7", frozenset()) == "promotion_history"

    def test_targets_are_their_own_group(self):
        assert assign_group("TARGET_B", frozenset()) == "target"
        assert assign_group("TARGET_D", frozenset()) == "target"


class TestCensusBlock:
    def test_run_spans_pop901_to_ac2(self, raw_df):
        columns = list(raw_df.columns)
        block = resolve_census_block(columns)
        assert "POP901" in block
        assert "AC2" in block
        # 286 columns in the contiguous run, less the three area codes.
        assert len(block) == 283

    def test_area_codes_are_excluded_from_the_block(self, raw_df):
        block = resolve_census_block(list(raw_df.columns))
        for column in CENSUS_BLOCK_EXCLUSIONS:
            assert column not in block

    def test_area_codes_are_grouped_as_geography_instead(self, raw_df):
        block = resolve_census_block(list(raw_df.columns))
        for column in CENSUS_BLOCK_EXCLUSIONS:
            assert assign_group(column, block) == "geography"

    def test_percentages_outside_the_run_are_still_census(self, raw_df):
        """MALEMILI..FEDGOV sit before the run. The plan's map missed them."""
        block = resolve_census_block(list(raw_df.columns))
        for column in CENSUS_OUTSIDE_BLOCK:
            assert column not in block
            assert assign_group(column, block) == "census"

    def test_missing_endpoint_raises(self):
        with pytest.raises(ValueError, match="not a column"):
            resolve_census_block(["A", "B"])

    def test_inverted_run_raises(self):
        with pytest.raises(ValueError, match="out of order"):
            resolve_census_block(["AC2", "POP901"])


class TestFigures:
    def test_directory_exists(self):
        if not FIGURES_DIR.exists():
            pytest.skip(f"{FIGURES_DIR} not found. Run: python model/src/eda.py")

    @pytest.mark.parametrize("name", EXPECTED_FIGURES)
    def test_expected_figure_was_produced(self, name):
        if not FIGURES_DIR.exists():
            pytest.skip(f"{FIGURES_DIR} not found. Run: python model/src/eda.py")
        path = FIGURES_DIR / name
        assert path.exists(), f"{name} missing. Run: python model/src/eda.py"
        assert path.stat().st_size > 0, f"{name} is empty"

    def test_no_undeclared_figures(self):
        """A PNG no script declares means the plans and the figures have drifted.

        eda.py and train.py both write into docs/figures/, so the declared contents of
        that directory are the union of their two figure lists and the check has to
        know about both. Asserting against the EDA's list alone would fail the moment
        the modeling phase writes its first figure, which is drift in the test rather
        than in the figures.

        Subset rather than equality: this test guards against strays, and the
        per-figure existence check above is what guards against a missing EDA figure.
        train.py's own figures are checked in test_train.py.
        """
        if not FIGURES_DIR.exists():
            pytest.skip(f"{FIGURES_DIR} not found. Run: python model/src/eda.py")
        found = {p.name for p in FIGURES_DIR.glob("*.png")}
        declared = set(EXPECTED_FIGURES) | set(MODELING_FIGURES)
        assert found <= declared, f"undeclared figures: {sorted(found - declared)}"


class TestPeekingRule:
    def test_training_frame_matches_make_split_defaults(self, raw_df):
        """The frame every target-related figure is stamped with.

        The EDA calls make_split(df) with no arguments, so this is the frame it gets.
        If these shapes move, every "training split (n=76,329)" label in the findings
        and on the figures is wrong.
        """
        train, test = make_split(raw_df)
        assert len(train) == EXPECTED_TRAIN_ROWS
        assert len(test) == EXPECTED_TEST_ROWS

    def test_training_split_is_a_strict_subset_of_the_full_frame(self, raw_df):
        train, _test = make_split(raw_df)
        assert len(train) < len(raw_df)
        assert set(train.index).issubset(set(raw_df.index))

    def test_target_work_never_sees_the_test_rows(self, raw_df):
        """train and test must not overlap, or the peeking rule is unenforceable."""
        train, test = make_split(raw_df)
        assert not set(train.index) & set(test.index)


class TestWilsonInterval:
    def test_matches_a_known_reference(self):
        """5 successes in 40 trials, the AGE<20 band that task 5 excludes.

        Reference values from the Wilson score formula computed by hand; the point of
        the interval is that 12.5% off 40 rows spans 5% to 26%, which is why that band
        does not get charted as if it were a finding.
        """
        low, high = wilson_interval(5, 40)
        assert low == pytest.approx(0.0546, abs=0.001)
        assert high == pytest.approx(0.2611, abs=0.001)

    def test_interval_brackets_the_observed_rate(self):
        low, high = wilson_interval(3874, 76329)
        assert low < 3874 / 76329 < high

    def test_more_data_narrows_the_interval(self):
        narrow_low, narrow_high = wilson_interval(500, 10000)
        wide_low, wide_high = wilson_interval(5, 100)
        assert (narrow_high - narrow_low) < (wide_high - wide_low)

    def test_stays_inside_zero_and_one(self):
        assert wilson_interval(0, 50)[0] == 0.0
        assert wilson_interval(50, 50)[1] == 1.0

    def test_zero_trials_does_not_divide_by_zero(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)


class TestBlankShare:
    def test_counts_whitespace_only_strings(self):
        series = pd.Series(["X", " ", "", "  ", "Y"], dtype="str")
        assert blank_share(series) == pytest.approx(0.6)

    def test_nan_is_not_blankness(self):
        """NaN is already counted as missing. This measures the other kind."""
        series = pd.Series(["X", None, "Y"], dtype="object")
        assert blank_share(series) == 0.0

    def test_numeric_columns_have_no_blanks(self):
        assert blank_share(pd.Series([1, 2, 3])) == 0.0

    def test_noexch_has_the_seven_blanks_the_plan_documents(self, raw_df):
        assert int((raw_df["NOEXCH"].str.strip() == "").sum()) == 7


class TestNearConstant:
    def test_finds_a_constant_column(self):
        frame = pd.DataFrame({"constant": ["L"] * 100, "varied": list(range(100))})
        result = near_constant_columns(frame)
        assert list(result["column"]) == ["constant"]

    def test_counts_nan_as_a_value(self):
        """A 99%-missing column is as useless as a 99%-single-code one."""
        frame = pd.DataFrame({"mostly_nan": [1.0] + [None] * 199})
        result = near_constant_columns(frame)
        assert "mostly_nan" in list(result["column"])

    def test_rfa_2r_is_the_only_fully_constant_column(self, raw_df):
        """The plan says to prefer RFA_2R/2F/2A. RFA_2R is 'L' for every row."""
        constants = [c for c in raw_df.columns if raw_df[c].nunique(dropna=False) == 1]
        assert constants == ["RFA_2R"]
