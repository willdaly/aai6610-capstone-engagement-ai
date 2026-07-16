"""Tests for loading the KDD Cup 1998 learning set and splitting it.

These pin the dataset facts recorded in CLAUDE.md and the course-wide split
convention. A failure here means either the data file is not what we think it is, or
the split changed underneath results that were already reported. Both are worth
stopping for.
"""

import pandas as pd
import pytest

from load_data import (
    EXPECTED_SHAPE,
    RANDOM_STATE,
    TARGET_BINARY,
    load_raw,
    make_split,
)

# Documented in CLAUDE.md, verified 2026-07-16.
EXPECTED_POSITIVES = 4843
EXPECTED_POSITIVE_RATE = 0.0508
# Stratification tolerance: 0.1 percentage points either side of the overall rate.
RATE_TOLERANCE = 0.001

# Seeds for the exact-count stratification check below. 2026 is the convention; the
# others are arbitrary. The point is to vary the seed, so keep more than one.
STRATIFICATION_SEEDS = [1, 7, 42, 2026, 12345]


class TestLoadRaw:
    def test_shape_is_95412_by_481(self, raw_df):
        assert raw_df.shape == (95412, 481)
        assert raw_df.shape == EXPECTED_SHAPE

    def test_target_b_has_exactly_4843_positives(self, raw_df):
        assert int((raw_df[TARGET_BINARY] == 1).sum()) == EXPECTED_POSITIVES

    def test_target_b_is_binary(self, raw_df):
        assert set(raw_df[TARGET_BINARY].unique()) == {0, 1}

    def test_positive_rate_is_about_5_08_percent(self, raw_df):
        rate = raw_df[TARGET_BINARY].mean()
        assert rate == pytest.approx(EXPECTED_POSITIVE_RATE, abs=RATE_TOLERANCE)

    def test_noexch_keeps_its_non_numeric_codes(self, raw_df):
        # The 42 rows holding 'X' or a space are why NOEXCH is declared as a string.
        # If inference ever coerces them to NaN, this catches it.
        counts = raw_df["NOEXCH"].value_counts()
        assert counts["X"] == 35
        assert counts[" "] == 7

    def test_missingness_matches_documented_rates(self, raw_df):
        # CLAUDE.md: AGE 24.8%, INCOME 22.3%. Never silently dropped.
        assert raw_df["AGE"].isna().mean() == pytest.approx(0.248, abs=0.001)
        assert raw_df["INCOME"].isna().mean() == pytest.approx(0.223, abs=0.001)

    def test_missing_file_says_how_to_fix_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.sh"):
            load_raw(path=tmp_path / "absent.txt")

    def test_wrong_shape_is_rejected(self, tmp_path):
        bad = tmp_path / "truncated.txt"
        bad.write_text("TARGET_B,TARGET_D\n1,10.0\n")
        with pytest.raises(ValueError, match="expected"):
            load_raw(path=bad)

    def test_validate_false_allows_a_non_conforming_file(self, tmp_path):
        bad = tmp_path / "truncated.txt"
        bad.write_text("TARGET_B,TARGET_D\n1,10.0\n")
        assert load_raw(path=bad, validate=False).shape == (1, 2)


class TestMakeSplit:
    def test_train_test_shapes(self, raw_df):
        train, test = make_split(raw_df)
        assert train.shape == (76329, 481)
        assert test.shape == (19083, 481)

    def test_test_portion_is_20_percent(self, raw_df):
        train, test = make_split(raw_df)
        assert len(test) / len(raw_df) == pytest.approx(0.2, abs=0.001)

    def test_split_partitions_every_row_exactly_once(self, raw_df):
        train, test = make_split(raw_df)
        assert len(train) + len(test) == len(raw_df)
        assert set(train.index).isdisjoint(test.index)
        assert set(train.index) | set(test.index) == set(raw_df.index)

    @pytest.mark.parametrize("part", ["train", "test"])
    def test_train_test_stratified_within_a_tenth_of_a_point(self, raw_df, part):
        train, test = make_split(raw_df)
        rate = {"train": train, "test": test}[part][TARGET_BINARY].mean()
        assert rate == pytest.approx(EXPECTED_POSITIVE_RATE, abs=RATE_TOLERANCE)

    @pytest.mark.parametrize("seed", STRATIFICATION_SEEDS)
    def test_stratification_holds_at_every_seed(self, raw_df, seed):
        # The 0.1pp check above states the convention but does not enforce it: at
        # n=19083 the positive rate has a sampling std of ~0.16pp, so an unstratified
        # split lands inside 0.1pp often enough to pass by luck, and at seed 2026 it
        # does exactly that (968 positives against 968.63 expected).
        #
        # Stratification's real signature is that the class counts come out
        # proportional at *any* seed, not just a lucky one. Hence: exact counts,
        # tolerance of one row for rounding, several seeds.
        train, test = make_split(raw_df, random_state=seed)
        for part in (train, test):
            expected = len(part) * EXPECTED_POSITIVES / len(raw_df)
            assert int(part[TARGET_BINARY].sum()) == pytest.approx(expected, abs=1)

    def test_split_is_reproducible(self, raw_df):
        first_train, first_test = make_split(raw_df)
        second_train, second_test = make_split(raw_df)
        assert first_train.index.equals(second_train.index)
        assert first_test.index.equals(second_test.index)

    def test_validation_split_is_reproducible(self, raw_df):
        first = make_split(raw_df, validation=True)
        second = make_split(raw_df, validation=True)
        for a, b in zip(first, second):
            assert a.index.equals(b.index)

    def test_default_random_state_is_the_course_convention(self, raw_df):
        # 2026 is course-wide. If the default drifts, every reported metric moves.
        assert RANDOM_STATE == 2026
        default_train, default_test = make_split(raw_df)
        explicit_train, explicit_test = make_split(raw_df, random_state=2026)
        assert default_train.index.equals(explicit_train.index)
        assert default_test.index.equals(explicit_test.index)

    def test_a_different_seed_gives_a_different_split(self, raw_df):
        _, test_2026 = make_split(raw_df)
        _, test_other = make_split(raw_df, random_state=7)
        assert not test_2026.index.equals(test_other.index)


class TestValidationCarveOut:
    def test_shapes(self, raw_df):
        train, val, test = make_split(raw_df, validation=True)
        assert train.shape == (64879, 481)
        assert val.shape == (11450, 481)
        assert test.shape == (19083, 481)

    def test_validation_is_15_percent_of_the_training_portion(self, raw_df):
        train, val, test = make_split(raw_df, validation=True)
        training_portion = len(train) + len(val)
        # 15% of train, not 15% of the whole dataset. Those differ by ~3,000 rows.
        assert len(val) / training_portion == pytest.approx(0.15, abs=0.001)
        assert training_portion == 76329

    def test_carve_out_partitions_every_row_exactly_once(self, raw_df):
        train, val, test = make_split(raw_df, validation=True)
        assert len(train) + len(val) + len(test) == len(raw_df)
        indices = [set(p.index) for p in (train, val, test)]
        assert set.union(*indices) == set(raw_df.index)
        assert sum(len(i) for i in indices) == len(raw_df)  # pairwise disjoint

    def test_test_set_is_untouched_by_requesting_validation(self, raw_df):
        _, test_without = make_split(raw_df)
        _, _, test_with = make_split(raw_df, validation=True)
        assert test_without.index.equals(test_with.index)

    def test_validation_comes_out_of_train_not_test(self, raw_df):
        _, test_without = make_split(raw_df)
        train, val, _ = make_split(raw_df, validation=True)
        assert set(val.index).isdisjoint(test_without.index)
        assert set(train.index).isdisjoint(test_without.index)

    @pytest.mark.parametrize("part", ["train", "val", "test"])
    def test_all_three_parts_stratified_within_a_tenth_of_a_point(self, raw_df, part):
        train, val, test = make_split(raw_df, validation=True)
        rate = {"train": train, "val": val, "test": test}[part][TARGET_BINARY].mean()
        assert rate == pytest.approx(EXPECTED_POSITIVE_RATE, abs=RATE_TOLERANCE)


class TestMakeSplitGuards:
    def test_missing_target_is_rejected(self, raw_df):
        with pytest.raises(ValueError, match=TARGET_BINARY):
            make_split(raw_df.drop(columns=[TARGET_BINARY]))

    def test_single_class_target_is_rejected(self):
        single = pd.DataFrame({TARGET_BINARY: [0] * 10})
        with pytest.raises(ValueError, match="single class"):
            make_split(single)
