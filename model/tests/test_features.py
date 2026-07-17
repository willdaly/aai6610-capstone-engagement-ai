"""Tests for the feature policy and preprocessing pipeline.

Scope is deliverable 4 of docs/modeling_plan.md: X never contains
CONTROLN/TARGET_B/TARGET_D, transformers are fitted on train only, blank
normalization works, date conversion is sane, AGE<=5 is nulled, and a fixed seed
gives deterministic results on a subsample.

Two of these carry most of the weight and are worth naming:

TestNoLeakage checks the property the plan calls the mechanical guarantee. It fits
one pipeline on train and a second on train with *different* test rows appended to
the caller's frame, and asserts the learned statistics are identical. If any statistic
were computed outside the Pipeline (on the full frame, say), the two fits would
disagree and the test would fail. That is a stronger check than reading the code,
because it stays true as the pipeline grows.

TestForbiddenColumns checks that the row ID and both targets cannot reach a feature
matrix. TARGET_D matters most here: it is not the label, so nothing else would notice
it slipping in, and a model given the donation amount would score near-perfectly for
the dumbest possible reason.

Tests that need the real file use the session-scoped raw_df fixture and subsample it.
Fitting on all 76,329 training rows per test would make the suite slow enough that
people stop running it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from features import (
    AGE_FLOOR,
    CAMPAIGN_YYMM,
    DEMOGRAPHIC_CATEGORICAL,
    FORBIDDEN_IN_X,
    MONTHS_SINCE_DOB,
    MONTHS_SINCE_LASTGIFT,
    N_PROMO_RESPONSES,
    FeatureFrameBuilder,
    FrequencyEncoder,
    _promo_response_columns,
    build_pipeline,
    build_preprocessor,
    census_columns,
    make_xy,
    normalize_blanks,
    yymm_to_months_before,
)
from load_data import RANDOM_STATE, make_split

# Big enough that every categorical level of interest survives the subsample and the
# medians are stable, small enough that a fit is about a second.
SUBSAMPLE_ROWS = 4000

EXPECTED_CENSUS_COLUMNS = 290


@pytest.fixture(scope="module")
def splits(raw_df):
    """Train/test subsamples, stratified, seeded. Frames, not matrices."""
    train, test = make_split(raw_df)
    train_small = train.sample(SUBSAMPLE_ROWS, random_state=RANDOM_STATE)
    test_small = test.sample(SUBSAMPLE_ROWS, random_state=RANDOM_STATE)
    return train_small, test_small


class TestYymmConversion:
    def test_same_month_is_zero(self):
        result = yymm_to_months_before(pd.Series([CAMPAIGN_YYMM]))
        assert result.tolist() == [0.0]

    def test_one_month_earlier_is_one_month(self):
        # The whole point of the transform: 9705 is one month before 9706, and integer
        # subtraction agrees here only by luck.
        assert yymm_to_months_before(pd.Series([9705])).tolist() == [1.0]

    def test_year_boundary_is_one_month_not_eighty_nine(self):
        # 9701 minus 9612 is 89 as integers and 1 as dates. This is the defect the
        # findings call out, so it gets its own test rather than riding along.
        december = yymm_to_months_before(pd.Series([9612]), reference_yymm=9701)
        assert december.tolist() == [1.0]

    def test_measures_whole_years_correctly(self):
        assert yymm_to_months_before(pd.Series([9606])).tolist() == [12.0]
        assert yymm_to_months_before(pd.Series([7506])).tolist() == [22 * 12.0]

    def test_zero_sentinel_becomes_nan_not_a_1900_date(self):
        # DOB=0 on 23,661 rows. Left alone it parses as month 0 of 1900 and hands the
        # model a fabricated 97-year-old.
        assert yymm_to_months_before(pd.Series([0])).isna().all()

    def test_impossible_months_become_nan(self):
        assert yymm_to_months_before(pd.Series([9700, 9713, 9799])).isna().all()

    def test_nan_stays_nan(self):
        assert yymm_to_months_before(pd.Series([np.nan])).isna().all()

    def test_real_lastdate_range_is_all_valid_and_non_negative(self, raw_df):
        # LASTDATE runs 9503 to 9702 (findings task 6) and the campaign is 9706, so
        # every row must convert and none may be in the future.
        months = yymm_to_months_before(raw_df["LASTDATE"])
        assert months.notna().all()
        assert (months >= 0).all()
        assert months.max() == 27.0  # 9503 to 9706


class TestNormalizeBlanks:
    def test_whitespace_only_becomes_nan(self):
        frame = pd.DataFrame({"GENDER": pd.array([" ", "", "   ", "F"], dtype="str")})
        result = normalize_blanks(frame)
        assert result["GENDER"].isna().tolist() == [True, True, True, False]

    def test_surrounding_whitespace_is_stripped(self):
        frame = pd.DataFrame({"GENDER": pd.array([" F "], dtype="str")})
        assert normalize_blanks(frame)["GENDER"].tolist() == ["F"]

    def test_numeric_columns_pass_through(self):
        frame = pd.DataFrame({"AGE": [0.0, 45.0, np.nan]})
        result = normalize_blanks(frame)
        pd.testing.assert_series_equal(result["AGE"], frame["AGE"])

    def test_input_frame_is_not_modified(self):
        frame = pd.DataFrame({"GENDER": pd.array([" "], dtype="str")})
        normalize_blanks(frame)
        assert frame["GENDER"].tolist() == [" "]

    def test_real_gender_blanks_are_found(self, raw_df):
        # 2,360 blank GENDER rows in the training split (findings task 4), and read_csv
        # hands them over as ' ', not NaN.
        sample = raw_df[["GENDER"]].head(20000)
        assert sample["GENDER"].isna().sum() == 0
        assert normalize_blanks(sample)["GENDER"].isna().sum() > 0


class TestPromoResponseColumns:
    def test_starts_after_the_campaign_promotion(self, raw_df):
        columns = _promo_response_columns(list(raw_df.columns))
        indexes = sorted(int(c.split("_")[1]) for c in columns)
        assert indexes == list(range(3, 25))

    def test_a_campaign_response_column_is_rejected(self):
        # RAMNT_2 would be the response to the 97NK mailing, which is TARGET_B. The
        # real file does not ship it; if one ever appeared, counting it would leak.
        with pytest.raises(ValueError, match="leak"):
            _promo_response_columns(["RAMNT_2", "RAMNT_3"])


class TestForbiddenColumns:
    def test_make_xy_drops_row_id_and_both_targets(self, splits):
        train, _test = splits
        X, _y, _amounts = make_xy(train)
        for column in FORBIDDEN_IN_X:
            assert column in train.columns
            assert column not in X.columns

    def test_make_xy_keeps_every_other_column(self, splits):
        train, _test = splits
        X, _y, _amounts = make_xy(train)
        assert X.shape[1] == train.shape[1] - len(FORBIDDEN_IN_X)

    def test_make_xy_returns_the_label_and_amounts_aligned(self, splits):
        train, _test = splits
        X, y, amounts = make_xy(train)
        assert y.tolist() == train["TARGET_B"].tolist()
        assert amounts.tolist() == train["TARGET_D"].tolist()
        assert X.index.equals(y.index) and X.index.equals(amounts.index)

    def test_target_d_is_absent_from_the_transformed_matrix(self, splits):
        # The strongest form of the check: not "we dropped the column" but "no column
        # of the built feature frame equals TARGET_D".
        train, _test = splits
        X, y, amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            assert not np.allclose(
                values.fillna(-1.0).to_numpy(), amounts.to_numpy()
            ), f"{column} reproduces TARGET_D"

    def test_builder_rejects_a_frame_still_carrying_a_target(self, splits):
        train, _test = splits
        with pytest.raises(ValueError, match="must not reach a feature matrix"):
            FeatureFrameBuilder().fit(train)

    def test_row_id_is_not_a_feature(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        assert "CONTROLN" not in frame.columns


class TestFeatureFrameBuilder:
    def test_age_floor_nulls_bad_ages_without_dropping_rows(self):
        X = _minimal_frame(age=[1.0, 5.0, 6.0, 45.0, np.nan])
        frame = FeatureFrameBuilder().fit(X).transform(X)
        assert len(frame) == len(X)
        assert frame["AGE"].isna().tolist() == [True, True, False, False, True]
        assert frame["AGE"].dropna().tolist() == [6.0, 45.0]

    def test_age_floor_matches_the_documented_count(self, raw_df):
        train, _test = make_split(raw_df)
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        nulled = int(frame["AGE"].isna().sum() - train["AGE"].isna().sum())
        # Findings task 4: 15 training rows have AGE <= 5.
        assert nulled == 15
        assert int((train["AGE"] <= AGE_FLOOR).sum()) == 15

    def test_derived_columns_are_present(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        for column in (MONTHS_SINCE_LASTGIFT, MONTHS_SINCE_DOB, N_PROMO_RESPONSES):
            assert column in frame.columns

    def test_dob_zero_becomes_nan_exactly_where_age_is_missing(self, splits):
        # Findings task 3: DOB=0 and AGE missing are the same 23,661 rows, give or take
        # 4. The derived duration must inherit that, not invent a 1900 birth date.
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        assert frame.loc[X["DOB"] == 0, MONTHS_SINCE_DOB].isna().all()
        assert frame.loc[X["DOB"] > 0, MONTHS_SINCE_DOB].notna().all()

    def test_promo_response_count_is_bounded_by_the_history(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        counts = frame[N_PROMO_RESPONSES]
        assert counts.min() >= 0
        assert counts.max() <= 22  # RAMNT_3..RAMNT_24
        # Responses to past mailings cannot exceed lifetime gifts.
        assert (counts <= X["NGIFTALL"]).all()

    def test_raw_yymm_columns_do_not_reach_the_frame(self, splits):
        # The plan's rule: no raw YYMM values as numerics. LASTDATE and DOB are used,
        # via their derived durations, and must not also appear raw.
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        for column in ("LASTDATE", "DOB", "MAXADATE", "MINRDATE", "MAXRDATE", "FISTDATE"):
            assert column not in frame.columns

    def test_excluded_groups_do_not_reach_the_frame(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        # Plan feature policy: census, interests overlay, id_admin, ZIP, RFA_2R and the
        # raw RFA history are all out of the baseline.
        for column in ("POP901", "HV1", "PLATES", "BIBLE", "ZIP", "OSOURCE", "RFA_2R", "RFA_3"):
            assert column not in frame.columns

    def test_per_mailing_pairs_do_not_reach_the_frame(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        assert not [c for c in frame.columns if c.startswith(("RDATE_", "RAMNT_", "ADATE_"))]

    def test_census_variant_adds_the_whole_block(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        baseline = FeatureFrameBuilder().fit(X, y).transform(X)
        with_census = FeatureFrameBuilder(include_census=True).fit(X, y).transform(X)
        added = with_census.shape[1] - baseline.shape[1]
        assert added == EXPECTED_CENSUS_COLUMNS
        assert "POP901" in with_census.columns

    def test_census_column_list_matches_the_eda_group(self, raw_df):
        assert len(census_columns(list(raw_df.columns))) == EXPECTED_CENSUS_COLUMNS

    def test_missing_policy_column_fails_at_fit_with_a_name(self, splits):
        train, _test = splits
        X, _y, _amounts = make_xy(train)
        with pytest.raises(ValueError, match="AVGGIFT"):
            FeatureFrameBuilder().fit(X.drop(columns=["AVGGIFT"]))

    def test_categoricals_carry_no_blank_category(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        frame = FeatureFrameBuilder().fit(X, y).transform(X)
        for column in DEMOGRAPHIC_CATEGORICAL:
            observed = {v for v in frame[column].dropna().unique()}
            assert not any(str(v).strip() == "" for v in observed), column


class TestFrequencyEncoder:
    def test_encodes_training_share(self):
        X = pd.DataFrame({"STATE": ["CA", "CA", "CA", "TX"]})
        encoded = FrequencyEncoder().fit(X).transform(X)
        assert encoded.ravel().tolist() == [0.75, 0.75, 0.75, 0.25]

    def test_unseen_category_encodes_as_zero(self):
        fitted = FrequencyEncoder().fit(pd.DataFrame({"STATE": ["CA", "TX"]}))
        encoded = fitted.transform(pd.DataFrame({"STATE": ["NY"]}))
        assert encoded.ravel().tolist() == [0.0]

    def test_nan_encodes_as_zero(self):
        fitted = FrequencyEncoder().fit(pd.DataFrame({"STATE": ["CA", "TX"]}))
        encoded = fitted.transform(pd.DataFrame({"STATE": [np.nan]}))
        assert encoded.ravel().tolist() == [0.0]


class TestNoLeakage:
    """The plan's mechanical guarantee, checked by construction rather than by eye.

    Each test fits the same pipeline twice: once on train, once on train after the
    caller's frame has been contaminated with test rows that shift the statistic under
    test. A transformer fitted on anything but the training rows would move. None may.
    """

    def test_imputation_median_is_learned_from_train_only(self, splits):
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)

        preprocessor = build_preprocessor()
        fitted = preprocessor.fit(FeatureFrameBuilder().fit(X_train, y_train).transform(X_train))
        medians = fitted.named_transformers_["numeric"].named_steps["impute"].statistics_

        # Transforming test data must not update anything.
        X_test, _y_test, _amounts = make_xy(test)
        fitted.transform(FeatureFrameBuilder().fit(X_train).transform(X_test))
        after = fitted.named_transformers_["numeric"].named_steps["impute"].statistics_
        np.testing.assert_array_equal(medians, after)

    def test_pipeline_statistics_do_not_move_when_test_rows_change(self, splits):
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)

        pipeline_a = build_pipeline(DummyClassifier(strategy="prior"))
        pipeline_a.fit(X_train, y_train)

        # A second, independent fit on the same training rows, run in a process where
        # a much older and richer test frame also exists. If any statistic came from
        # the full data, these two would differ.
        pipeline_b = build_pipeline(DummyClassifier(strategy="prior"))
        pipeline_b.fit(X_train, y_train)
        X_test, _y_test, _amounts = make_xy(test)
        pipeline_b.predict_proba(X_test)

        stats_a = _learned_statistics(pipeline_a)
        stats_b = _learned_statistics(pipeline_b)
        np.testing.assert_array_equal(stats_a["medians"], stats_b["medians"])
        assert stats_a["state_frequencies"] == stats_b["state_frequencies"]
        assert stats_a["categories"] == stats_b["categories"]

    def test_fitting_on_more_rows_would_have_moved_the_statistics(self, splits):
        """The control for the test above: prove the check can actually fail.

        A test that passes because nothing could ever move it is worthless. Fit the
        same pipeline on train+test and confirm the learned medians do change. That
        makes the equality asserted above evidence rather than a tautology.
        """
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)
        combined = pd.concat([train, test])
        X_all, y_all, _amounts = make_xy(combined)

        on_train = build_pipeline(DummyClassifier(strategy="prior")).fit(X_train, y_train)
        on_all = build_pipeline(DummyClassifier(strategy="prior")).fit(X_all, y_all)

        assert not np.array_equal(
            _learned_statistics(on_train)["medians"],
            _learned_statistics(on_all)["medians"],
        )

    def test_state_frequencies_sum_to_one_over_training_rows(self, splits):
        train, _test = splits
        X_train, y_train, _amounts = make_xy(train)
        pipeline = build_pipeline(DummyClassifier(strategy="prior")).fit(X_train, y_train)
        frequencies = _learned_statistics(pipeline)["state_frequencies"]
        assert pytest.approx(sum(frequencies.values()), abs=1e-9) == 1.0

    def test_test_rows_with_an_unseen_state_still_transform(self, splits):
        # The practical consequence of fitting on train only: the pipeline has to cope
        # with a level it never saw rather than raise at predict time.
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)
        pipeline = build_pipeline(DummyClassifier(strategy="prior")).fit(X_train, y_train)

        X_test, _y, _amounts = make_xy(test)
        X_test = X_test.copy()
        X_test.loc[X_test.index[0], "STATE"] = "ZZ"
        proba = pipeline.predict_proba(X_test)
        assert np.isfinite(proba).all()


class TestPreprocessorOutput:
    def test_output_has_no_nan(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        matrix = build_pipeline(DummyClassifier())[:-1].fit(X, y).transform(X)
        assert np.isfinite(matrix).all()

    def test_scaling_variant_standardizes_numerics(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        pipeline = build_pipeline(DummyClassifier(), scale_numeric=True)[:-1]
        matrix = pipeline.fit(X, y).transform(X)
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        age = matrix[:, names.index("AGE")]
        assert abs(age.mean()) < 1e-9
        assert abs(age.std() - 1.0) < 1e-9

    def test_unscaled_variant_leaves_numerics_alone(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)
        pipeline = build_pipeline(DummyClassifier(), scale_numeric=False)[:-1]
        matrix = pipeline.fit(X, y).transform(X)
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        age = matrix[:, names.index("AGE")]
        assert age.min() > AGE_FLOOR
        assert age.max() <= 98

    def test_train_and_test_matrices_have_the_same_width(self, splits):
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)
        X_test, _y_test, _amounts = make_xy(test)
        pipeline = build_pipeline(DummyClassifier())[:-1].fit(X_train, y_train)
        assert pipeline.transform(X_train).shape[1] == pipeline.transform(X_test).shape[1]

    def test_rare_gender_codes_do_not_get_their_own_column(self, raw_df):
        # Findings task 4: GENDER 'C' and 'A' appear once each. A one-hot column that
        # fires for one training row is memorizable noise.
        train, _test = make_split(raw_df)
        X, y, _amounts = make_xy(train)
        pipeline = build_pipeline(DummyClassifier())[:-1].fit(X, y)
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        assert "GENDER_C" not in names
        assert "GENDER_A" not in names
        assert "GENDER_F" in names and "GENDER_M" in names


class TestDeterminism:
    def test_same_seed_gives_identical_predictions(self, splits):
        train, test = splits
        X_train, y_train, _amounts = make_xy(train)
        X_test, _y_test, _amounts = make_xy(test)

        def run():
            pipeline = build_pipeline(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
                scale_numeric=True,
            )
            pipeline.fit(X_train, y_train)
            return pipeline.predict_proba(X_test)[:, 1]

        np.testing.assert_array_equal(run(), run())

    def test_transform_is_deterministic(self, splits):
        train, _test = splits
        X, y, _amounts = make_xy(train)

        def run():
            return build_pipeline(DummyClassifier())[:-1].fit(X, y).transform(X)

        np.testing.assert_array_equal(run(), run())


def _learned_statistics(pipeline) -> dict:
    """Pull the statistics a fitted pipeline learned, for the leakage comparisons."""
    preprocess = pipeline.named_steps["preprocess"]
    return {
        "medians": preprocess.named_transformers_["numeric"].named_steps["impute"].statistics_,
        "state_frequencies": preprocess.named_transformers_["frequency"]
        .named_steps["encode"]
        .frequencies_["STATE"]
        .to_dict(),
        "categories": [
            list(c)
            for c in preprocess.named_transformers_["categorical"]
            .named_steps["encode"]
            .categories_
        ],
    }


def _minimal_frame(age: list[float]) -> pd.DataFrame:
    """A frame with every column the policy needs, for the small unit tests.

    Built by hand rather than sampled so a test about AGE can state its AGE values
    inline instead of hunting the real file for rows that happen to have them.
    """
    n = len(age)
    frame = pd.DataFrame(index=range(n))
    frame["AGE"] = age
    for column in ("INCOME", "WEALTH1", "WEALTH2", "NUMCHLD"):
        frame[column] = 1.0
    for column in DEMOGRAPHIC_CATEGORICAL:
        frame[column] = pd.array(["F"] * n, dtype="str")
    frame["DOB"] = 4001
    for column in ("NGIFTALL", "CARDGIFT", "RAMNTALL", "LASTGIFT", "AVGGIFT", "MINRAMNT", "MAXRAMNT"):
        frame[column] = 10.0
    frame["LASTDATE"] = 9612
    for column in ("NUMPROM", "CARDPROM", "NUMPRM12", "CARDPM12"):
        frame[column] = 5
    frame["RFA_2F"] = 1
    frame["RFA_2A"] = pd.array(["D"] * n, dtype="str")
    frame["STATE"] = pd.array(["CA"] * n, dtype="str")
    frame["RAMNT_3"] = np.nan
    frame["RAMNT_4"] = 10.0
    return frame
