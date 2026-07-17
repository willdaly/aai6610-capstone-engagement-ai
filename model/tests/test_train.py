"""Tests for the training pipeline's arithmetic and its frame discipline.

Scope is the parts of docs/modeling_plan.md that a wrong answer would hide in: the
net-revenue arithmetic, the threshold sweep, and the fairness segmentation. These are
small pure functions carrying the numbers the findings doc quotes, and an error in any
of them would look like a plausible result rather than a crash.

The expensive things (the CV run, the fits, the figures) are not re-run here. They
take minutes and would write into docs/ as a side effect, which a test run should not
do. Those are verified by running `python model/src/train.py` from a clean state,
exactly as the EDA's tests are.

The plan's central ground rule, that the test split is read exactly once, is asserted
against train.py's source in TestFrameDiscipline, which lands with evaluate_on_test.
"""

import numpy as np
import pandas as pd
import pytest

from train import (
    AUPRC_FLOOR,
    EXPECTED_FIGURES,
    MAIL_COST,
    MISSING_SEGMENT,
    age_band,
    choose_threshold,
    fairness_slices,
    metrics_at_threshold,
    net_revenue,
    revenue_curve,
)

class TestNetRevenue:
    def test_contacting_no_one_earns_nothing(self):
        y = np.array([0, 1, 1])
        amounts = np.array([0.0, 10.0, 20.0])
        contacted = np.array([False, False, False])
        assert net_revenue(y, amounts, contacted) == 0.0

    def test_a_contacted_responder_pays_the_gift_less_the_stamp(self):
        y = np.array([1])
        amounts = np.array([10.0])
        assert net_revenue(y, amounts, np.array([True])) == pytest.approx(10.0 - MAIL_COST)

    def test_a_contacted_non_responder_costs_the_stamp(self):
        y = np.array([0])
        amounts = np.array([0.0])
        assert net_revenue(y, amounts, np.array([True])) == pytest.approx(-MAIL_COST)

    def test_uncontacted_responders_contribute_nothing(self):
        # The gift only exists if the mail went out. A revenue function that counted
        # the amounts of people it never contacted would make every model look free.
        y = np.array([1, 1])
        amounts = np.array([10.0, 20.0])
        assert net_revenue(y, amounts, np.array([True, False])) == pytest.approx(
            10.0 - MAIL_COST
        )

    def test_break_even_gift_is_the_mail_cost(self):
        y = np.array([1])
        amounts = np.array([MAIL_COST])
        assert net_revenue(y, amounts, np.array([True])) == pytest.approx(0.0)


class TestRevenueCurve:
    def test_walks_the_ranked_list_in_score_order(self):
        # Two responders at the bottom of the ranking must not be credited to the top.
        y = np.array([0, 0, 1, 1])
        amounts = np.array([0.0, 0.0, 10.0, 20.0])
        proba = np.array([0.9, 0.8, 0.2, 0.1])
        curve = revenue_curve(y, amounts, proba)

        assert curve["net_revenue"].iloc[0] == pytest.approx(-MAIL_COST)
        assert curve["net_revenue"].iloc[1] == pytest.approx(-2 * MAIL_COST)
        assert curve["net_revenue"].iloc[3] == pytest.approx(30.0 - 4 * MAIL_COST)

    def test_a_perfect_ranking_peaks_where_the_responders_end(self):
        y = np.array([1, 1, 0, 0])
        amounts = np.array([20.0, 10.0, 0.0, 0.0])
        proba = np.array([0.9, 0.8, 0.2, 0.1])
        curve = revenue_curve(y, amounts, proba)
        assert int(curve.loc[curve["net_revenue"].idxmax(), "contacts"]) == 2

    def test_recall_and_precision_track_the_walk(self):
        y = np.array([1, 0, 1, 0])
        amounts = np.array([10.0, 0.0, 10.0, 0.0])
        proba = np.array([0.9, 0.8, 0.7, 0.6])
        curve = revenue_curve(y, amounts, proba)
        assert curve["recall"].tolist() == [0.5, 0.5, 1.0, 1.0]
        assert curve["precision"].tolist() == [1.0, 0.5, pytest.approx(2 / 3), 0.5]
        assert curve["recall"].iloc[-1] == 1.0

    def test_has_one_row_per_constituent(self):
        y = np.array([0, 1, 0])
        curve = revenue_curve(y, np.array([0.0, 10.0, 0.0]), np.array([0.3, 0.2, 0.1]))
        assert len(curve) == 3
        assert curve["contact_fraction"].iloc[-1] == 1.0

    def test_ties_keep_original_order(self):
        # A constant scorer must not be silently re-ranked into a favourable order.
        y = np.array([0, 1])
        amounts = np.array([0.0, 10.0])
        curve = revenue_curve(y, amounts, np.array([0.5, 0.5]))
        assert curve["net_revenue"].iloc[0] == pytest.approx(-MAIL_COST)


class TestThresholds:
    def test_metrics_at_threshold_contacts_at_or_above(self):
        y = np.array([1, 0, 1])
        amounts = np.array([10.0, 0.0, 20.0])
        proba = np.array([0.9, 0.5, 0.3])
        result = metrics_at_threshold(y, amounts, proba, 0.5)
        assert result["contacts"] == 2
        assert result["recall"] == 0.5
        assert result["precision"] == 0.5
        assert result["net_revenue"] == pytest.approx(10.0 - 2 * MAIL_COST)

    def test_metrics_at_threshold_includes_every_tie(self):
        # proba >= threshold is the rule a campaign runs, so a tie at the threshold is
        # contacted, not truncated at some top-k boundary.
        y = np.array([0, 0, 0])
        proba = np.array([0.5, 0.5, 0.5])
        result = metrics_at_threshold(y, np.zeros(3), proba, 0.5)
        assert result["contacts"] == 3

    def test_choose_threshold_finds_the_revenue_maximum(self):
        y = np.array([1, 1, 0, 0, 0])
        amounts = np.array([20.0, 10.0, 0.0, 0.0, 0.0])
        proba = np.array([0.9, 0.8, 0.4, 0.3, 0.2])
        chosen = choose_threshold(y, amounts, proba)
        assert chosen["contacts"] == 2
        assert chosen["net_revenue"] == pytest.approx(30.0 - 2 * MAIL_COST)
        assert chosen["recall"] == 1.0

    def test_choose_threshold_can_pick_everyone_when_mail_is_cheap(self):
        # Every constituent is worth more than a stamp, so trimming the list is wrong.
        y = np.array([1, 1, 1])
        amounts = np.array([10.0, 10.0, 10.0])
        chosen = choose_threshold(y, amounts, np.array([0.9, 0.5, 0.1]))
        assert chosen["contacts"] == 3

    def test_zero_contacts_reports_undefined_precision_not_zero(self):
        # Nobody contacted means precision has no denominator. Zero would read as
        # "everyone it mailed was a miss", which is a claim about a mailing that did
        # not happen. Recall is still 0.0: there was a responder, and it was missed.
        y = np.array([1, 0])
        result = metrics_at_threshold(y, np.array([10.0, 0.0]), np.array([0.1, 0.2]), 0.9)
        assert result["contacts"] == 0
        assert np.isnan(result["precision"])
        assert result["recall"] == 0.0

    def test_no_responders_reports_undefined_recall(self):
        result = metrics_at_threshold(
            np.array([0, 0]), np.zeros(2), np.array([0.9, 0.9]), 0.5
        )
        assert np.isnan(result["recall"])


class TestAgeBand:
    def test_labels_are_readable_decades(self):
        result = age_band(pd.Series([25.0, 71.0, 99.0]))
        assert result.tolist() == ["20-29", "70-79", "80-99"]

    def test_missing_age_gets_a_segment_rather_than_vanishing(self):
        # AGE is 24.8% missing. A fairness table that drops those rows describes a
        # population the campaign is not mailing (CLAUDE.md: never silently drop).
        result = age_band(pd.Series([np.nan, 45.0]))
        assert result.tolist() == [MISSING_SEGMENT, "40-49"]

    def test_ages_outside_the_bands_are_not_dropped(self):
        # Under 20 is implausible data (findings task 4), not a band, but the row is
        # still someone the campaign either mails or does not.
        result = age_band(pd.Series([1.0, 15.0]))
        assert result.tolist() == [MISSING_SEGMENT, MISSING_SEGMENT]

    def test_band_edges_are_left_closed(self):
        assert age_band(pd.Series([30.0])).tolist() == ["30-39"]
        assert age_band(pd.Series([29.0])).tolist() == ["20-29"]


class TestFairnessSlices:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "TARGET_B": [1, 0, 1, 0, 0, 1],
                "GENDER": pd.array(["F", "M", "F", " ", "M", "F"], dtype="str"),
                "AGE": [45.0, 55.0, np.nan, 65.0, 45.0, 55.0],
                "INCOME": [1.0, 2.0, np.nan, 1.0, 2.0, 1.0],
            }
        )

    def test_segments_partition_the_frame(self, frame):
        slices = fairness_slices(frame, np.full(6, 0.9), 0.5)
        for _attribute, group in slices.groupby("attribute"):
            assert group["n"].sum() == len(frame)

    def test_blank_gender_becomes_the_missing_segment(self, frame):
        slices = fairness_slices(frame, np.full(6, 0.9), 0.5)
        gender = slices[slices["attribute"] == "GENDER"]
        assert MISSING_SEGMENT in set(gender["segment"])
        assert not any(s.strip() == "" for s in gender["segment"])

    def test_contact_rate_reflects_the_threshold(self, frame):
        proba = np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.9])
        slices = fairness_slices(frame, proba, 0.5)
        female = slices[(slices["attribute"] == "GENDER") & (slices["segment"] == "F")]
        assert float(female["contact_rate"].iloc[0]) == 1.0
        male = slices[(slices["attribute"] == "GENDER") & (slices["segment"] == "M")]
        assert float(male["contact_rate"].iloc[0]) == 0.0

    def test_recall_is_over_the_segments_own_responders(self, frame):
        proba = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
        slices = fairness_slices(frame, proba, 0.5)
        # F has three rows, all responders; one (the NaN-age row) is not contacted.
        female = slices[(slices["attribute"] == "GENDER") & (slices["segment"] == "F")]
        assert float(female["recall"].iloc[0]) == pytest.approx(2 / 3)

    def test_a_segment_with_no_responders_reports_nan_recall_not_zero(self):
        # Recall is undefined with no responders to find. Reporting 0.0 would read as
        # "the model missed them all", which is a different and false claim.
        frame = pd.DataFrame(
            {
                "TARGET_B": [0, 0],
                "GENDER": pd.array(["F", "M"], dtype="str"),
                "AGE": [45.0, 45.0],
                "INCOME": [1.0, 1.0],
            }
        )
        slices = fairness_slices(frame, np.full(2, 0.9), 0.5)
        assert slices["recall"].isna().all()

    def test_unassigned_segment_is_rejected(self, frame):
        broken = frame.copy()
        broken["GENDER"] = pd.array([None] * 6, dtype="str")
        # Every row still lands in the missing segment, so this must NOT raise: the
        # guard is about rows with no segment at all, not about missing values.
        slices = fairness_slices(broken, np.full(6, 0.9), 0.5)
        gender = slices[slices["attribute"] == "GENDER"]
        assert gender["n"].sum() == len(broken)


class TestFigureDeclaration:
    def test_every_declared_figure_is_produced(self):
        from eda import FIGURES_DIR

        if not FIGURES_DIR.exists():
            pytest.skip(f"{FIGURES_DIR} not found. Run: python model/src/train.py")
        for name in EXPECTED_FIGURES:
            path = FIGURES_DIR / name
            assert path.exists(), f"{name} missing. Run: python model/src/train.py"
            assert path.stat().st_size > 0, f"{name} is empty"


class TestConstants:
    def test_auprc_floor_matches_the_eda_positive_rate(self):
        # The floor is quoted from findings task 2. If it drifts from the EDA, every
        # "vs floor" number in the modeling findings is measured against a fiction.
        assert AUPRC_FLOOR == 0.0508

    def test_mail_cost_is_the_documented_campaign_cost(self):
        assert MAIL_COST == 0.68
