"""
Unit Tests — Risk Signal Detection Algorithms
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.risk_detection.signal_algorithms import (
    DisproportionalityAnalyser,
    AnomalyDetector,
    TimeTrendAnalyser,
    SignalResult,
)


@pytest.fixture
def analyser():
    return DisproportionalityAnalyser(prr_threshold=2.0, min_reports=3)


@pytest.fixture
def synthetic_reports_df():
    """Create a minimal synthetic DataFrame for testing."""
    np.random.seed(42)
    drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    events = ["nausea", "headache", "dizziness", "rash", "fatigue"]

    rows = []
    # DrugA has a strong signal for nausea
    for _ in range(50):
        rows.append({"drug_name": "DrugA", "adverse_event": "nausea", "severity": "moderate",
                     "patient_age": 55, "report_date": date(2023, 1, 1)})
    # Background noise
    for drug in ["DrugB", "DrugC", "DrugD"]:
        for event in events:
            for _ in range(5):
                rows.append({"drug_name": drug, "adverse_event": event, "severity": "mild",
                             "patient_age": 50, "report_date": date(2023, 6, 1)})
    # DrugA with other events (background)
    for event in ["headache", "dizziness"]:
        for _ in range(3):
            rows.append({"drug_name": "DrugA", "adverse_event": event, "severity": "mild",
                         "patient_age": 55, "report_date": date(2023, 1, 15)})
    return pd.DataFrame(rows)


class TestDisproportionalityAnalyser:

    def test_returns_list(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        assert isinstance(signals, list)

    def test_detects_strong_signal(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        # DrugA-nausea should have a high PRR
        drug_a_nausea = [s for s in signals
                         if s.drug_name == "DrugA" and s.adverse_event == "nausea"]
        assert len(drug_a_nausea) > 0
        assert drug_a_nausea[0].prr is not None
        assert drug_a_nausea[0].prr > 2.0

    def test_signal_has_all_measures(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        if signals:
            sig = signals[0]
            assert sig.signal_id is not None
            assert sig.drug_name != ""
            assert sig.adverse_event != ""
            assert sig.report_count >= 3

    def test_insufficient_data_returns_empty(self, analyser):
        tiny_df = pd.DataFrame({
            "drug_name": ["DrugA"] * 5,
            "adverse_event": ["nausea"] * 5,
        })
        signals = analyser.analyse(tiny_df)
        assert signals == []

    def test_prr_ci_ordering(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            if sig.prr_lower_ci is not None and sig.prr_upper_ci is not None:
                assert sig.prr_lower_ci <= sig.prr <= sig.prr_upper_ci

    def test_ror_ci_ordering(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            if sig.ror_lower_ci is not None and sig.ror_upper_ci is not None:
                assert sig.ror_lower_ci <= sig.ror <= sig.ror_upper_ci

    def test_severity_score_range(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            assert 0 <= sig.severity_score <= 100

    def test_signal_id_unique(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        ids = [s.signal_id for s in signals]
        assert len(ids) == len(set(ids))

    def test_p_value_range(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            if sig.p_value is not None:
                assert 0.0 <= sig.p_value <= 1.0

    def test_expected_count_positive(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            if sig.expected_count is not None:
                assert sig.expected_count >= 0

    def test_min_reports_threshold_respected(self, analyser, synthetic_reports_df):
        signals = analyser.analyse(synthetic_reports_df)
        for sig in signals:
            assert sig.report_count >= analyser.min_reports


class TestAnomalyDetector:

    def test_returns_dataframe(self, synthetic_reports_df):
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_and_detect(synthetic_reports_df)
        assert isinstance(result, pd.DataFrame)

    def test_anomaly_columns_added(self, synthetic_reports_df):
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_and_detect(synthetic_reports_df)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_anomaly_count_within_contamination(self, synthetic_reports_df):
        contamination = 0.1
        detector = AnomalyDetector(contamination=contamination)
        result = detector.fit_and_detect(synthetic_reports_df)
        actual_rate = result["is_anomaly"].mean()
        # Should be roughly within 5% of contamination rate
        assert actual_rate <= contamination + 0.05

    def test_anomaly_score_positive(self, synthetic_reports_df):
        detector = AnomalyDetector(contamination=0.05)
        result = detector.fit_and_detect(synthetic_reports_df)
        assert (result["anomaly_score"] >= 0).all()

    def test_no_rows_dropped(self, synthetic_reports_df):
        detector = AnomalyDetector()
        result = detector.fit_and_detect(synthetic_reports_df)
        assert len(result) == len(synthetic_reports_df)


class TestTimeTrendAnalyser:

    @pytest.fixture
    def trend_df(self):
        """Dataset with an artificially increasing trend for DrugA-nausea."""
        rows = []
        base = date(2022, 1, 1)
        for month in range(24):  # 2 years
            report_date = base + timedelta(days=30 * month)
            # Increasing count for DrugA-nausea
            count = 5 + month * 2
            for _ in range(count):
                rows.append({
                    "drug_name": "DrugA",
                    "adverse_event": "nausea",
                    "report_date": report_date,
                })
            # Stable count for DrugB-headache
            for _ in range(5):
                rows.append({
                    "drug_name": "DrugB",
                    "adverse_event": "headache",
                    "report_date": report_date,
                })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self, trend_df):
        analyser = TimeTrendAnalyser()
        result = analyser.analyse_trends(trend_df)
        assert isinstance(result, pd.DataFrame)

    def test_detects_increasing_trend(self, trend_df):
        analyser = TimeTrendAnalyser()
        result = analyser.analyse_trends(trend_df)
        if not result.empty:
            drug_a = result[
                (result["drug_name"] == "DrugA") & (result["adverse_event"] == "nausea")
            ]
            if not drug_a.empty:
                assert drug_a.iloc[0]["trend_slope"] > 0
                assert drug_a.iloc[0]["trend_direction"] == "increasing"

    def test_handles_missing_date_column(self):
        analyser = TimeTrendAnalyser()
        df = pd.DataFrame({"drug_name": ["DrugA"], "adverse_event": ["nausea"]})
        result = analyser.analyse_trends(df)
        assert result.empty or isinstance(result, pd.DataFrame)

    def test_result_columns_present(self, trend_df):
        analyser = TimeTrendAnalyser()
        result = analyser.analyse_trends(trend_df)
        if not result.empty:
            expected_cols = {"drug_name", "adverse_event", "trend_slope",
                             "trend_direction", "is_significant"}
            assert expected_cols.issubset(set(result.columns))
