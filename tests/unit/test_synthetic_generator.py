"""
Unit Tests — Synthetic Dataset Generator
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.data_ingestion.synthetic_generator import generate_synthetic_dataset


class TestSyntheticGenerator:

    def test_generates_correct_count(self):
        df = generate_synthetic_dataset(n_records=500, seed=1)
        assert len(df) == 500

    def test_required_columns_present(self):
        df = generate_synthetic_dataset(n_records=100, seed=2)
        required = {
            "report_id", "drug_name", "adverse_event", "severity",
            "patient_age", "gender", "country", "report_date",
            "clinical_phase", "drug_class", "outcome", "is_serious",
        }
        assert required.issubset(set(df.columns))

    def test_no_null_drug_names(self):
        df = generate_synthetic_dataset(n_records=200, seed=3)
        assert df["drug_name"].isna().sum() == 0

    def test_no_null_adverse_events(self):
        df = generate_synthetic_dataset(n_records=200, seed=4)
        assert df["adverse_event"].isna().sum() == 0

    def test_severity_values_valid(self):
        df = generate_synthetic_dataset(n_records=200, seed=5)
        valid = {"mild", "moderate", "severe", "life_threatening", "fatal"}
        assert set(df["severity"].unique()).issubset(valid)

    def test_gender_values_valid(self):
        df = generate_synthetic_dataset(n_records=200, seed=6)
        valid = {"male", "female", "other", "unknown"}
        assert set(df["gender"].unique()).issubset(valid)

    def test_ages_in_plausible_range(self):
        df = generate_synthetic_dataset(n_records=300, seed=7)
        assert df["patient_age"].min() >= 18
        assert df["patient_age"].max() <= 95

    def test_report_ids_unique(self):
        df = generate_synthetic_dataset(n_records=500, seed=8)
        assert df["report_id"].nunique() == 500

    def test_report_ids_have_correct_prefix(self):
        df = generate_synthetic_dataset(n_records=100, seed=9)
        assert df["report_id"].str.startswith("PVR-").all()

    def test_is_serious_is_boolean(self):
        df = generate_synthetic_dataset(n_records=100, seed=10)
        assert df["is_serious"].dtype == bool

    def test_seriousness_correlates_with_severity(self):
        df = generate_synthetic_dataset(n_records=1000, seed=11)
        fatal_serious = df[df["severity"] == "fatal"]["is_serious"].all()
        assert fatal_serious  # All fatal cases should be serious

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(n_records=100, seed=42)
        df2 = generate_synthetic_dataset(n_records=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_dataset(n_records=100, seed=1)
        df2 = generate_synthetic_dataset(n_records=100, seed=2)
        assert not df1["report_id"].equals(df2["report_id"])

    def test_multiple_drugs_present(self):
        df = generate_synthetic_dataset(n_records=500, seed=13)
        assert df["drug_name"].nunique() > 5

    def test_multiple_countries_present(self):
        df = generate_synthetic_dataset(n_records=500, seed=14)
        assert df["country"].nunique() > 3

    def test_source_text_not_empty(self):
        df = generate_synthetic_dataset(n_records=100, seed=15)
        assert df["source_text"].notna().all()
        assert (df["source_text"].str.len() > 10).all()

    def test_derived_columns_present(self):
        df = generate_synthetic_dataset(n_records=100, seed=16)
        assert "report_year" in df.columns
        assert "report_quarter" in df.columns
        assert "report_month" in df.columns

    def test_large_dataset_performance(self):
        import time
        t0 = time.perf_counter()
        df = generate_synthetic_dataset(n_records=10_000, seed=99)
        elapsed = time.perf_counter() - t0
        assert len(df) == 10_000
        assert elapsed < 30  # Should complete within 30 seconds

    def test_clinical_phase_values_valid(self):
        df = generate_synthetic_dataset(n_records=200, seed=17)
        valid = {"phase_1", "phase_2", "phase_3", "phase_4", "post_market"}
        assert set(df["clinical_phase"].unique()).issubset(valid)
