"""
Unit Tests — Data Cleaning Service
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.data_ingestion.data_cleaner import DataCleaner


@pytest.fixture
def cleaner():
    return DataCleaner()


@pytest.fixture
def minimal_df():
    return pd.DataFrame({
        "drug_name": ["Metformin", "Warfarin", "Ibuprofen"],
        "adverse_event": ["nausea", "bleeding", "headache"],
        "report_date": ["2023-01-15", "2023-06-20", "2023-09-05"],
    })


@pytest.fixture
def full_df():
    return pd.DataFrame({
        "drug_name": ["Metformin", "Warfarin", "Ibuprofen", "Atorvastatin", None],
        "adverse_event": ["nausea", "bleeding", "headache", "myalgia", "dizziness"],
        "report_date": ["2023-01-15", "2023-06-20", "2023-09-05", "2024-01-10", "2024-02-01"],
        "severity": ["mild", "SEVERE", "moderate", "life threatening", "unknown"],
        "gender": ["male", "Female", "M", "f", "unknown"],
        "outcome": ["recovered", "not recovered", "recovery", "fatal", ""],
        "clinical_phase": ["phase 1", "Phase 3", "post market", "phase_4", "unknown"],
        "patient_age": [45, 72, -5, 85, 200],
        "country": ["united states", "germany", "UK", "Japan", None],
        "is_serious": [False, True, False, True, False],
    })


class TestDataCleaner:

    def test_minimal_df_passes(self, cleaner, minimal_df):
        result = cleaner.clean(minimal_df)
        assert len(result) == 3
        assert "drug_name" in result.columns
        assert "adverse_event" in result.columns

    def test_missing_required_columns_raises(self, cleaner):
        df = pd.DataFrame({"drug_name": ["Metformin"]})
        with pytest.raises(ValueError, match="Required columns missing"):
            cleaner.clean(df)

    def test_severity_normalisation(self, cleaner, full_df):
        result = cleaner.clean(full_df.dropna(subset=["drug_name"]))
        assert all(v in ("mild", "moderate", "severe", "life_threatening", "fatal", "unknown")
                   for v in result["severity"])

    def test_gender_normalisation(self, cleaner, full_df):
        result = cleaner.clean(full_df.dropna(subset=["drug_name"]))
        assert all(v in ("male", "female", "other", "unknown")
                   for v in result["gender"])

    def test_outcome_normalisation(self, cleaner, full_df):
        result = cleaner.clean(full_df.dropna(subset=["drug_name"]))
        assert all(v in ("recovered", "recovering", "not_recovered", "fatal", "unknown")
                   for v in result["outcome"])

    def test_clinical_phase_normalisation(self, cleaner, full_df):
        result = cleaner.clean(full_df.dropna(subset=["drug_name"]))
        assert all(v in ("phase_1", "phase_2", "phase_3", "phase_4", "post_market", "unknown")
                   for v in result["clinical_phase"])

    def test_age_winsorising(self, cleaner, full_df):
        result = cleaner.clean(full_df.dropna(subset=["drug_name"]))
        # Ages -5 and 200 should be NaN
        ages = result["patient_age"].dropna()
        assert all(0 <= a <= 120 for a in ages)

    def test_null_drug_name_dropped(self, cleaner, full_df):
        result = cleaner.clean(full_df)
        # Row with None drug_name should be dropped
        assert result["drug_name"].isna().sum() == 0

    def test_report_id_generated(self, cleaner, minimal_df):
        result = cleaner.clean(minimal_df)
        assert "report_id" in result.columns
        assert result["report_id"].notna().all()
        assert result["report_id"].str.startswith("PVR-").all()

    def test_age_group_added(self, cleaner, minimal_df):
        minimal_df["patient_age"] = [35, 68, 80]
        result = cleaner.clean(minimal_df)
        assert "patient_age_group" in result.columns

    def test_is_serious_flag(self, cleaner, minimal_df):
        minimal_df["severity"] = ["fatal", "mild", "severe"]
        result = cleaner.clean(minimal_df)
        assert result.loc[result["severity"] == "fatal", "is_serious"].all()

    def test_duplicate_detection(self, cleaner, minimal_df):
        # Add a true duplicate
        df_with_dup = pd.concat([minimal_df, minimal_df.iloc[[0]]], ignore_index=True)
        df_with_dup["patient_age"] = [45, 72, 55, 45]
        result = cleaner.clean(df_with_dup)
        assert "is_duplicate" in result.columns
        assert result["is_duplicate"].sum() >= 1

    def test_date_parsing(self, cleaner, minimal_df):
        result = cleaner.clean(minimal_df)
        assert pd.api.types.is_datetime64_any_dtype(result["report_date"])

    def test_column_standardisation(self, cleaner):
        df = pd.DataFrame({
            "Drug Name": ["Metformin"],
            "Adverse Event": ["nausea"],
            "Report Date": ["2023-01-01"],
        })
        result = cleaner.clean(df)
        assert "drug_name" in result.columns
        assert "adverse_event" in result.columns
        assert "report_date" in result.columns
