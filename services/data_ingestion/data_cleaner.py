"""
Data Cleaning & Normalisation Pipeline
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Normalisation maps
# ---------------------------------------------------------------------------

SEVERITY_MAP = {
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "serious": "severe",
    "life threatening": "life_threatening",
    "life-threatening": "life_threatening",
    "life_threatening": "life_threatening",
    "fatal": "fatal",
    "death": "fatal",
    "unknown": "unknown",
    "not specified": "unknown",
    "": "unknown",
}

GENDER_MAP = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "other": "other",
    "unknown": "unknown",
    "not specified": "unknown",
    "": "unknown",
}

OUTCOME_MAP = {
    "recovered": "recovered",
    "recovery": "recovered",
    "resolved": "recovered",
    "recovering": "recovering",
    "improving": "recovering",
    "not recovered": "not_recovered",
    "not_recovered": "not_recovered",
    "fatal": "fatal",
    "death": "fatal",
    "died": "fatal",
    "unknown": "unknown",
    "not specified": "unknown",
    "": "unknown",
}

CLINICAL_PHASE_MAP = {
    "phase 1": "phase_1",
    "phase_1": "phase_1",
    "phase i": "phase_1",
    "phase 2": "phase_2",
    "phase_2": "phase_2",
    "phase ii": "phase_2",
    "phase 3": "phase_3",
    "phase_3": "phase_3",
    "phase iii": "phase_3",
    "phase 4": "phase_4",
    "phase_4": "phase_4",
    "phase iv": "phase_4",
    "post market": "post_market",
    "post_market": "post_market",
    "post-market": "post_market",
    "post marketing": "post_market",
    "unknown": "unknown",
    "": "unknown",
}


# ---------------------------------------------------------------------------
# Cleaner class
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Production-grade data cleaning pipeline for pharmacovigilance data.

    Handles:
    - Missing value imputation
    - Field normalisation (severity, gender, outcome, etc.)
    - Duplicate detection
    - Data type coercion
    - Basic outlier detection
    """

    # Columns that must be present
    REQUIRED_COLUMNS = {"drug_name", "adverse_event", "report_date"}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on a raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Cleaned and normalised DataFrame.
        """
        df = df.copy()

        # 1. Column name standardisation
        df = self._standardise_columns(df)

        # 2. Validate required columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        # 3. Field-level normalisation
        df = self._normalise_text_fields(df)
        df = self._normalise_severity(df)
        df = self._normalise_gender(df)
        df = self._normalise_outcome(df)
        df = self._normalise_clinical_phase(df)

        # 4. Date parsing
        df = self._parse_dates(df)

        # 5. Age handling
        df = self._process_age(df)

        # 6. Seriousness flag
        df = self._compute_seriousness(df)

        # 7. Duplicate marking
        df = self._mark_duplicates(df)

        # 8. Age group
        df = self._add_age_group(df)

        # 9. Report ID generation (if missing)
        df = self._ensure_report_id(df)

        # 10. Drop rows with critical null fields
        initial_len = len(df)
        df = df.dropna(subset=["drug_name", "adverse_event"])
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"[Cleaner] Dropped {dropped} rows with null drug_name or adverse_event.")

        # 11. Final type conversions
        df = self._final_types(df)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case, strip, and snake_case column names."""
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True)
            .str.replace(r"[^\w_]", "", regex=True)
        )
        return df

    @staticmethod
    def _normalise_text_fields(df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and normalise case for key text fields."""
        text_cols = ["drug_name", "adverse_event", "country", "drug_class", "source_type"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                if col in ("country", "drug_class", "source_type"):
                    df[col] = df[col].str.title()
                elif col in ("drug_name",):
                    # Preserve original casing for drug names
                    pass
                else:
                    df[col] = df[col].str.lower()
        return df

    @staticmethod
    def _normalise_severity(df: pd.DataFrame) -> pd.DataFrame:
        if "severity" not in df.columns:
            df["severity"] = "unknown"
            return df
        df["severity"] = (
            df["severity"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: SEVERITY_MAP.get(x, "unknown"))
        )
        return df

    @staticmethod
    def _normalise_gender(df: pd.DataFrame) -> pd.DataFrame:
        if "gender" not in df.columns:
            df["gender"] = "unknown"
            return df
        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: GENDER_MAP.get(x, "unknown"))
        )
        return df

    @staticmethod
    def _normalise_outcome(df: pd.DataFrame) -> pd.DataFrame:
        if "outcome" not in df.columns:
            df["outcome"] = "unknown"
            return df
        df["outcome"] = (
            df["outcome"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: OUTCOME_MAP.get(x, "unknown"))
        )
        return df

    @staticmethod
    def _normalise_clinical_phase(df: pd.DataFrame) -> pd.DataFrame:
        if "clinical_phase" not in df.columns:
            df["clinical_phase"] = "unknown"
            return df
        df["clinical_phase"] = (
            df["clinical_phase"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: CLINICAL_PHASE_MAP.get(x, "unknown"))
        )
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        for date_col in ("report_date", "receipt_date"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        return df

    @staticmethod
    def _process_age(df: pd.DataFrame) -> pd.DataFrame:
        if "patient_age" not in df.columns:
            df["patient_age"] = np.nan
            return df
        df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
        # Winsorise implausible ages
        df.loc[df["patient_age"] < 0, "patient_age"] = np.nan
        df.loc[df["patient_age"] > 120, "patient_age"] = np.nan
        return df

    @staticmethod
    def _compute_seriousness(df: pd.DataFrame) -> pd.DataFrame:
        if "is_serious" not in df.columns:
            serious_severities = {"severe", "life_threatening", "fatal"}
            df["is_serious"] = df.get("severity", pd.Series()).isin(serious_severities)
        else:
            df["is_serious"] = df["is_serious"].astype(bool)
        return df

    @staticmethod
    def _mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Mark likely duplicate reports based on key fields."""
        key_cols = [c for c in ["report_id", "drug_name", "adverse_event", "report_date", "patient_age"]
                    if c in df.columns]
        if not key_cols:
            df["is_duplicate"] = False
            return df

        dup_mask = df.duplicated(subset=key_cols, keep="first")
        df["is_duplicate"] = dup_mask
        n_dups = dup_mask.sum()
        if n_dups > 0:
            print(f"[Cleaner] Flagged {n_dups} potential duplicate records.")
        return df

    @staticmethod
    def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
        if "patient_age_group" in df.columns and df["patient_age_group"].notna().any():
            return df  # Already set

        if "patient_age" not in df.columns:
            df["patient_age_group"] = "unknown"
            return df

        bins = [0, 17, 44, 64, 74, 120]
        labels = ["paediatric", "young_adult", "middle_aged", "elderly", "very_elderly"]
        df["patient_age_group"] = pd.cut(
            df["patient_age"], bins=bins, labels=labels, right=True
        ).astype(str)
        df["patient_age_group"] = df["patient_age_group"].replace("nan", "unknown")
        return df

    @staticmethod
    def _ensure_report_id(df: pd.DataFrame) -> pd.DataFrame:
        import uuid
        if "report_id" not in df.columns:
            df["report_id"] = [f"PVR-{uuid.uuid4().hex[:12].upper()}" for _ in range(len(df))]
        else:
            mask = df["report_id"].isna() | (df["report_id"] == "")
            df.loc[mask, "report_id"] = [
                f"PVR-{uuid.uuid4().hex[:12].upper()}" for _ in range(mask.sum())
            ]
        return df

    @staticmethod
    def _final_types(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct dtypes on final output."""
        bool_cols = ["is_serious", "is_duplicate"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        if "confidence_score" in df.columns:
            df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        return df
