"""
Risk Signal Detection Algorithms
AI Pharmacovigilance Intelligence Platform

Implements industry-standard pharmacovigilance signal detection methods:

  1. Proportional Reporting Ratio (PRR) — EMA standard
  2. Reporting Odds Ratio (ROR) — FDA preferred
  3. Information Component (IC) / EB05 — WHO VigiBase method
  4. Chi-Square test for statistical significance
  5. Isolation Forest anomaly detection (ML-based)
  6. Time-trend analysis for emerging signals

References
----------
- ICH E2B(R3) Clinical Safety Data Management
- EMA Guideline on the Use of Statistical Signal Detection Methods
- WHO Programme for International Drug Monitoring
"""

from __future__ import annotations

import math
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """Detected pharmacovigilance safety signal."""
    signal_id: str
    drug_name: str
    adverse_event: str
    signal_type: str
    # Disproportionality measures
    prr: Optional[float] = None
    prr_lower_ci: Optional[float] = None
    prr_upper_ci: Optional[float] = None
    ror: Optional[float] = None
    ror_lower_ci: Optional[float] = None
    ror_upper_ci: Optional[float] = None
    ic: Optional[float] = None
    eb05: Optional[float] = None
    chi_square: Optional[float] = None
    p_value: Optional[float] = None
    # Counts
    report_count: int = 0
    expected_count: Optional[float] = None
    # Assessment
    severity_score: float = 0.0
    is_new: bool = True
    detection_date: date = field(default_factory=date.today)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Disproportionality analysis
# ---------------------------------------------------------------------------

class DisproportionalityAnalyser:
    """
    Computes standard pharmacovigilance disproportionality statistics.

    The 2x2 contingency table used:

                    | Drug X | All other drugs | Total
    ----------------+--------+-----------------+------
    Event E         |   a    |       b         | a+b
    All other events|   c    |       d         | c+d
    ----------------+--------+-----------------+------
    Total           |  a+c   |      b+d        |  N
    """

    def __init__(
        self,
        prr_threshold: float = 2.0,
        ror_threshold: float = 2.0,
        min_reports: int = 3,
        chi2_alpha: float = 0.05,
    ) -> None:
        self.prr_threshold = prr_threshold
        self.ror_threshold = ror_threshold
        self.min_reports = min_reports
        self.chi2_alpha = chi2_alpha

    def analyse(self, df: pd.DataFrame) -> List[SignalResult]:
        """
        Run full disproportionality analysis on a reports DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: drug_name, adverse_event

        Returns
        -------
        List[SignalResult]
            All detected signals meeting threshold criteria.
        """
        signals: List[SignalResult] = []
        N = len(df)
        if N < 10:
            logger.warning("Insufficient data for signal detection (n={})", N)
            return signals

        # Count drug-event co-occurrences
        drug_event_counts = (
            df.groupby(["drug_name", "adverse_event"])
            .size()
            .reset_index()
        )
        drug_event_counts.columns = ["drug_name", "adverse_event", "a"]

        # Total reports per drug
        drug_totals = df.groupby("drug_name").size().to_frame("drug_total")
        # Total reports per event
        event_totals = df.groupby("adverse_event").size().to_frame("event_total")

        drug_event_counts = drug_event_counts.join(drug_totals, on="drug_name")
        drug_event_counts = drug_event_counts.join(event_totals, on="adverse_event")

        for _, row in drug_event_counts.iterrows():
            a = int(row["a"])
            if a < self.min_reports:
                continue

            drug_total = int(row["drug_total"])
            event_total = int(row["event_total"])

            # 2×2 table cells
            b = event_total - a               # Event in other drugs
            c = drug_total - a               # Other events in this drug
            d = N - a - b - c                # Other events in other drugs

            if b < 0 or c < 0 or d < 0:
                continue

            result = self._compute_measures(
                drug=str(row["drug_name"]),
                event=str(row["adverse_event"]),
                a=a, b=b, c=c, d=d, N=N,
            )

            if result and self._meets_threshold(result):
                signals.append(result)

        logger.info("Disproportionality analysis: {} signals detected from {} drug-event pairs.",
                    len(signals), len(drug_event_counts))
        return signals

    def _compute_measures(
        self, drug: str, event: str,
        a: int, b: int, c: int, d: int, N: int
    ) -> Optional[SignalResult]:
        """Compute PRR, ROR, IC, and chi-square for a drug-event pair."""
        signal_id = f"SIG-{uuid.uuid4().hex[:10].upper()}"

        # --- PRR ---
        try:
            if (a + c) == 0 or (b + d) == 0 or b == 0:
                prr = None
                prr_lo = prr_hi = None
            else:
                rate_drug = a / (a + c)
                rate_other = b / (b + d)
                if rate_other == 0:
                    prr = None
                    prr_lo = prr_hi = None
                else:
                    prr = rate_drug / rate_other
                    # 95% CI using log-normal approximation
                    se_log_prr = math.sqrt(1/a - 1/(a+c) + 1/b - 1/(b+d))
                    prr_lo = math.exp(math.log(prr) - 1.96 * se_log_prr)
                    prr_hi = math.exp(math.log(prr) + 1.96 * se_log_prr)
        except (ValueError, ZeroDivisionError):
            prr = prr_lo = prr_hi = None

        # --- ROR ---
        try:
            if b == 0 or c == 0:
                ror = None
                ror_lo = ror_hi = None
            else:
                ror = (a * d) / (b * c)
                se_log_ror = math.sqrt(1/a + 1/b + 1/c + 1/d)
                ror_lo = math.exp(math.log(ror) - 1.96 * se_log_ror)
                ror_hi = math.exp(math.log(ror) + 1.96 * se_log_ror)
        except (ValueError, ZeroDivisionError):
            ror = ror_lo = ror_hi = None

        # --- Information Component (WHO Bayesian method) ---
        try:
            expected = (a + b) * (a + c) / N
            if expected > 0:
                ic = math.log2((a + 0.5) / (expected + 0.5))
                # EB05: 5th percentile of Bayesian credibility interval
                alpha_post = a + 0.5
                beta_post = expected + 0.5
                eb05 = math.log2(alpha_post / beta_post) - 1.96 * math.sqrt(
                    1 / (alpha_post) + 1 / (beta_post)
                )
            else:
                ic = eb05 = None
                expected = None
        except (ValueError, ZeroDivisionError):
            ic = eb05 = expected = None

        # --- Chi-Square ---
        try:
            contingency = np.array([[a, b], [c, d]], dtype=float)
            # Add small constant to avoid zero-cell issues
            contingency = np.maximum(contingency, 0.5)
            chi2, p_val, _, _ = stats.chi2_contingency(contingency, correction=True)
        except Exception:
            chi2 = p_val = None

        # Severity scoring (composite)
        severity_score = self._compute_severity_score(
            prr=prr, ror=ror, ic=ic, a=a, p_value=p_val
        )

        return SignalResult(
            signal_id=signal_id,
            drug_name=drug,
            adverse_event=event,
            signal_type="disproportionality",
            prr=round(prr, 4) if prr is not None else None,
            prr_lower_ci=round(prr_lo, 4) if prr_lo is not None else None,
            prr_upper_ci=round(prr_hi, 4) if prr_hi is not None else None,
            ror=round(ror, 4) if ror is not None else None,
            ror_lower_ci=round(ror_lo, 4) if ror_lo is not None else None,
            ror_upper_ci=round(ror_hi, 4) if ror_hi is not None else None,
            ic=round(ic, 4) if ic is not None else None,
            eb05=round(eb05, 4) if eb05 is not None else None,
            chi_square=round(chi2, 4) if chi2 is not None else None,
            p_value=round(p_val, 6) if p_val is not None else None,
            report_count=a,
            expected_count=round(expected, 4) if expected is not None else None,
            severity_score=severity_score,
            metadata={"a": a, "b": b, "c": c, "d": d, "N": N},
        )

    def _meets_threshold(self, result: SignalResult) -> bool:
        """Return True if signal meets any threshold criterion."""
        prr_flag = result.prr is not None and result.prr >= self.prr_threshold and \
                   result.prr_lower_ci is not None and result.prr_lower_ci >= 1.0
        ror_flag = result.ror is not None and result.ror >= self.ror_threshold and \
                   result.ror_lower_ci is not None and result.ror_lower_ci >= 1.0
        ic_flag = result.eb05 is not None and result.eb05 > 0
        chi2_flag = result.p_value is not None and result.p_value < self.chi2_alpha
        return any([prr_flag, ror_flag, ic_flag, chi2_flag])

    @staticmethod
    def _compute_severity_score(
        prr: Optional[float],
        ror: Optional[float],
        ic: Optional[float],
        a: int,
        p_value: Optional[float],
    ) -> float:
        """Composite severity score for signal prioritisation (0-100)."""
        score = 0.0
        if prr is not None:
            score += min(prr / 10.0, 1.0) * 30
        if ror is not None:
            score += min(ror / 10.0, 1.0) * 25
        if ic is not None:
            score += min(max(ic, 0) / 5.0, 1.0) * 20
        if a > 0:
            score += min(math.log10(a + 1) / 3.0, 1.0) * 15
        if p_value is not None:
            score += (1 - min(p_value / 0.05, 1.0)) * 10
        return round(min(score, 100.0), 2)


# ---------------------------------------------------------------------------
# Anomaly Detection (ML-based)
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    ML-based anomaly detection for unusual adverse event reporting patterns.
    Uses Isolation Forest for unsupervised detection.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        self._fitted = False

    def fit_and_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Isolation Forest on report features and return anomaly labels.

        Returns
        -------
        pd.DataFrame
            Input df with added 'anomaly_score' and 'is_anomaly' columns.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder

        df = df.copy()
        feature_cols = []

        # Build numeric feature matrix
        if "patient_age" in df.columns:
            df["_age"] = pd.to_numeric(df["patient_age"], errors="coerce").fillna(50)
            feature_cols.append("_age")

        if "severity" in df.columns:
            severity_order = {"mild": 1, "moderate": 2, "severe": 3, "life_threatening": 4, "fatal": 5, "unknown": 0}
            df["_severity_num"] = df["severity"].map(severity_order).fillna(0)
            feature_cols.append("_severity_num")

        if "report_date" in df.columns:
            df["_date_ordinal"] = pd.to_datetime(df["report_date"], errors="coerce").map(
                lambda x: x.toordinal() if pd.notna(x) else 0
            )
            feature_cols.append("_date_ordinal")

        # Add Drug and Event encoding (New)
        if "drug_name" in df.columns:
            le_drug = LabelEncoder()
            df["_drug_enc"] = le_drug.fit_transform(df["drug_name"].astype(str))
            feature_cols.append("_drug_enc")

        if "adverse_event" in df.columns:
            le_event = LabelEncoder()
            df["_event_enc"] = le_event.fit_transform(df["adverse_event"].astype(str))
            feature_cols.append("_event_enc")

        if not feature_cols:
            logger.warning("No numeric features available for anomaly detection.")
            df["anomaly_score"] = 0.0
            df["is_anomaly"] = False
            return df

        X = df[feature_cols].values

        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        preds = self._model.fit_predict(X)
        scores = self._model.score_samples(X)

        df["anomaly_score"] = np.round(-scores, 4)  # Higher = more anomalous
        df["is_anomaly"] = preds == -1

        n_anomalies = (preds == -1).sum()
        logger.info("Anomaly detection: {} anomalies detected ({:.1f}%)",
                    n_anomalies, n_anomalies / len(df) * 100)

        # Clean up temp columns
        for col in feature_cols:
            df.drop(columns=[col], inplace=True, errors="ignore")

        return df


# ---------------------------------------------------------------------------
# Time-Trend Analyser
# ---------------------------------------------------------------------------

class TimeTrendAnalyser:
    """
    Detects emerging safety signals by analysing temporal trends
    in adverse event reporting rates.
    """

    def analyse_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect drug-event pairs with significantly increasing reporting trends.

        Returns DataFrame with trend statistics per drug-event pair.
        """
        if "report_date" not in df.columns or "drug_name" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.dropna(subset=["report_date"])
        df["month"] = df["report_date"].dt.to_period("M")

        results = []
        pairs = (
            df.groupby(["drug_name", "adverse_event"])
            .size()
            .reset_index(name="total_count")
            .query("total_count >= 5")
        )

        for _, pair_row in pairs.iterrows():
            drug = pair_row["drug_name"]
            event = pair_row["adverse_event"]

            subset = df[(df["drug_name"] == drug) & (df["adverse_event"] == event)].copy()
            monthly = subset.groupby("month").size().reset_index(name="count")

            if len(monthly) < 3:
                continue

            # Convert period to numeric index for regression
            monthly["month_idx"] = range(len(monthly))
            x = monthly["month_idx"].values
            y = monthly["count"].values

            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                results.append({
                    "drug_name": drug,
                    "adverse_event": event,
                    "total_count": int(pair_row["total_count"]),
                    "trend_slope": round(float(slope), 4),
                    "trend_r_squared": round(float(r_value ** 2), 4),
                    "trend_p_value": round(float(p_value), 6),
                    "trend_direction": trend_direction,
                    "is_significant": p_value < 0.05 and slope > 0,
                    "monthly_data": monthly[["month_idx", "count"]].to_dict("records"),
                })
            except Exception:
                continue

        trend_df = pd.DataFrame(results)
        if not trend_df.empty:
            trend_df = trend_df.sort_values("trend_slope", ascending=False)
        logger.info("Time-trend analysis: {} drug-event pairs analysed.", len(trend_df))
        return trend_df
