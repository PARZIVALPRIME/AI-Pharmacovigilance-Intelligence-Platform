"""
Risk Signal Detection Service
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from database.connection import SessionLocal
from database.models import AdverseEventReport, RiskSignal, SignalStatus, AuditLog
from services.risk_detection.signal_algorithms import (
    DisproportionalityAnalyser,
    AnomalyDetector,
    TimeTrendAnalyser,
    SignalResult,
)


class RiskSignalDetectionService:
    """
    Orchestrates all risk signal detection algorithms against the database.

    Usage
    -----
    service = RiskSignalDetectionService()
    results = service.run_full_detection()
    signals = service.get_active_signals()
    """

    def __init__(
        self,
        prr_threshold: float = 2.0,
        min_reports: int = 3,
        contamination: float = 0.05,
    ) -> None:
        self.disproportionality = DisproportionalityAnalyser(
            prr_threshold=prr_threshold,
            min_reports=min_reports,
        )
        self.anomaly_detector = AnomalyDetector(contamination=contamination)
        self.trend_analyser = TimeTrendAnalyser()
        logger.info("RiskSignalDetectionService initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_detection(self) -> dict:
        """Run all signal detection algorithms and persist results."""
        start = time.perf_counter()
        logger.info("Starting risk signal detection pipeline…")

        df = self._load_reports_df()
        if df.empty:
            logger.warning("No reports available for signal detection.")
            return {"status": "no_data", "signals_detected": 0}

        # 1. Disproportionality
        disp_signals = self.disproportionality.analyse(df)
        saved = self._persist_signals(disp_signals)

        # 2. Anomaly detection
        anomaly_df = self.anomaly_detector.fit_and_detect(df)
        anomaly_count = int(anomaly_df.get("is_anomaly", pd.Series([False])).sum()) if not anomaly_df.empty else 0

        # 3. Time-trend analysis
        trend_df = self.trend_analyser.analyse_trends(df)
        significant_trends = len(trend_df[trend_df.get("is_significant", False)]) if not trend_df.empty else 0

        elapsed = time.perf_counter() - start
        summary = {
            "status": "success",
            "disproportionality_signals": len(disp_signals),
            "signals_saved": saved,
            "anomalies_detected": anomaly_count,
            "significant_trends": significant_trends,
            "total_reports_analysed": len(df),
            "elapsed_seconds": round(elapsed, 2),
        }

        self._write_audit_log(summary)
        logger.info("Risk detection complete: {}", summary)
        return summary

    def get_active_signals(self, limit: int = 100) -> List[dict]:
        """Retrieve active risk signals from the database."""
        with SessionLocal() as session:
            signals = (
                session.query(RiskSignal)
                .filter(RiskSignal.status.in_([SignalStatus.DETECTED, SignalStatus.UNDER_REVIEW]))
                .order_by(RiskSignal.severity_score.desc())
                .limit(limit)
                .all()
            )
            return [self._signal_to_dict(s) for s in signals]

    def get_signals_for_drug(self, drug_name: str) -> List[dict]:
        """Get all signals for a specific drug."""
        with SessionLocal() as session:
            signals = (
                session.query(RiskSignal)
                .filter(RiskSignal.drug_name.ilike(f"%{drug_name}%"))
                .order_by(RiskSignal.severity_score.desc())
                .all()
            )
            return [self._signal_to_dict(s) for s in signals]

    def get_summary_stats(self) -> dict:
        """Return summary statistics for risk signals."""
        with SessionLocal() as session:
            total = session.query(RiskSignal).count()
            detected = session.query(RiskSignal).filter(RiskSignal.status == SignalStatus.DETECTED).count()
            confirmed = session.query(RiskSignal).filter(RiskSignal.status == SignalStatus.CONFIRMED).count()
            high_severity = session.query(RiskSignal).filter(RiskSignal.severity_score >= 70).count()

        return {
            "total_signals": total,
            "new_signals": detected,
            "confirmed_signals": confirmed,
            "high_severity_signals": high_severity,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_reports_df() -> pd.DataFrame:
        """Load adverse event reports from database as DataFrame."""
        with SessionLocal() as session:
            reports = session.query(
                AdverseEventReport.drug_name,
                AdverseEventReport.adverse_event,
                AdverseEventReport.severity,
                AdverseEventReport.patient_age,
                AdverseEventReport.report_date,
                AdverseEventReport.country,
                AdverseEventReport.is_serious,
            ).filter(
                AdverseEventReport.is_duplicate == False
            ).all()

        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "drug_name": r.drug_name,
                "adverse_event": r.adverse_event,
                "severity": r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                "patient_age": r.patient_age,
                "report_date": r.report_date,
                "country": r.country,
                "is_serious": r.is_serious,
            }
            for r in reports
        ])
        return df

    def _persist_signals(self, signals: List[SignalResult]) -> int:
        """Persist detected signals to the database, avoiding duplicates."""
        saved = 0
        with SessionLocal() as session:
            for sig in signals:
                # Check for existing signal for same drug-event
                existing = session.query(RiskSignal).filter(
                    RiskSignal.drug_name == sig.drug_name,
                    RiskSignal.adverse_event == sig.adverse_event,
                ).first()

                if existing:
                    # Update metrics
                    existing.prr = sig.prr
                    existing.ror = sig.ror
                    existing.ic = sig.ic
                    existing.eb05 = sig.eb05
                    existing.chi_square = sig.chi_square
                    existing.p_value = sig.p_value
                    existing.report_count = sig.report_count
                    existing.expected_count = sig.expected_count
                    existing.severity_score = sig.severity_score
                    existing.is_new = False
                else:
                    db_signal = RiskSignal(
                        signal_id=sig.signal_id,
                        drug_name=sig.drug_name,
                        adverse_event=sig.adverse_event,
                        signal_type=sig.signal_type,
                        prr=sig.prr,
                        prr_lower_ci=sig.prr_lower_ci,
                        prr_upper_ci=sig.prr_upper_ci,
                        ror=sig.ror,
                        ror_lower_ci=sig.ror_lower_ci,
                        ror_upper_ci=sig.ror_upper_ci,
                        ic=sig.ic,
                        eb05=sig.eb05,
                        chi_square=sig.chi_square,
                        p_value=sig.p_value,
                        report_count=sig.report_count,
                        expected_count=sig.expected_count,
                        detection_date=sig.detection_date,
                        status=SignalStatus.DETECTED,
                        severity_score=sig.severity_score,
                        is_new=True,
                        metadata_json=sig.metadata,
                    )
                    session.add(db_signal)
                    saved += 1

            session.commit()
        return saved

    @staticmethod
    def _signal_to_dict(signal: RiskSignal) -> dict:
        return {
            "id": signal.id,
            "signal_id": signal.signal_id,
            "drug_name": signal.drug_name,
            "adverse_event": signal.adverse_event,
            "signal_type": signal.signal_type,
            "prr": signal.prr,
            "ror": signal.ror,
            "ic": signal.ic,
            "eb05": signal.eb05,
            "p_value": signal.p_value,
            "report_count": signal.report_count,
            "expected_count": signal.expected_count,
            "severity_score": signal.severity_score,
            "status": signal.status.value if hasattr(signal.status, "value") else str(signal.status),
            "detection_date": str(signal.detection_date) if signal.detection_date else None,
            "is_new": signal.is_new,
        }

    @staticmethod
    def _write_audit_log(summary: dict) -> None:
        try:
            with SessionLocal() as session:
                audit = AuditLog(
                    action="risk_signal_detection",
                    entity_type="RiskSignal",
                    details=summary,
                    success=True,
                )
                session.add(audit)
                session.commit()
        except Exception as exc:
            logger.warning("Audit log write failed: {}", exc)
