"""
Compliance Service
AI Pharmacovigilance Intelligence Platform

Orchestrates compliance monitoring and metrics reporting.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, List
from .metrics_engine import ComplianceMetricsEngine

class ComplianceService:
    """Service for managing platform compliance and KPIs."""

    def __init__(self) -> None:
        self.engine = ComplianceMetricsEngine()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Return full compliance dashboard payload."""
        sub_metrics = self.engine.get_submission_metrics()
        sig_metrics = self.engine.get_signal_metrics()
        capa_metrics = self.engine.get_capa_metrics()

        return {
            "overall_score": self.engine.get_overall_compliance_score(),
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": [
                {
                    "metric_name": "Submission Timeliness",
                    "metric_value": sub_metrics["on_time_rate"],
                    "unit": "%",
                    "status": "on_track" if sub_metrics["on_time_rate"] >= 95 else "at_risk",
                    "period": "Last 12 Months"
                },
                {
                    "metric_name": "CAPA Closure Rate",
                    "metric_value": capa_metrics["closure_rate"],
                    "unit": "%",
                    "status": "on_track" if capa_metrics["closure_rate"] >= 90 else "at_risk",
                    "period": "YTD"
                },
                {
                    "metric_name": "Signal Backlog",
                    "metric_value": float(sig_metrics["review_backlog"]),
                    "unit": "count",
                    "status": "on_track" if sig_metrics["review_backlog"] < 10 else "at_risk",
                    "period": "Current"
                }
            ],
            "details": {
                "submissions": sub_metrics,
                "signals": sig_metrics,
                "capas": capa_metrics
            }
        }
