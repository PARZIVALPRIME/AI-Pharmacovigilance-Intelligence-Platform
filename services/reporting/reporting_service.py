"""
Reporting Service
AI Pharmacovigilance Intelligence Platform

Generates aggregate safety reports in multiple formats:
  - PDF (regulatory-style safety summary)
  - Excel (detailed data export with multiple sheets)
  - JSON (machine-readable API output)

Each report includes:
  - Executive summary statistics
  - Top adverse events ranking
  - Drug safety profiles
  - Risk signal summary
  - Trend analysis
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from database.connection import SessionLocal
from database.models import AdverseEventReport, RiskSignal, Report, AuditLog, AggregateReportType

# Output directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EXPORT_DIR = ROOT_DIR / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data retrieval helpers
# ---------------------------------------------------------------------------

class ReportDataProvider:
    """Fetches and aggregates data needed for reports."""

    @staticmethod
    def get_summary_stats() -> dict:
        with SessionLocal() as session:
            total = session.query(AdverseEventReport).count()
            serious = session.query(AdverseEventReport).filter(
                AdverseEventReport.is_serious == True
            ).count()
            signals = session.query(RiskSignal).count()
        return {
            "total_reports": total,
            "serious_reports": serious,
            "seriousness_rate": round(serious / max(total, 1) * 100, 2),
            "total_signals": signals,
            "report_generated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def get_top_adverse_events(n: int = 20) -> pd.DataFrame:
        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.adverse_event,
                AdverseEventReport.severity,
            ).filter(
                AdverseEventReport.is_duplicate == False
            ).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["adverse_event", "severity"])
        ae_counts = df.groupby("adverse_event").size().reset_index()
        ae_counts.columns = ["adverse_event", "count"]
        top = ae_counts.sort_values("count", ascending=False).head(n)
        return top

    @staticmethod
    def get_drug_safety_profile() -> pd.DataFrame:
        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.drug_name,
                AdverseEventReport.drug_class,
                AdverseEventReport.severity,
                AdverseEventReport.is_serious,
            ).filter(AdverseEventReport.is_duplicate == False).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["drug_name", "drug_class", "severity", "is_serious"])
        profile = df.groupby("drug_name").agg(
            total_reports=("drug_name", "count"),
            serious_reports=("is_serious", "sum"),
            drug_class=("drug_class", "first"),
        ).reset_index()
        profile["seriousness_rate"] = (
            profile["serious_reports"] / profile["total_reports"] * 100
        ).round(2)
        profile = profile.sort_values("total_reports", ascending=False)
        return profile

    @staticmethod
    def get_signals_summary() -> pd.DataFrame:
        with SessionLocal() as session:
            signals = session.query(RiskSignal).order_by(
                RiskSignal.severity_score.desc()
            ).limit(100).all()

        if not signals:
            return pd.DataFrame()

        return pd.DataFrame([{
            "drug_name": s.drug_name,
            "adverse_event": s.adverse_event,
            "prr": s.prr,
            "ror": s.ror,
            "report_count": s.report_count,
            "severity_score": s.severity_score,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            "detection_date": str(s.detection_date) if s.detection_date else None,
        } for s in signals])

    @staticmethod
    def get_trend_data() -> pd.DataFrame:
        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.report_date,
                AdverseEventReport.drug_name,
                AdverseEventReport.adverse_event,
                AdverseEventReport.severity,
            ).filter(
                AdverseEventReport.is_duplicate == False,
                AdverseEventReport.report_date.isnot(None),
            ).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["report_date", "drug_name", "adverse_event", "severity"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["year_month"] = df["report_date"].dt.to_period("M").astype(str)
        trend = df.groupby(["year_month"]).size().reset_index()
        trend.columns = ["year_month", "report_count"]
        return trend.sort_values("year_month")

    @staticmethod
    def get_geographic_distribution() -> pd.DataFrame:
        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.country,
                AdverseEventReport.region,
                AdverseEventReport.is_serious,
            ).filter(AdverseEventReport.is_duplicate == False).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["country", "region", "is_serious"])
        geo = df.groupby("country").agg(
            total=("country", "count"),
            serious=("is_serious", "sum"),
            region=("region", "first"),
        ).reset_index()
        geo["seriousness_rate"] = (geo["serious"] / geo["total"] * 100).round(2)
        return geo.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

class JSONReportGenerator:
    """Generates machine-readable JSON safety reports."""

    def generate(self, output_path: Optional[Path] = None) -> dict:
        provider = ReportDataProvider()

        report = {
            "report_metadata": {
                "report_id": f"RPT-{uuid.uuid4().hex[:10].upper()}",
                "report_type": "aggregate_safety_report",
                "generated_at": datetime.utcnow().isoformat(),
                "format": "json",
                "version": "1.0.0",
            },
            "summary": provider.get_summary_stats(),
            "top_adverse_events": provider.get_top_adverse_events(20).to_dict("records"),
            "drug_safety_profiles": provider.get_drug_safety_profile().head(30).to_dict("records"),
            "risk_signals": provider.get_signals_summary().head(20).to_dict("records"),
            "monthly_trend": provider.get_trend_data().to_dict("records"),
            "geographic_distribution": provider.get_geographic_distribution().head(30).to_dict("records"),
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("JSON report saved → {}", output_path)

        return report


class ExcelReportGenerator:
    """Generates detailed multi-sheet Excel safety reports."""

    def generate(self, output_path: Optional[Path] = None) -> bytes:
        provider = ReportDataProvider()
        output_path = output_path or EXPORT_DIR / f"safety_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        output_path = Path(output_path)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary = provider.get_summary_stats()
            summary_df = pd.DataFrame([
                {"Metric": k.replace("_", " ").title(), "Value": v}
                for k, v in summary.items()
            ])
            summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)

            # Sheet 2: Top Adverse Events
            ae_df = provider.get_top_adverse_events(50)
            if not ae_df.empty:
                ae_df.to_excel(writer, sheet_name="Top Adverse Events", index=False)

            # Sheet 3: Drug Safety Profiles
            drug_df = provider.get_drug_safety_profile()
            if not drug_df.empty:
                drug_df.to_excel(writer, sheet_name="Drug Safety Profiles", index=False)

            # Sheet 4: Risk Signals
            signals_df = provider.get_signals_summary()
            if not signals_df.empty:
                signals_df.to_excel(writer, sheet_name="Risk Signals", index=False)

            # Sheet 5: Monthly Trends
            trend_df = provider.get_trend_data()
            if not trend_df.empty:
                trend_df.to_excel(writer, sheet_name="Monthly Trends", index=False)

            # Sheet 6: Geographic Distribution
            geo_df = provider.get_geographic_distribution()
            if not geo_df.empty:
                geo_df.to_excel(writer, sheet_name="Geographic Distribution", index=False)

        excel_bytes = buffer.getvalue()

        with open(output_path, "wb") as f:
            f.write(excel_bytes)
        logger.info("Excel report saved → {}", output_path)
        return excel_bytes


class PDFReportGenerator:
    """Generates regulatory-style PDF safety summary reports using ReportLab."""

    def generate(self, output_path: Optional[Path] = None) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, HRFlowable, PageBreak,
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.warning("ReportLab not available. Returning empty bytes.")
            return b""

        provider = ReportDataProvider()
        output_path = output_path or EXPORT_DIR / f"safety_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = Path(output_path)

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=18, textColor=colors.HexColor("#1a237e"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "Heading1", parent=styles["Heading1"],
            fontSize=13, textColor=colors.HexColor("#283593"),
            spaceBefore=16, spaceAfter=8,
        )
        body_style = styles["BodyText"]
        body_style.fontSize = 9

        story = []

        # Title page
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph("AI Pharmacovigilance Intelligence Platform", title_style))
        story.append(Paragraph("Aggregate Safety Report", heading_style))
        story.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            body_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a237e")))
        story.append(Spacer(1, 0.5 * cm))

        # Executive summary
        story.append(Paragraph("1. Executive Summary", heading_style))
        summary = provider.get_summary_stats()
        summary_data = [["Metric", "Value"]] + [
            [k.replace("_", " ").title(), str(v)]
            for k, v in summary.items()
            if k != "report_generated_at"
        ]
        self._add_table(story, summary_data, styles)

        story.append(Spacer(1, 0.5 * cm))

        # Top adverse events
        story.append(Paragraph("2. Top Adverse Events", heading_style))
        ae_df = provider.get_top_adverse_events(15)
        if not ae_df.empty:
            ae_data = [["Adverse Event", "Report Count"]] + ae_df.values.tolist()
            self._add_table(story, ae_data, styles)
        story.append(Spacer(1, 0.5 * cm))

        # Drug safety profiles
        story.append(Paragraph("3. Drug Safety Profiles (Top 15)", heading_style))
        drug_df = provider.get_drug_safety_profile().head(15)
        if not drug_df.empty:
            drug_data = [["Drug", "Drug Class", "Total Reports", "Serious Reports", "Seriousness Rate %"]]
            for _, row in drug_df.iterrows():
                drug_data.append([
                    row["drug_name"],
                    str(row.get("drug_class", "")),
                    str(row["total_reports"]),
                    str(int(row["serious_reports"])),
                    f"{row['seriousness_rate']:.1f}%",
                ])
            self._add_table(story, drug_data, styles)
        story.append(Spacer(1, 0.5 * cm))

        # Risk signals
        story.append(Paragraph("4. Risk Signals (Top 15)", heading_style))
        sig_df = provider.get_signals_summary().head(15)
        if not sig_df.empty:
            sig_data = [["Drug", "Adverse Event", "PRR", "ROR", "Reports", "Severity Score"]]
            for _, row in sig_df.iterrows():
                sig_data.append([
                    str(row["drug_name"]),
                    str(row["adverse_event"]),
                    f"{row['prr']:.2f}" if pd.notna(row.get("prr")) else "N/A",
                    f"{row['ror']:.2f}" if pd.notna(row.get("ror")) else "N/A",
                    str(row.get("report_count", "")),
                    f"{row.get('severity_score', 0):.1f}",
                ])
            self._add_table(story, sig_data, styles)
        story.append(Spacer(1, 0.5 * cm))

        # Geographic distribution (New)
        story.append(Paragraph("5. Geographic Distribution", heading_style))
        geo_df = provider.get_geographic_distribution().head(15)
        if not geo_df.empty:
            geo_data = [["Country", "Region", "Total Reports", "Serious Reports", "Seriousness Rate %"]]
            for _, row in geo_df.iterrows():
                geo_data.append([
                    str(row["country"]),
                    str(row.get("region", "")),
                    str(row["total"]),
                    str(int(row["serious"])),
                    f"{row['seriousness_rate']:.1f}%",
                ])
            self._add_table(story, geo_data, styles)
        story.append(Spacer(1, 0.5 * cm))

        # Trends analysis (New)
        story.append(Paragraph("6. Reporting Trends", heading_style))
        trend_df = provider.get_trend_data().tail(12)
        if not trend_df.empty:
            trend_data = [["Month", "Report Count"]]
            for _, row in trend_df.iterrows():
                trend_data.append([
                    str(row["year_month"]),
                    str(row["report_count"]),
                ])
            self._add_table(story, trend_data, styles)
        story.append(Spacer(1, 0.5 * cm))

        story.append(PageBreak())

        # Disclaimer
        story.append(Paragraph("Disclaimer", heading_style))
        story.append(Paragraph(
            "This report is generated automatically by the AI Pharmacovigilance Intelligence Platform "
            "for informational and research purposes. It does not constitute regulatory guidance "
            "or medical advice. All signals require review by qualified pharmacovigilance professionals "
            "before any regulatory action.",
            body_style,
        ))

        doc.build(story)
        pdf_bytes = buffer.getvalue()

        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("PDF report saved → {}", output_path)
        return pdf_bytes

    @staticmethod
    def _add_table(story: list, data: list, styles) -> None:
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        if len(data) <= 1:
            return

        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)


# ---------------------------------------------------------------------------
# Main Reporting Service
# ---------------------------------------------------------------------------

class ReportingService:
    """
    Unified reporting service supporting PDF, Excel, and JSON output formats.

    Usage
    -----
    service = ReportingService()
    pdf_bytes  = service.generate_report("pdf")
    excel_bytes = service.generate_report("excel")
    json_data  = service.generate_report("json")
    """

    def __init__(self) -> None:
        self._json_gen = JSONReportGenerator()
        self._excel_gen = ExcelReportGenerator()
        self._pdf_gen = PDFReportGenerator()

    def generate_report(self, format: str = "json", report_type: str = "aggregate_safety", output_path: Optional[Path] = None):
        """
        Generate a safety report in the specified format.

        Parameters
        ----------
        format : str
            One of "pdf", "excel", "json"
        report_type : str
            One of "aggregate_safety", "psur", "pbrer", "dsur"
        output_path : Optional[Path]
            Custom output path. Auto-generated if not provided.

        Returns
        -------
        dict | bytes
            JSON dict, or bytes for PDF/Excel
        """
        format = format.lower().strip()
        report_type = report_type.lower().strip()
        report_name = f"{report_type.upper()}_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info("Generating {} report: {}", format, report_name)

        if format == "json":
            result = self._json_gen.generate(output_path=output_path)
        elif format == "excel":
            result = self._excel_gen.generate(output_path=output_path)
        elif format == "pdf":
            result = self._pdf_gen.generate(output_path=output_path)
        else:
            raise ValueError(f"Unsupported report format: {format!r}. Use pdf, excel, or json.")

        self._log_report(report_name, format, report_type, output_path)
        return result

    @staticmethod
    def _log_report(name: str, fmt: str, rtype: str, path: Optional[Path]) -> None:
        try:
            with SessionLocal() as session:
                # Add Report metadata
                record = Report(
                    report_name=name,
                    report_type=rtype,
                    format=fmt,
                    file_path=str(path) if path else None,
                    generation_date=datetime.utcnow(),
                    status="completed",
                )
                session.add(record)

                # Add Audit Log entry (JD requirement: audit readiness)
                audit = AuditLog(
                    action="generate_report",
                    entity_type="Report",
                    entity_id=name,
                    user="system",
                    details={"format": fmt, "type": rtype, "path": str(path) if path else None},
                )
                session.add(audit)
                session.commit()
        except Exception as exc:
            logger.warning("Report metadata/audit log save failed: {}", exc)
