"""
Data Ingestion Service
AI Pharmacovigilance Intelligence Platform

Orchestrates the full data ingestion pipeline:
  1. Source acquisition (FAERS download or synthetic generation)
  2. Data cleaning & normalisation
  3. Bulk database insertion
  4. Ingestion metrics / audit logging
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from database.connection import SessionLocal, create_all_tables
from database.models import AdverseEventReport, Drug, AuditLog, SeverityLevel, OutcomeType, GenderType, ClinicalPhase, ReportStatus
from services.data_ingestion.synthetic_generator import generate_synthetic_dataset, download_faers_dataset
from services.data_ingestion.data_cleaner import DataCleaner

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


class DataIngestionService:
    """
    Manages the end-to-end data ingestion workflow for pharmacovigilance data.

    Usage
    -----
    service = DataIngestionService()
    service.run_full_pipeline(n_records=10_000)
    """

    BATCH_SIZE = 500

    def __init__(self) -> None:
        self.cleaner = DataCleaner()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_pipeline(self, n_records: int = 10_000, force_regenerate: bool = False) -> dict:
        """
        Execute the complete ingestion pipeline.

        Returns a summary dict with counts and timing.
        """
        start_time = time.perf_counter()
        logger.info("Starting data ingestion pipeline — target records: {}", n_records)

        # 1. Schema creation
        create_all_tables()
        logger.info("Database schema ensured.")

        # 2. Acquire raw data
        raw_df = self._acquire_data(n_records=n_records, force_regenerate=force_regenerate)
        logger.info("Acquired {} raw records.", len(raw_df))

        # 3. Clean & normalise
        clean_df = self.cleaner.clean(raw_df)
        logger.info("Cleaned dataset: {} records.", len(clean_df))

        # 4. Save processed CSV
        processed_path = PROCESSED_DIR / "pharmacovigilance_processed.csv"
        clean_df.to_csv(processed_path, index=False)
        logger.info("Saved processed CSV → {}", processed_path)

        # 5. Database load
        load_stats = self._load_to_database(clean_df)
        logger.info("Database load complete: {}", load_stats)

        elapsed = time.perf_counter() - start_time
        summary = {
            "status": "success",
            "raw_records": len(raw_df),
            "clean_records": len(clean_df),
            "inserted": load_stats.get("inserted", 0),
            "skipped": load_stats.get("skipped", 0),
            "elapsed_seconds": round(elapsed, 2),
            "processed_file": str(processed_path),
        }

        self._write_audit_log(summary)
        logger.info("Pipeline complete in {:.2f}s — summary: {}", elapsed, summary)
        return summary

    def load_from_csv(self, csv_path: Path) -> dict:
        """Ingest from an existing CSV file."""
        logger.info("Loading from CSV: {}", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        clean_df = self.cleaner.clean(df)
        return self._load_to_database(clean_df)

    def get_ingestion_stats(self) -> dict:
        """Return current database ingestion statistics."""
        with SessionLocal() as session:
            total = session.query(AdverseEventReport).count()
            serious = session.query(AdverseEventReport).filter(
                AdverseEventReport.is_serious == True
            ).count()
            drugs = session.query(Drug).count()
            return {
                "total_reports": total,
                "serious_reports": serious,
                "drugs_registered": drugs,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _acquire_data(self, n_records: int, force_regenerate: bool) -> pd.DataFrame:
        """
        Try FAERS download first; fall back to synthetic generation.
        Uses cached file if it exists.
        """
        synthetic_path = RAW_DIR / "synthetic_pvdata.csv"

        if synthetic_path.exists() and not force_regenerate:
            logger.info("Loading cached synthetic dataset from {}", synthetic_path)
            return pd.read_csv(synthetic_path, low_memory=False)

        # Try real FAERS
        real_df = download_faers_dataset(RAW_DIR)
        if real_df is not None and len(real_df) >= 1000:
            real_df.to_csv(synthetic_path, index=False)
            return real_df

        # Generate synthetic
        logger.info("Generating synthetic dataset with {} records...", n_records)
        df = generate_synthetic_dataset(n_records=n_records)
        df.to_csv(synthetic_path, index=False)
        logger.info("Synthetic dataset saved → {}", synthetic_path)
        return df

    def _load_to_database(self, df: pd.DataFrame) -> dict:
        """Bulk-insert cleaned records into the database."""
        inserted = 0
        skipped = 0
        errors = 0

        # Ensure drug registry is populated first
        drug_id_map = self._ensure_drug_registry(df)

        # Batch insert adverse event reports
        with SessionLocal() as session:
            batch = []
            for _, row in df.iterrows():
                try:
                    report = self._row_to_report(row, drug_id_map)
                    batch.append(report)

                    if len(batch) >= self.BATCH_SIZE:
                        ins, skip = self._flush_batch(session, batch)
                        inserted += ins
                        skipped += skip
                        batch = []

                except Exception as exc:
                    errors += 1
                    logger.warning("Row conversion error: {}", exc)
                    continue

            # Final batch
            if batch:
                ins, skip = self._flush_batch(session, batch)
                inserted += ins
                skipped += skip

        return {"inserted": inserted, "skipped": skipped, "errors": errors}

    def _ensure_drug_registry(self, df: pd.DataFrame) -> dict[str, int]:
        """Upsert drugs into the Drug table and return name→id map."""
        drug_id_map: dict[str, int] = {}
        unique_drugs = df[["drug_name", "drug_class"]].drop_duplicates()

        with SessionLocal() as session:
            for _, row in unique_drugs.iterrows():
                name = str(row.get("drug_name", "")).strip()
                drug_class = str(row.get("drug_class", "")).strip() or None
                if not name:
                    continue

                existing = session.query(Drug).filter(Drug.name == name).first()
                if existing:
                    drug_id_map[name] = existing.id
                else:
                    brand = str(df[df["drug_name"] == name]["brand_name"].iloc[0]).strip() if "brand_name" in df.columns else None
                    drug = Drug(
                        name=name,
                        brand_name=brand,
                        drug_class=drug_class,
                        is_active=True,
                    )
                    session.add(drug)
                    session.flush()
                    drug_id_map[name] = drug.id

            session.commit()

        return drug_id_map

    @staticmethod
    def _row_to_report(row: pd.Series, drug_id_map: dict) -> AdverseEventReport:
        """Convert a cleaned DataFrame row to an AdverseEventReport ORM object."""
        drug_name = str(row.get("drug_name", "")).strip()
        report_id = str(row.get("report_id", f"PVR-{uuid.uuid4().hex[:12].upper()}")).strip()

        # Safe enum coercion
        def safe_enum(enum_class, value, default):
            try:
                return enum_class(str(value).lower())
            except (ValueError, AttributeError):
                return default

        report_date = row.get("report_date")
        if hasattr(report_date, "date"):
            report_date = report_date.date()
        elif isinstance(report_date, str):
            try:
                report_date = date.fromisoformat(report_date)
            except Exception:
                report_date = None

        receipt_date = row.get("receipt_date")
        if hasattr(receipt_date, "date"):
            receipt_date = receipt_date.date()

        age = row.get("patient_age")
        try:
            age = float(age) if age is not None and str(age) not in ("nan", "None", "") else None
        except (ValueError, TypeError):
            age = None

        confidence = row.get("confidence_score")
        try:
            confidence = float(confidence) if confidence is not None and str(confidence) not in ("nan", "None", "") else None
        except (ValueError, TypeError):
            confidence = None

        return AdverseEventReport(
            report_id=report_id,
            drug_id=drug_id_map.get(drug_name),
            drug_name=drug_name,
            adverse_event=str(row.get("adverse_event", "")).strip(),
            severity=safe_enum(SeverityLevel, row.get("severity", "unknown"), SeverityLevel.UNKNOWN),
            outcome=safe_enum(OutcomeType, row.get("outcome", "unknown"), OutcomeType.UNKNOWN),
            patient_age=age,
            patient_age_group=str(row.get("patient_age_group", "unknown")).strip() or None,
            gender=safe_enum(GenderType, row.get("gender", "unknown"), GenderType.UNKNOWN),
            country=str(row.get("country", "")).strip() or None,
            region=str(row.get("region", "")).strip() or None,
            report_date=report_date,
            receipt_date=receipt_date,
            clinical_phase=safe_enum(ClinicalPhase, row.get("clinical_phase", "unknown"), ClinicalPhase.UNKNOWN),
            drug_class=str(row.get("drug_class", "")).strip() or None,
            source_text=str(row.get("source_text", "")).strip() or None,
            source_type=str(row.get("source_type", "")).strip() or None,
            is_serious=bool(row.get("is_serious", False)),
            confidence_score=confidence,
            status=ReportStatus.PENDING,
            nlp_processed=False,
        )

    @staticmethod
    def _flush_batch(session: Session, batch: list) -> tuple[int, int]:
        """Attempt to insert a batch; skip duplicates gracefully."""
        inserted = 0
        skipped = 0
        for report in batch:
            try:
                existing = session.query(AdverseEventReport).filter(
                    AdverseEventReport.report_id == report.report_id
                ).first()
                if existing:
                    skipped += 1
                else:
                    session.add(report)
                    inserted += 1
            except Exception:
                skipped += 1
        session.commit()
        return inserted, skipped

    @staticmethod
    def _write_audit_log(summary: dict) -> None:
        """Write pipeline run to audit log table."""
        try:
            with SessionLocal() as session:
                audit = AuditLog(
                    action="data_ingestion_pipeline",
                    entity_type="AdverseEventReport",
                    entity_id="batch",
                    details=summary,
                    success=True,
                )
                session.add(audit)
                session.commit()
        except Exception as exc:
            logger.warning("Audit log write failed: {}", exc)
