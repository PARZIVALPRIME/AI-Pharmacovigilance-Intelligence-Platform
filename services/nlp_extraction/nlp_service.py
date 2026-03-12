"""
NLP Extraction Service
AI Pharmacovigilance Intelligence Platform

Service layer wrapping the extractor backends with:
  - Batch processing with progress tracking
  - Database persistence of NLP results
  - Performance metrics collection
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from database.connection import SessionLocal
from database.models import AdverseEventReport, NLPExtraction, ReportStatus
from services.nlp_extraction.extractor import (
    ExtractionResult,
    get_extractor,
    BaseExtractor,
)


class NLPExtractionService:
    """
    Orchestrates NLP extraction across all unprocessed adverse event reports.

    Usage
    -----
    service = NLPExtractionService(mode="rule_based")
    results = service.process_pending_reports(batch_size=100)
    result  = service.extract_from_text("Patient experienced nausea after Drug A")
    """

    def __init__(self, mode: str = "rule_based") -> None:
        self.extractor: BaseExtractor = get_extractor(mode)
        self.mode = mode
        logger.info("NLPExtractionService initialised with mode='{}'", mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract adverse events from a single text string."""
        logger.debug("Extracting from text ({} chars)", len(text))
        return self.extractor.extract(text)

    def process_pending_reports(self, batch_size: int = 200, limit: Optional[int] = None) -> dict:
        """
        Process all NLP-unprocessed adverse event reports in the database.

        Returns summary dict with processed count and average confidence.
        """
        logger.info("Starting NLP batch processing (batch_size={}, limit={})…", batch_size, limit)
        start = time.perf_counter()
        processed = 0
        total_confidence = 0.0

        with SessionLocal() as session:
            query = (
                session.query(AdverseEventReport)
                .filter(AdverseEventReport.nlp_processed == False)
                .filter(AdverseEventReport.source_text.isnot(None))
            )
            if limit:
                query = query.limit(limit)

            reports = query.all()
            logger.info("Found {} unprocessed reports.", len(reports))

            for i, report in enumerate(reports, 1):
                try:
                    result = self.extractor.extract(report.source_text or "")

                    # Persist NLP extraction
                    extraction = NLPExtraction(
                        report_id=report.id,
                        model_name=self.mode,
                        extracted_drugs=result.drugs,
                        extracted_events=result.adverse_events,
                        extracted_symptoms=result.symptoms,
                        extracted_severity=result.severity,
                        entities_raw={
                            "entities": [
                                {
                                    "text": e.text,
                                    "label": e.label,
                                    "confidence": e.confidence,
                                }
                                for e in result.entities
                            ]
                        },
                        processing_time_ms=result.processing_time_ms,
                        confidence_score=result.confidence_score,
                    )
                    session.add(extraction)

                    # Update report status
                    report.nlp_processed = True
                    report.status = ReportStatus.PROCESSED
                    if result.confidence_score:
                        report.confidence_score = result.confidence_score

                    processed += 1
                    total_confidence += result.confidence_score or 0.0

                    if i % batch_size == 0:
                        session.commit()
                        logger.info("NLP batch {}: {} reports processed so far.", i // batch_size, i)

                except Exception as exc:
                    logger.error("NLP processing failed for report {}: {}", report.id, exc)
                    continue

            session.commit()

        elapsed = time.perf_counter() - start
        avg_conf = total_confidence / max(processed, 1)
        summary = {
            "processed": processed,
            "avg_confidence": round(avg_conf, 4),
            "elapsed_seconds": round(elapsed, 2),
            "mode": self.mode,
        }
        logger.info("NLP batch complete: {}", summary)
        return summary

    def get_extraction_stats(self) -> dict:
        """Return NLP processing statistics from the database."""
        with SessionLocal() as session:
            total_reports = session.query(AdverseEventReport).count()
            processed = session.query(AdverseEventReport).filter(
                AdverseEventReport.nlp_processed == True
            ).count()
            total_extractions = session.query(NLPExtraction).count()

        return {
            "total_reports": total_reports,
            "nlp_processed": processed,
            "pending": total_reports - processed,
            "total_extractions": total_extractions,
            "completion_pct": round(processed / max(total_reports, 1) * 100, 1),
        }
