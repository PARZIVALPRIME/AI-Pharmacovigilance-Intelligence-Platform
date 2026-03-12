"""
Master Pipeline Orchestrator
AI Pharmacovigilance Intelligence Platform

Runs the full end-to-end pharmacovigilance pipeline:
  Step 1: Data Ingestion
  Step 2: NLP Extraction
  Step 3: Risk Signal Detection
  Step 4: Report Generation

Can be triggered via CLI, API, or scheduled job.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

# Root path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from loguru import logger
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    RICH_AVAILABLE = False

app = typer.Typer(
    name="pharma-pipeline",
    help="AI Pharmacovigilance Platform Pipeline Orchestrator",
    add_completion=False,
)


def _print_banner():
    print("=" * 70)
    print(" AI Pharmacovigilance Intelligence Platform — Pipeline Orchestrator")
    print("=" * 70)
    print(f" Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)


def run_step(name: str, func, *args, **kwargs) -> dict:
    """Execute a pipeline step with timing and error handling."""
    print(f"\n[STEP] {name}")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"  ✓ Completed in {elapsed}s")
        return {"status": "success", "elapsed": elapsed, "result": result}
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"  ✗ FAILED after {elapsed}s: {exc}")
        logger.error("Pipeline step '{}' failed: {}", name, exc)
        return {"status": "error", "elapsed": elapsed, "error": str(exc)}


@app.command()
def full(
    n_records: int = typer.Option(10_000, help="Number of synthetic records to generate"),
    force_regen: bool = typer.Option(False, help="Force regeneration of dataset"),
    nlp_mode: str = typer.Option("rule_based", help="NLP extraction mode: rule_based|transformer|ensemble"),
    nlp_limit: Optional[int] = typer.Option(None, help="Limit NLP processing to N records"),
    skip_nlp: bool = typer.Option(False, help="Skip NLP processing step"),
    skip_signals: bool = typer.Option(False, help="Skip risk signal detection"),
    skip_report: bool = typer.Option(False, help="Skip report generation"),
    report_format: str = typer.Option("json", help="Report format: json|excel|pdf"),
):
    """Run the full pharmacovigilance pipeline."""
    _print_banner()
    pipeline_start = time.perf_counter()
    results = {}

    # Step 1: Data Ingestion
    from services.data_ingestion import DataIngestionService
    service_1 = DataIngestionService()
    results["ingestion"] = run_step(
        "1. Data Ingestion",
        service_1.run_full_pipeline,
        n_records=n_records,
        force_regenerate=force_regen,
    )

    # Step 2: NLP Extraction
    if not skip_nlp:
        from services.nlp_extraction import NLPExtractionService
        service_2 = NLPExtractionService(mode=nlp_mode)
        results["nlp"] = run_step(
            "2. NLP Adverse Event Extraction",
            service_2.process_pending_reports,
            limit=nlp_limit,
        )
    else:
        print("\n[STEP] 2. NLP Extraction — SKIPPED")

    # Step 3: Risk Signal Detection
    if not skip_signals:
        from services.risk_detection import RiskSignalDetectionService
        service_3 = RiskSignalDetectionService()
        results["signals"] = run_step(
            "3. Risk Signal Detection",
            service_3.run_full_detection,
        )
    else:
        print("\n[STEP] 3. Risk Signal Detection — SKIPPED")

    # Step 4: Report Generation
    if not skip_report:
        from services.reporting import ReportingService
        service_4 = ReportingService()
        results["report"] = run_step(
            "4. Report Generation",
            service_4.generate_report,
            format=report_format,
        )
    else:
        print("\n[STEP] 4. Report Generation — SKIPPED")

    # Summary
    total_elapsed = round(time.perf_counter() - pipeline_start, 2)
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)
    for step, res in results.items():
        status = "✓" if res.get("status") == "success" else "✗"
        print(f"  {status} {step:20s} [{res.get('elapsed', 0):.1f}s]")
    print(f"\n  Total time: {total_elapsed}s")
    print("=" * 70)

    return results


@app.command()
def ingest(
    n_records: int = typer.Option(10_000, help="Records to generate"),
    force_regen: bool = typer.Option(False, help="Force regeneration"),
):
    """Run only the data ingestion step."""
    _print_banner()
    from services.data_ingestion import DataIngestionService
    result = DataIngestionService().run_full_pipeline(n_records=n_records, force_regenerate=force_regen)
    print(f"Ingestion complete: {result}")


@app.command()
def detect():
    """Run only the risk signal detection step."""
    _print_banner()
    from services.risk_detection import RiskSignalDetectionService
    result = RiskSignalDetectionService().run_full_detection()
    print(f"Detection complete: {result}")


@app.command()
def report(
    format: str = typer.Option("json", help="Output format: json|excel|pdf"),
):
    """Generate a safety report."""
    _print_banner()
    from services.reporting import ReportingService
    result = ReportingService().generate_report(format=format)
    print(f"Report generated: format={format}")


if __name__ == "__main__":
    app()
