"""
FastAPI Application — API Gateway
AI Pharmacovigilance Intelligence Platform

Production-grade REST API with:
  - Full CRUD for adverse event reports
  - NLP extraction endpoint
  - Risk signal detection
  - Report generation (PDF / Excel / JSON)
  - AI assistant chat endpoint
  - Analytics endpoints
  - Health checks
  - OpenAPI documentation
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure parent packages are resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Depends,
    BackgroundTasks,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api_gateway.schemas import (
    HealthResponse,
    AdverseEventReportSchema,
    AdverseEventListResponse,
    NLPExtractRequest,
    NLPExtractResponse,
    NLPBatchRequest,
    RiskSignalSchema,
    SignalDetectionRequest,
    SignalListResponse,
    ReportGenerationRequest,
    ReportGenerationResponse,
    AIQueryRequest,
    AIQueryResponse,
    ConversationHistoryResponse,
    AnalyticsSummaryResponse,
    IngestionRequest,
    IngestionResponse,
    StandardResponse,
    SignalStatusUpdateRequest,
    SignalStatusUpdateResponse,
    AuditLogSchema,
    AuditLogListResponse,
    DrugSchema,
    DrugListResponse,
    DrugCreateRequest,
    HASubmissionSchema,
    HASubmissionListResponse,
    HASubmissionCreateRequest,
    CAPASchema,
    CAPAListResponse,
    CAPACreateRequest,
    ComplianceDashboardResponse,
)
from database.connection import get_db, health_check, create_all_tables
from database.models import (
    AdverseEventReport,
    RiskSignal,
    Drug,
    SignalStatus,
    SeverityLevel,
    AuditLog,
    HASubmission,
    CAPA,
    CausalityAssessment,
)
from services.data_ingestion import DataIngestionService
from services.nlp_extraction import NLPExtractionService, get_extractor
from services.risk_detection import RiskSignalDetectionService
from services.reporting import ReportingService
from services.ai_assistant import AIAssistantService
from services.compliance import ComplianceService

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

APP_START_TIME = time.time()

app = FastAPI(
    title="AI Pharmacovigilance Intelligence Platform",
    description=(
        "Production-grade pharmacovigilance platform providing adverse event analysis, "
        "risk signal detection, NLP extraction, and AI-powered querying capabilities."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "PharmaAI Team",
        "email": "pharmacovigilance@pharmai.io",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Log request timing and add X-Process-Time header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 4)
    response.headers["X-Process-Time"] = str(elapsed)
    return response


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialise database schema on startup."""
    logger.info("API Gateway starting up…")
    create_all_tables()
    logger.info("Database schema initialised.")


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def get_ingestion_service() -> DataIngestionService:
    return DataIngestionService()

def get_nlp_service() -> NLPExtractionService:
    return NLPExtractionService(mode="rule_based")

def get_risk_service() -> RiskSignalDetectionService:
    return RiskSignalDetectionService()

def get_reporting_service() -> ReportingService:
    return ReportingService()

# Single assistant instance (state preserved across requests in same process)
_assistant_sessions: Dict[str, AIAssistantService] = {}

def get_assistant(session_id: str = "default") -> AIAssistantService:
    if session_id not in _assistant_sessions:
        _assistant_sessions[session_id] = AIAssistantService()
    return _assistant_sessions[session_id]


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@app.get("/health", response_model=dict, tags=["Health"])
async def health():
    """System health check."""
    db_status = health_check()
    return {
        "status": "healthy" if db_status["status"] == "healthy" else "degraded",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "uptime_seconds": round(time.time() - APP_START_TIME, 1),
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with platform info."""
    return {
        "platform": "AI Pharmacovigilance Intelligence Platform",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
    }


# ---------------------------------------------------------------------------
# Data Ingestion Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/ingestion/run", response_model=IngestionResponse, tags=["Data Ingestion"])
async def run_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    service: DataIngestionService = Depends(get_ingestion_service),
):
    """
    Trigger the data ingestion pipeline.
    Generates or downloads pharmacovigilance data and loads it into the database.
    """
    try:
        result = service.run_full_pipeline(
            n_records=request.n_records,
            force_regenerate=request.force_regenerate,
        )
        return IngestionResponse(**result)
    except Exception as exc:
        logger.error("Ingestion error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/ingestion/stats", tags=["Data Ingestion"])
async def ingestion_stats(service: DataIngestionService = Depends(get_ingestion_service)):
    """Return current ingestion statistics."""
    return service.get_ingestion_stats()


# ---------------------------------------------------------------------------
# Adverse Event Reports Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/reports", response_model=AdverseEventListResponse, tags=["Adverse Events"])
async def list_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    drug_name: Optional[str] = Query(None),
    adverse_event: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    is_serious: Optional[bool] = Query(None),
    drug_class: Optional[str] = Query(None),
    db=Depends(get_db),
):
    """
    List adverse event reports with filtering and pagination.
    """
    query = db.query(AdverseEventReport).filter(AdverseEventReport.is_duplicate == False)

    if drug_name:
        query = query.filter(AdverseEventReport.drug_name.ilike(f"%{drug_name}%"))
    if adverse_event:
        query = query.filter(AdverseEventReport.adverse_event.ilike(f"%{adverse_event}%"))
    if severity:
        try:
            sev_enum = SeverityLevel(severity.lower())
            query = query.filter(AdverseEventReport.severity == sev_enum)
        except ValueError:
            pass
    if country:
        query = query.filter(AdverseEventReport.country.ilike(f"%{country}%"))
    if is_serious is not None:
        query = query.filter(AdverseEventReport.is_serious == is_serious)
    if drug_class:
        query = query.filter(AdverseEventReport.drug_class.ilike(f"%{drug_class}%"))

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return AdverseEventListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[AdverseEventReportSchema.model_validate(r) for r in items],
    )


@app.get("/api/v1/reports/{report_id}", response_model=AdverseEventReportSchema, tags=["Adverse Events"])
async def get_report(report_id: str, db=Depends(get_db)):
    """Get a specific adverse event report by report_id."""
    report = db.query(AdverseEventReport).filter(
        AdverseEventReport.report_id == report_id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id!r} not found.")
    return AdverseEventReportSchema.model_validate(report)


@app.get("/api/v1/reports/stats/summary", tags=["Adverse Events"])
async def reports_summary(db=Depends(get_db)):
    """Return aggregate statistics for adverse event reports."""
    total = db.query(AdverseEventReport).filter(AdverseEventReport.is_duplicate == False).count()
    serious = db.query(AdverseEventReport).filter(
        AdverseEventReport.is_serious == True,
        AdverseEventReport.is_duplicate == False,
    ).count()
    drugs = db.query(AdverseEventReport.drug_name).distinct().count()
    events = db.query(AdverseEventReport.adverse_event).distinct().count()
    countries = db.query(AdverseEventReport.country).distinct().count()

    # Date range
    from sqlalchemy import func
    min_date = db.query(func.min(AdverseEventReport.report_date)).scalar()
    max_date = db.query(func.max(AdverseEventReport.report_date)).scalar()

    return {
        "total_reports": total,
        "serious_reports": serious,
        "seriousness_rate_pct": round(serious / max(total, 1) * 100, 2),
        "unique_drugs": drugs,
        "unique_adverse_events": events,
        "unique_countries": countries,
        "date_range": {
            "from": str(min_date) if min_date else None,
            "to": str(max_date) if max_date else None,
        },
    }


@app.get("/api/v1/reports/stats/top-events", tags=["Adverse Events"])
async def top_adverse_events(
    n: int = Query(20, ge=1, le=100),
    drug_name: Optional[str] = Query(None),
    db=Depends(get_db),
):
    """Return top N most frequently reported adverse events."""
    from sqlalchemy import func

    query = db.query(
        AdverseEventReport.adverse_event,
        func.count(AdverseEventReport.id).label("count"),
    ).filter(AdverseEventReport.is_duplicate == False)

    if drug_name:
        query = query.filter(AdverseEventReport.drug_name.ilike(f"%{drug_name}%"))

    results = (
        query.group_by(AdverseEventReport.adverse_event)
        .order_by(func.count(AdverseEventReport.id).desc())
        .limit(n)
        .all()
    )

    return [{"adverse_event": r[0], "count": r[1]} for r in results]


@app.get("/api/v1/reports/stats/top-drugs", tags=["Adverse Events"])
async def top_drugs(n: int = Query(20, ge=1, le=100), db=Depends(get_db)):
    """Return top N drugs by report count."""
    from sqlalchemy import func

    from sqlalchemy import Integer as SAInteger, case

    results = (
        db.query(
            AdverseEventReport.drug_name,
            AdverseEventReport.drug_class,
            func.count(AdverseEventReport.id).label("count"),
            func.sum(
                case(
                    (AdverseEventReport.is_serious == True, 1),
                    else_=0,
                ).cast(SAInteger)
            ).label("serious"),
        )
        .filter(AdverseEventReport.is_duplicate == False)
        .group_by(AdverseEventReport.drug_name, AdverseEventReport.drug_class)
        .order_by(func.count(AdverseEventReport.id).desc())
        .limit(n)
        .all()
    )

    return [{
        "drug_name": r[0],
        "drug_class": r[1],
        "total_reports": r[2],
        "serious_reports": r[3] or 0,
    } for r in results]


@app.get("/api/v1/reports/stats/by-country", tags=["Adverse Events"])
async def reports_by_country(db=Depends(get_db)):
    """Return report counts grouped by country."""
    from sqlalchemy import func

    results = (
        db.query(
            AdverseEventReport.country,
            AdverseEventReport.region,
            func.count(AdverseEventReport.id).label("total"),
        )
        .filter(AdverseEventReport.is_duplicate == False, AdverseEventReport.country.isnot(None))
        .group_by(AdverseEventReport.country, AdverseEventReport.region)
        .order_by(func.count(AdverseEventReport.id).desc())
        .all()
    )

    return [{"country": r[0], "region": r[1], "total": r[2]} for r in results]


@app.get("/api/v1/reports/stats/monthly-trend", tags=["Adverse Events"])
async def monthly_trend(db=Depends(get_db)):
    """Return monthly report volume trend."""
    from sqlalchemy import func, extract

    rows = (
        db.query(AdverseEventReport.report_date)
        .filter(AdverseEventReport.report_date.isnot(None))
        .all()
    )

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["report_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["year_month"] = df["report_date"].dt.to_period("M").astype(str)
    trend_counts = df.groupby("year_month").size().reset_index()
    trend_counts.columns = ["year_month", "count"]
    return trend_counts.sort_values("year_month").to_dict("records")


# ---------------------------------------------------------------------------
# NLP Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/nlp/extract", response_model=NLPExtractResponse, tags=["NLP Extraction"])
async def nlp_extract(request: NLPExtractRequest):
    """
    Extract adverse events, drugs, and symptoms from free text using NLP.

    Example input: "Patient experienced dizziness and nausea after taking Metformin."
    """
    try:
        extractor = get_extractor(request.mode)
        result = extractor.extract(request.text)
        return NLPExtractResponse(
            source_text=result.source_text,
            drugs=result.drugs,
            adverse_events=result.adverse_events,
            symptoms=result.symptoms,
            severity=result.severity,
            confidence_score=result.confidence_score,
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.error("NLP extraction error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/nlp/process-pending", tags=["NLP Extraction"])
async def process_pending_reports(
    request: NLPBatchRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger NLP processing of all unprocessed reports in the database."""
    service = NLPExtractionService(mode=request.mode)
    background_tasks.add_task(service.process_pending_reports, limit=request.limit)
    return {"status": "processing", "message": "NLP batch processing started in background."}


@app.get("/api/v1/nlp/stats", tags=["NLP Extraction"])
async def nlp_stats():
    """Return NLP processing statistics."""
    service = NLPExtractionService()
    return service.get_extraction_stats()


# ---------------------------------------------------------------------------
# Risk Signal Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/signals/detect", tags=["Risk Signals"])
async def detect_signals(
    request: SignalDetectionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full risk signal detection pipeline.
    Runs PRR, ROR, IC, chi-square, anomaly detection, and trend analysis.
    """
    service = RiskSignalDetectionService(
        prr_threshold=request.prr_threshold,
        min_reports=request.min_reports,
        contamination=request.contamination,
    )
    background_tasks.add_task(service.run_full_detection)
    return {
        "status": "processing",
        "message": "Risk signal detection started in background.",
    }


@app.post("/api/v1/signals/detect/sync", tags=["Risk Signals"])
async def detect_signals_sync(request: SignalDetectionRequest):
    """Run risk signal detection synchronously and return results."""
    service = RiskSignalDetectionService(
        prr_threshold=request.prr_threshold,
        min_reports=request.min_reports,
        contamination=request.contamination,
    )
    result = service.run_full_detection()
    return result


@app.get("/api/v1/signals", tags=["Risk Signals"])
async def list_signals(
    limit: int = Query(100, ge=1, le=1000),
    drug_name: Optional[str] = Query(None),
    db=Depends(get_db),
):
    """List detected risk signals, optionally filtered by drug."""
    query = db.query(RiskSignal).order_by(RiskSignal.severity_score.desc())

    if drug_name:
        query = query.filter(RiskSignal.drug_name.ilike(f"%{drug_name}%"))

    signals = query.limit(limit).all()
    return {
        "total": db.query(RiskSignal).count(),
        "items": [RiskSignalSchema.model_validate(s) for s in signals],
    }


@app.get("/api/v1/signals/stats", tags=["Risk Signals"])
async def signal_stats():
    """Return risk signal summary statistics."""
    service = RiskSignalDetectionService()
    return service.get_summary_stats()


@app.get("/api/v1/signals/{drug_name}", tags=["Risk Signals"])
async def signals_for_drug(drug_name: str):
    """Get all risk signals for a specific drug."""
    service = RiskSignalDetectionService()
    return service.get_signals_for_drug(drug_name)


# ---------------------------------------------------------------------------
# Reporting Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/reports/generate", tags=["Reporting"])
async def generate_report(request: ReportGenerationRequest):
    """
    Generate an aggregate safety report in the specified format.
    Returns JSON data or file bytes (PDF/Excel).
    """
    service = ReportingService()
    fmt = request.format.lower()

    try:
        if fmt == "json":
            data = service.generate_report("json")
            return JSONResponse(content=data)

        elif fmt == "excel":
            excel_data = service.generate_report("excel")
            if not isinstance(excel_data, bytes):
                raise HTTPException(status_code=500, detail="Excel generation failed to return bytes.")
            return StreamingResponse(
                BytesIO(excel_data),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=safety_report_{datetime.utcnow().strftime('%Y%m%d')}.xlsx"},
            )

        elif fmt == "pdf":
            pdf_data = service.generate_report("pdf")
            if not isinstance(pdf_data, bytes) or not pdf_data:
                raise HTTPException(status_code=500, detail="PDF generation failed. Ensure reportlab is installed.")
            return StreamingResponse(
                BytesIO(pdf_data),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=safety_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"},
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use pdf, excel, or json.")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report generation error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# AI Assistant Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/assistant/chat", response_model=AIQueryResponse, tags=["AI Assistant"])
async def chat(request: AIQueryRequest):
    """
    Chat with the PharmAI assistant.

    Example questions:
    - "What are the most common adverse events for Metformin?"
    - "Show safety signals detected in the dataset"
    - "How many reports of nausea?"
    """
    session_id = request.session_id or "default"
    assistant = get_assistant(session_id)

    try:
        result = assistant.chat(request.question)
        return AIQueryResponse(
            answer=result["answer"],
            intent=result.get("intent"),
            confidence=float(result.get("confidence", 0.0)),
            data=result.get("data", []),
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("AI assistant error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/assistant/history/{session_id}", tags=["AI Assistant"])
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    assistant = get_assistant(session_id)
    return {
        "session_id": session_id,
        "messages": assistant.get_history(),
    }


@app.delete("/api/v1/assistant/history/{session_id}", tags=["AI Assistant"])
async def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _assistant_sessions:
        _assistant_sessions[session_id].clear_history()
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/v1/assistant/suggested-queries", tags=["AI Assistant"])
async def suggested_queries():
    """Return suggested example queries for the AI assistant."""
    assistant = AIAssistantService()
    return {"queries": assistant.get_suggested_queries()}


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/analytics/severity-distribution", tags=["Analytics"])
async def severity_distribution(db=Depends(get_db)):
    """Return adverse event report count by severity level."""
    from sqlalchemy import func

    results = (
        db.query(AdverseEventReport.severity, func.count(AdverseEventReport.id).label("count"))
        .filter(AdverseEventReport.is_duplicate == False)
        .group_by(AdverseEventReport.severity)
        .all()
    )
    return [{"severity": r[0].value if hasattr(r[0], "value") else str(r[0]), "count": r[1]} for r in results]


@app.get("/api/v1/analytics/gender-distribution", tags=["Analytics"])
async def gender_distribution(db=Depends(get_db)):
    """Return adverse event report count by gender."""
    from sqlalchemy import func

    results = (
        db.query(AdverseEventReport.gender, func.count(AdverseEventReport.id).label("count"))
        .filter(AdverseEventReport.is_duplicate == False)
        .group_by(AdverseEventReport.gender)
        .all()
    )
    return [{"gender": r[0].value if hasattr(r[0], "value") else str(r[0]), "count": r[1]} for r in results]


@app.get("/api/v1/analytics/age-group-distribution", tags=["Analytics"])
async def age_group_distribution(db=Depends(get_db)):
    """Return report count by patient age group."""
    from sqlalchemy import func

    results = (
        db.query(AdverseEventReport.patient_age_group, func.count(AdverseEventReport.id).label("count"))
        .filter(AdverseEventReport.is_duplicate == False)
        .group_by(AdverseEventReport.patient_age_group)
        .all()
    )
    return [{"age_group": r[0], "count": r[1]} for r in results]


@app.get("/api/v1/analytics/drug-class-distribution", tags=["Analytics"])
async def drug_class_distribution(db=Depends(get_db)):
    """Return report count by drug class."""
    from sqlalchemy import func

    results = (
        db.query(AdverseEventReport.drug_class, func.count(AdverseEventReport.id).label("count"))
        .filter(AdverseEventReport.is_duplicate == False, AdverseEventReport.drug_class.isnot(None))
        .group_by(AdverseEventReport.drug_class)
        .order_by(func.count(AdverseEventReport.id).desc())
        .all()
    )
    return [{"drug_class": r[0], "count": r[1]} for r in results]


# ---------------------------------------------------------------------------
# Signal Management Endpoints
# ---------------------------------------------------------------------------

@app.patch("/api/v1/signals/{signal_id}", response_model=SignalStatusUpdateResponse, tags=["Risk Signals"])
async def update_signal_status(
    signal_id: str,
    request: SignalStatusUpdateRequest,
    db=Depends(get_db),
):
    """Update the status of a risk signal (JD requirement for workflow)."""
    signal = db.query(RiskSignal).filter(RiskSignal.signal_id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

    old_status = signal.status.value if hasattr(signal.status, "value") else str(signal.status)
    try:
        signal.status = SignalStatus(request.status.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")

    if request.reviewer_notes:
        signal.notes = (signal.notes or "") + f"\n[{datetime.utcnow()}] {request.reviewer_notes}"

    db.commit()

    # Log action
    audit = AuditLog(
        action="update_signal_status",
        entity_type="RiskSignal",
        entity_id=signal_id,
        user=request.reviewed_by or "system",
        details={"old_status": old_status, "new_status": request.status},
    )
    db.add(audit)
    db.commit()

    return SignalStatusUpdateResponse(
        signal_id=signal_id,
        previous_status=old_status,
        new_status=request.status,
        updated_at=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Audit Log Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/audit-logs", response_model=AuditLogListResponse, tags=["Compliance"])
async def list_audit_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    action: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    db=Depends(get_db),
):
    """Retrieve audit logs for compliance monitoring (JD requirement)."""
    query = db.query(AuditLog).order_by(AuditLog.timestamp.desc())

    if action:
        query = query.filter(AuditLog.action == action)
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type)

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return AuditLogListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[AuditLogSchema.model_validate(log) for log in items],
    )


# ---------------------------------------------------------------------------
# Drug Registry Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/drugs", response_model=DrugListResponse, tags=["Drug Registry"])
async def list_drugs(db=Depends(get_db)):
    """List all registered drugs."""
    drugs = db.query(Drug).all()
    return DrugListResponse(
        total=len(drugs),
        items=[DrugSchema.model_validate(d) for d in drugs],
    )


@app.post("/api/v1/drugs", response_model=DrugSchema, tags=["Drug Registry"])
async def create_drug(request: DrugCreateRequest, db=Depends(get_db)):
    """Add a new drug to the registry."""
    drug = Drug(**request.model_dump())
    db.add(drug)
    try:
        db.commit()
        db.refresh(drug)
        return DrugSchema.model_validate(drug)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# HA Submission Tracking Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/ha-submissions", response_model=HASubmissionListResponse, tags=["HA Tracker"])
async def list_ha_submissions(db=Depends(get_db)):
    """List all Health Authority AR&RM submissions (JD requirement)."""
    subs = db.query(HASubmission).order_by(HASubmission.due_date.asc()).all()
    return HASubmissionListResponse(
        total=len(subs),
        items=[HASubmissionSchema.model_validate(s) for s in subs],
    )


@app.post("/api/v1/ha-submissions", response_model=HASubmissionSchema, tags=["HA Tracker"])
async def create_ha_submission(request: HASubmissionCreateRequest, db=Depends(get_db)):
    """Track a new Health Authority submission."""
    submission_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
    sub = HASubmission(
        submission_id=submission_id,
        **request.model_dump()
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return HASubmissionSchema.model_validate(sub)


# ---------------------------------------------------------------------------
# CAPA / Quality Incident Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/capas", response_model=CAPAListResponse, tags=["CAPA"])
async def list_capas(db=Depends(get_db)):
    """List all RCA/CAPA & Quality Incidents (JD requirement)."""
    capas = db.query(CAPA).order_by(CAPA.opened_date.desc()).all()
    return CAPAListResponse(
        total=len(capas),
        items=[CAPASchema.model_validate(c) for c in capas],
    )


@app.post("/api/v1/capas", response_model=CAPASchema, tags=["CAPA"])
async def create_capa(request: CAPACreateRequest, db=Depends(get_db)):
    """Create a new CAPA or Quality Incident record."""
    capa_id = f"CAPA-{uuid.uuid4().hex[:8].upper()}"
    capa = CAPA(
        capa_id=capa_id,
        opened_date=date.today(),
        **request.model_dump()
    )
    db.add(capa)
    db.commit()
    db.refresh(capa)
    return CAPASchema.model_validate(capa)


# ---------------------------------------------------------------------------
# Compliance Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/compliance/dashboard", response_model=ComplianceDashboardResponse, tags=["Compliance"])
async def get_compliance_dashboard(service: ComplianceService = Depends(ComplianceService)):
    """Retrieve compliance KPIs and operational metrics (JD requirement)."""
    return service.get_dashboard_data()

