"""
API Gateway — Pydantic Schemas
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Common schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    database: dict
    uptime_seconds: float


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=500, description="Records per page")


class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


# ---------------------------------------------------------------------------
# Adverse Event schemas
# ---------------------------------------------------------------------------

class AdverseEventReportSchema(BaseModel):
    id: Optional[int] = None
    report_id: Optional[str] = None
    drug_name: str
    adverse_event: str
    severity: Optional[str] = None
    outcome: Optional[str] = None
    patient_age: Optional[float] = None
    patient_age_group: Optional[str] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    report_date: Optional[date] = None
    clinical_phase: Optional[str] = None
    drug_class: Optional[str] = None
    is_serious: bool = False
    confidence_score: Optional[float] = None
    source_text: Optional[str] = None

    class Config:
        from_attributes = True


class AdverseEventListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[AdverseEventReportSchema]


class AdverseEventFilterParams(BaseModel):
    drug_name: Optional[str] = None
    adverse_event: Optional[str] = None
    severity: Optional[str] = None
    country: Optional[str] = None
    is_serious: Optional[bool] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    drug_class: Optional[str] = None
    clinical_phase: Optional[str] = None


# ---------------------------------------------------------------------------
# NLP schemas
# ---------------------------------------------------------------------------

class NLPExtractRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="Medical text to analyse")
    mode: str = Field("rule_based", description="Extraction mode: rule_based, transformer, ensemble")


class NLPExtractResponse(BaseModel):
    source_text: str
    drugs: List[str]
    adverse_events: List[str]
    symptoms: List[str]
    severity: Optional[str]
    confidence_score: float
    model_used: str
    processing_time_ms: float


class NLPBatchRequest(BaseModel):
    limit: Optional[int] = Field(None, ge=1, le=10000)
    mode: str = "rule_based"


# ---------------------------------------------------------------------------
# Risk Signal schemas
# ---------------------------------------------------------------------------

class RiskSignalSchema(BaseModel):
    id: Optional[int] = None
    signal_id: Optional[str] = None
    drug_name: str
    adverse_event: str
    signal_type: Optional[str] = None
    prr: Optional[float] = None
    prr_lower_ci: Optional[float] = None
    prr_upper_ci: Optional[float] = None
    ror: Optional[float] = None
    ic: Optional[float] = None
    eb05: Optional[float] = None
    chi_square: Optional[float] = None
    p_value: Optional[float] = None
    report_count: int
    expected_count: Optional[float] = None
    severity_score: Optional[float] = None
    status: Optional[str] = None
    detection_date: Optional[date] = None
    is_new: bool = True

    class Config:
        from_attributes = True


class SignalDetectionRequest(BaseModel):
    prr_threshold: float = Field(2.0, ge=1.0, le=100.0)
    min_reports: int = Field(3, ge=1)
    contamination: float = Field(0.05, ge=0.01, le=0.5)


class SignalListResponse(BaseModel):
    total: int
    items: List[RiskSignalSchema]


# ---------------------------------------------------------------------------
# Reporting schemas
# ---------------------------------------------------------------------------

class ReportGenerationRequest(BaseModel):
    format: str = Field("json", description="Output format: pdf, excel, json")
    include_signals: bool = True
    include_trends: bool = True
    top_n_events: int = Field(20, ge=5, le=100)


class ReportGenerationResponse(BaseModel):
    report_id: str
    format: str
    status: str
    file_path: Optional[str] = None
    generated_at: str
    data: Optional[dict] = None


# ---------------------------------------------------------------------------
# AI Assistant schemas
# ---------------------------------------------------------------------------

class AIQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    session_id: Optional[str] = None


class AIQueryResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    confidence: float
    data: Any
    timestamp: str
    session_id: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]


# ---------------------------------------------------------------------------
# Analytics schemas
# ---------------------------------------------------------------------------

class DrugSafetyProfileSchema(BaseModel):
    drug_name: str
    drug_class: Optional[str]
    total_reports: int
    serious_reports: int
    seriousness_rate: float


class TopAdverseEventSchema(BaseModel):
    adverse_event: str
    count: int


class GeographicDistributionSchema(BaseModel):
    country: str
    region: Optional[str]
    total: int
    serious: int
    seriousness_rate: float


class MonthlyTrendSchema(BaseModel):
    year_month: str
    report_count: int


class AnalyticsSummaryResponse(BaseModel):
    total_reports: int
    serious_reports: int
    seriousness_rate: float
    total_signals: int
    unique_drugs: int
    unique_adverse_events: int
    date_range: Dict[str, Optional[str]]


# ---------------------------------------------------------------------------
# Ingestion schemas
# ---------------------------------------------------------------------------

class IngestionRequest(BaseModel):
    n_records: int = Field(10_000, ge=100, le=1_000_000)
    force_regenerate: bool = False


class IngestionResponse(BaseModel):
    status: str
    raw_records: int
    clean_records: int
    inserted: int
    skipped: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Signal Status Update schemas
# ---------------------------------------------------------------------------

class SignalStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New status: detected, under_review, confirmed, rejected, closed")
    reviewer_notes: Optional[str] = Field(None, max_length=2000)
    reviewed_by: Optional[str] = Field(None, max_length=100)


class SignalStatusUpdateResponse(BaseModel):
    signal_id: str
    previous_status: str
    new_status: str
    updated_at: str


# ---------------------------------------------------------------------------
# Audit Log schemas
# ---------------------------------------------------------------------------

class AuditLogSchema(BaseModel):
    id: Optional[int] = None
    timestamp: Optional[str] = None
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    user: Optional[str] = None
    ip_address: Optional[str] = None
    details: Optional[Dict] = None
    success: bool = True
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[AuditLogSchema]


# ---------------------------------------------------------------------------
# Drug CRUD schemas
# ---------------------------------------------------------------------------

class DrugCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    drug_class: Optional[str] = None
    atc_code: Optional[str] = None
    approval_status: Optional[str] = None
    manufacturer: Optional[str] = None


class DrugSchema(BaseModel):
    id: Optional[int] = None
    name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    drug_class: Optional[str] = None
    atc_code: Optional[str] = None
    approval_status: Optional[str] = None
    manufacturer: Optional[str] = None
    is_active: bool = True

    class Config:
        from_attributes = True


class DrugListResponse(BaseModel):
    total: int
    items: List[DrugSchema]


# ---------------------------------------------------------------------------
# Compliance / Metrics schemas
# ---------------------------------------------------------------------------

class ComplianceMetricSchema(BaseModel):
    metric_name: str
    metric_value: float
    target_value: Optional[float] = None
    unit: Optional[str] = None
    status: str  # on_track, at_risk, overdue
    period: Optional[str] = None


class ComplianceDashboardResponse(BaseModel):
    overall_score: float
    metrics: List[ComplianceMetricSchema]
    generated_at: str


# ---------------------------------------------------------------------------
# HA Submission Tracker schemas
# ---------------------------------------------------------------------------

class HASubmissionSchema(BaseModel):
    id: Optional[int] = None
    submission_id: Optional[str] = None
    product_name: str
    report_type: str
    health_authority: str
    country: Optional[str] = None
    region: Optional[str] = None
    data_lock_point: Optional[date] = None
    due_date: date
    submission_date: Optional[date] = None
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    qc_completed: bool = False
    qc_date: Optional[date] = None
    qc_by: Optional[str] = None
    reference_number: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class HASubmissionCreateRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=255)
    report_type: str = Field(..., description="psur, pbrer, dsur, pader, other")
    health_authority: str = Field(..., min_length=1, max_length=255)
    country: Optional[str] = None
    region: Optional[str] = None
    data_lock_point: Optional[date] = None
    due_date: date
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


class HASubmissionListResponse(BaseModel):
    total: int
    items: List[HASubmissionSchema]


# ---------------------------------------------------------------------------
# CAPA schemas
# ---------------------------------------------------------------------------

class CAPASchema(BaseModel):
    id: Optional[int] = None
    capa_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    category: str
    root_cause: Optional[str] = None
    rca_method: Optional[str] = None
    corrective_action: Optional[str] = None
    corrective_action_owner: Optional[str] = None
    corrective_action_due: Optional[date] = None
    preventive_action: Optional[str] = None
    preventive_action_owner: Optional[str] = None
    preventive_action_due: Optional[date] = None
    status: Optional[str] = None
    priority: str = "medium"
    opened_date: Optional[date] = None
    target_closure_date: Optional[date] = None
    actual_closure_date: Optional[date] = None
    opened_by: Optional[str] = None
    related_signal_id: Optional[str] = None
    related_submission_id: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class CAPACreateRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=500)
    description: Optional[str] = None
    category: str = Field(..., min_length=1, max_length=100)
    priority: str = Field("medium", description="low, medium, high, critical")
    opened_by: Optional[str] = None
    target_closure_date: Optional[date] = None
    related_signal_id: Optional[str] = None
    related_submission_id: Optional[str] = None
    notes: Optional[str] = None


class CAPAListResponse(BaseModel):
    total: int
    items: List[CAPASchema]
