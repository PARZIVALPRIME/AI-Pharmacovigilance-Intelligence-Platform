"""
Database Models — AI Pharmacovigilance Intelligence Platform

SQLAlchemy ORM model definitions covering the full pharmacovigilance
data domain: adverse event reports, drugs, risk signals, analysis
results, and audit trails.
"""

from __future__ import annotations

import enum
from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Declarative base with auto-updated timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SeverityLevel(str, enum.Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class OutcomeType(str, enum.Enum):
    RECOVERED = "recovered"
    RECOVERING = "recovering"
    NOT_RECOVERED = "not_recovered"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class ReportStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FLAGGED = "flagged"
    ARCHIVED = "archived"


class SignalStatus(str, enum.Enum):
    DETECTED = "detected"
    UNDER_REVIEW = "under_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CLOSED = "closed"


class GenderType(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClinicalPhase(str, enum.Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    POST_MARKET = "post_market"
    UNKNOWN = "unknown"


class CausalityLevel(str, enum.Enum):
    """WHO-UMC causality assessment scale."""
    CERTAIN = "certain"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    CONDITIONAL = "conditional"
    UNASSESSABLE = "unassessable"


class CAPAStatus(str, enum.Enum):
    """Corrective and Preventive Action lifecycle states."""
    OPEN = "open"
    INVESTIGATION = "investigation"
    ACTION_PLANNED = "action_planned"
    ACTION_IN_PROGRESS = "action_in_progress"
    VERIFICATION = "verification"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class HASubmissionStatus(str, enum.Enum):
    """Health Authority submission tracking states."""
    PLANNED = "planned"
    IN_PREPARATION = "in_preparation"
    UNDER_QC = "under_qc"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    OVERDUE = "overdue"


class AggregateReportType(str, enum.Enum):
    """Aggregate report types per ICH guidelines."""
    PSUR = "psur"
    PBRER = "pbrer"
    DSUR = "dsur"
    PADER = "pader"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Drug(Base):
    """Master drug registry."""

    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    generic_name: Mapped[Optional[str]] = mapped_column(String(255))
    brand_name: Mapped[Optional[str]] = mapped_column(String(255))
    drug_class: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    atc_code: Mapped[Optional[str]] = mapped_column(String(20))
    approval_status: Mapped[Optional[str]] = mapped_column(String(50))
    manufacturer: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    adverse_events: Mapped[List["AdverseEventReport"]] = relationship(
        "AdverseEventReport", back_populates="drug", lazy="dynamic"
    )
    risk_signals: Mapped[List["RiskSignal"]] = relationship(
        "RiskSignal", back_populates="drug", lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint("name", "drug_class", name="uq_drug_name_class"),
        Index("ix_drug_name_lower", func.lower(name)),
    )

    def __repr__(self) -> str:
        return f"<Drug id={self.id} name={self.name!r}>"


class AdverseEventReport(Base):
    """Core adverse event case report (ICSR-like structure)."""

    __tablename__ = "adverse_event_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    drug_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drugs.id"), index=True)
    drug_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    adverse_event: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    adverse_event_normalized: Mapped[Optional[str]] = mapped_column(String(500), index=True)
    severity: Mapped[SeverityLevel] = mapped_column(
        Enum(SeverityLevel), default=SeverityLevel.UNKNOWN, index=True
    )
    outcome: Mapped[OutcomeType] = mapped_column(
        Enum(OutcomeType), default=OutcomeType.UNKNOWN
    )
    patient_age: Mapped[Optional[float]] = mapped_column(Float)
    patient_age_group: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    gender: Mapped[GenderType] = mapped_column(
        Enum(GenderType), default=GenderType.UNKNOWN, index=True
    )
    country: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    region: Mapped[Optional[str]] = mapped_column(String(100))
    report_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    receipt_date: Mapped[Optional[date]] = mapped_column(Date)
    clinical_phase: Mapped[ClinicalPhase] = mapped_column(
        Enum(ClinicalPhase), default=ClinicalPhase.UNKNOWN
    )
    drug_class: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    source_text: Mapped[Optional[str]] = mapped_column(Text)
    source_type: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus), default=ReportStatus.PENDING, index=True
    )
    is_serious: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    nlp_processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # Relationships
    drug: Mapped[Optional["Drug"]] = relationship("Drug", back_populates="adverse_events")
    nlp_extractions: Mapped[List["NLPExtraction"]] = relationship(
        "NLPExtraction", back_populates="report", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_aer_drug_event", "drug_name", "adverse_event"),
        Index("ix_aer_date_severity", "report_date", "severity"),
        Index("ix_aer_country_drug", "country", "drug_name"),
    )

    def __repr__(self) -> str:
        return f"<AdverseEventReport id={self.id} drug={self.drug_name!r} event={self.adverse_event!r}>"


class NLPExtraction(Base):
    """Results from NLP processing of free-text reports."""

    __tablename__ = "nlp_extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("adverse_event_reports.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    extracted_drugs: Mapped[Optional[list]] = mapped_column(JSON)
    extracted_events: Mapped[Optional[list]] = mapped_column(JSON)
    extracted_symptoms: Mapped[Optional[list]] = mapped_column(JSON)
    extracted_severity: Mapped[Optional[str]] = mapped_column(String(50))
    entities_raw: Mapped[Optional[dict]] = mapped_column(JSON)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)

    # Relationship
    report: Mapped["AdverseEventReport"] = relationship(
        "AdverseEventReport", back_populates="nlp_extractions"
    )

    def __repr__(self) -> str:
        return f"<NLPExtraction id={self.id} report_id={self.report_id}>"


class RiskSignal(Base):
    """Detected pharmacovigilance risk signals."""

    __tablename__ = "risk_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    drug_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drugs.id"), index=True)
    drug_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    adverse_event: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    signal_type: Mapped[str] = mapped_column(String(100), index=True)
    # Disproportionality measures
    prr: Mapped[Optional[float]] = mapped_column(Float)          # Proportional Reporting Ratio
    prr_lower_ci: Mapped[Optional[float]] = mapped_column(Float)
    prr_upper_ci: Mapped[Optional[float]] = mapped_column(Float)
    ror: Mapped[Optional[float]] = mapped_column(Float)          # Reporting Odds Ratio
    ror_lower_ci: Mapped[Optional[float]] = mapped_column(Float)
    ror_upper_ci: Mapped[Optional[float]] = mapped_column(Float)
    ic: Mapped[Optional[float]] = mapped_column(Float)           # Information Component
    eb05: Mapped[Optional[float]] = mapped_column(Float)         # Empirical Bayes 5th percentile
    chi_square: Mapped[Optional[float]] = mapped_column(Float)
    p_value: Mapped[Optional[float]] = mapped_column(Float)
    # Counts
    report_count: Mapped[int] = mapped_column(Integer, default=0)
    expected_count: Mapped[Optional[float]] = mapped_column(Float)
    # Metadata
    detection_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    status: Mapped[SignalStatus] = mapped_column(
        Enum(SignalStatus), default=SignalStatus.DETECTED, index=True
    )
    severity_score: Mapped[Optional[float]] = mapped_column(Float)
    is_new: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)

    # Relationships
    drug: Mapped[Optional["Drug"]] = relationship("Drug", back_populates="risk_signals")

    __table_args__ = (
        UniqueConstraint("drug_name", "adverse_event", name="uq_signal_drug_event"),
        Index("ix_signal_drug_event", "drug_name", "adverse_event"),
        Index("ix_signal_date_status", "detection_date", "status"),
    )

    def __repr__(self) -> str:
        return f"<RiskSignal id={self.id} drug={self.drug_name!r} event={self.adverse_event!r} prr={self.prr}>"


class AggregateAnalysis(Base):
    """Pre-computed aggregate analysis results for dashboard performance."""

    __tablename__ = "aggregate_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    analysis_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    dimension: Mapped[Optional[str]] = mapped_column(String(100))
    dimension_value: Mapped[Optional[str]] = mapped_column(String(255))
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[Optional[float]] = mapped_column(Float)
    metric_data: Mapped[Optional[dict]] = mapped_column(JSON)
    record_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<AggregateAnalysis type={self.analysis_type} date={self.analysis_date}>"


class Report(Base):
    """Generated safety reports metadata."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_name: Mapped[str] = mapped_column(String(255), nullable=False)
    report_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    generation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    parameters: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="completed")
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    generated_by: Mapped[Optional[str]] = mapped_column(String(100))

    def __repr__(self) -> str:
        return f"<Report id={self.id} name={self.report_name!r} type={self.report_type}>"


class AuditLog(Base):
    """Audit trail for all system actions — required for regulatory compliance."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    entity_type: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    entity_id: Mapped[Optional[str]] = mapped_column(String(100))
    user: Mapped[Optional[str]] = mapped_column(String(100))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action} entity={self.entity_type}>"


class CausalityAssessment(Base):
    """WHO-UMC causality assessment for drug-adverse event pairs."""

    __tablename__ = "causality_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("adverse_event_reports.id"), index=True)
    drug_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    adverse_event: Mapped[str] = mapped_column(String(500), nullable=False)
    causality_level: Mapped[CausalityLevel] = mapped_column(
        Enum(CausalityLevel), default=CausalityLevel.UNASSESSABLE, index=True
    )
    assessor: Mapped[Optional[str]] = mapped_column(String(100))
    assessment_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    temporal_relationship: Mapped[Optional[bool]] = mapped_column(Boolean)
    dechallenge_positive: Mapped[Optional[bool]] = mapped_column(Boolean)
    rechallenge_positive: Mapped[Optional[bool]] = mapped_column(Boolean)
    alternative_causes_excluded: Mapped[Optional[bool]] = mapped_column(Boolean)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_causality_drug_event", "drug_name", "adverse_event"),
    )

    def __repr__(self) -> str:
        return f"<CausalityAssessment id={self.id} drug={self.drug_name!r} level={self.causality_level}>"


class HASubmission(Base):
    """Health Authority Aggregate Report & Risk Management (AR&RM) submission tracker."""

    __tablename__ = "ha_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    report_type: Mapped[AggregateReportType] = mapped_column(
        Enum(AggregateReportType), nullable=False, index=True
    )
    health_authority: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    country: Mapped[Optional[str]] = mapped_column(String(100))
    region: Mapped[Optional[str]] = mapped_column(String(100))
    data_lock_point: Mapped[Optional[date]] = mapped_column(Date)
    due_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    submission_date: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[HASubmissionStatus] = mapped_column(
        Enum(HASubmissionStatus), default=HASubmissionStatus.PLANNED, index=True
    )
    assigned_to: Mapped[Optional[str]] = mapped_column(String(100))
    qc_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    qc_date: Mapped[Optional[date]] = mapped_column(Date)
    qc_by: Mapped[Optional[str]] = mapped_column(String(100))
    reference_number: Mapped[Optional[str]] = mapped_column(String(100))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_ha_product_authority", "product_name", "health_authority"),
        Index("ix_ha_due_date_status", "due_date", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<HASubmission id={self.id} product={self.product_name!r} "
            f"type={self.report_type} ha={self.health_authority!r} status={self.status}>"
        )


class CAPA(Base):
    """Corrective and Preventive Action (CAPA) / Quality Incident tracking.

    Supports the JD requirement: 'preparation, follow-up and closure of
    RCA/CAPA & Quality Incidents'.
    """

    __tablename__ = "capas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capa_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # RCA fields
    root_cause: Mapped[Optional[str]] = mapped_column(Text)
    rca_method: Mapped[Optional[str]] = mapped_column(String(100))  # e.g. 5-Why, Fishbone
    rca_completed_date: Mapped[Optional[date]] = mapped_column(Date)
    # Corrective action
    corrective_action: Mapped[Optional[str]] = mapped_column(Text)
    corrective_action_owner: Mapped[Optional[str]] = mapped_column(String(100))
    corrective_action_due: Mapped[Optional[date]] = mapped_column(Date)
    corrective_action_completed: Mapped[Optional[date]] = mapped_column(Date)
    # Preventive action
    preventive_action: Mapped[Optional[str]] = mapped_column(Text)
    preventive_action_owner: Mapped[Optional[str]] = mapped_column(String(100))
    preventive_action_due: Mapped[Optional[date]] = mapped_column(Date)
    preventive_action_completed: Mapped[Optional[date]] = mapped_column(Date)
    # Lifecycle
    status: Mapped[CAPAStatus] = mapped_column(
        Enum(CAPAStatus), default=CAPAStatus.OPEN, index=True
    )
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    opened_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    target_closure_date: Mapped[Optional[date]] = mapped_column(Date)
    actual_closure_date: Mapped[Optional[date]] = mapped_column(Date)
    opened_by: Mapped[Optional[str]] = mapped_column(String(100))
    closed_by: Mapped[Optional[str]] = mapped_column(String(100))
    # Related entities
    related_signal_id: Mapped[Optional[str]] = mapped_column(String(50))
    related_submission_id: Mapped[Optional[str]] = mapped_column(String(50))
    # Verification
    effectiveness_check: Mapped[Optional[str]] = mapped_column(Text)
    effectiveness_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_date: Mapped[Optional[date]] = mapped_column(Date)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        Index("ix_capa_status_priority", "status", "priority"),
    )

    def __repr__(self) -> str:
        return f"<CAPA id={self.id} capa_id={self.capa_id!r} status={self.status}>"
