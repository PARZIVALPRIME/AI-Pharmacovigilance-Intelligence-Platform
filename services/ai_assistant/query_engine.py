"""
AI Assistant Service — Query Engine
AI Pharmacovigilance Intelligence Platform

Provides intelligent natural language querying of pharmacovigilance data
using a rule-based query engine with optional LangChain/OpenAI integration.

The query engine can work in two modes:
  1. Local mode: SQL-based retrieval + structured response generation (no API key needed)
  2. LLM mode: LangChain + OpenAI for natural language understanding and generation

This design ensures the system works out-of-the-box without an OpenAI key,
while supporting full LLM capabilities when a key is provided.
"""

from __future__ import annotations

import re
from datetime import datetime
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
from database.models import (
    AdverseEventReport,
    RiskSignal,
    Drug,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Local Query Engine (no LLM required)
# ---------------------------------------------------------------------------

class LocalQueryEngine:
    """
    Rule-based pharmacovigilance query engine that understands common
    question patterns and retrieves structured answers from the database.

    Supports queries like:
      - "What are the most common adverse events for Metformin?"
      - "Show risk signals for Warfarin"
      - "How many reports of nausea are there?"
      - "List the top 10 drugs by report count"
      - "What is the seriousness rate for Lisinopril?"
    """

    # Intent patterns
    INTENT_PATTERNS = {
        "adverse_events_for_drug": [
            r"adverse\s+events?\s+for\s+(?P<drug>[\w\s]+)",
            r"side\s+effects?\s+of\s+(?P<drug>[\w\s]+)",
            r"reactions?\s+(?:to|for|with)\s+(?P<drug>[\w\s]+)",
            r"events?\s+(?:associated\s+with|related\s+to)\s+(?P<drug>[\w\s]+)",
            r"what\s+(?:happens?|are\s+the\s+effects?)\s+(?:when\s+taking\s+)?(?P<drug>[\w\s]+)",
        ],
        "signals_for_drug": [
            r"(?:risk\s+)?signals?\s+(?:for|of|with)\s+(?P<drug>[\w\s]+)",
            r"safety\s+signals?\s+(?:for|of)\s+(?P<drug>[\w\s]+)",
            r"alerts?\s+(?:for|about)\s+(?P<drug>[\w\s]+)",
        ],
        "event_count": [
            r"how\s+many\s+(?:reports?\s+of\s+)?(?P<event>[\w\s]+)",
            r"count\s+of\s+(?P<event>[\w\s]+)",
            r"frequency\s+of\s+(?P<event>[\w\s]+)",
        ],
        "top_drugs": [
            r"top\s+(?P<n>\d+)?\s*drugs?",
            r"most\s+reported\s+drugs?",
            r"drugs?\s+with\s+most\s+reports?",
        ],
        "top_events": [
            r"top\s+(?P<n>\d+)?\s*adverse\s+events?",
            r"most\s+(?:common|frequent)\s+adverse\s+events?",
            r"top\s+(?P<n>\d+)?\s*events?",
        ],
        "all_signals": [
            r"all\s+(?:risk\s+)?signals?",
            r"show\s+(?:me\s+)?(?:all\s+)?(?:risk\s+)?signals?",
            r"detected\s+signals?",
            r"list\s+signals?",
        ],
        "seriousness_rate": [
            r"seriousness\s+rate\s+(?:for|of)\s+(?P<drug>[\w\s]+)",
            r"how\s+serious\s+is\s+(?P<drug>[\w\s]+)",
        ],
        "drug_class_events": [
            r"(?:adverse\s+)?events?\s+(?:for|in)\s+(?P<drug_class>[\w\s]+)\s+class",
            r"(?P<drug_class>[\w\s]+)\s+class\s+(?:adverse\s+)?events?",
        ],
        "date_range_analysis": [
            r"reports?\s+(?:from|between|since|in)\s+(?P<date_expr>.+)",
            r"what\s+happened\s+(?:in|during|between)\s+(?P<date_expr>.+)",
        ],
        "geographic_analysis": [
            r"(?:reports?|events?|signals?)\s+(?:in|from|for)\s+(?P<location>[\w\s]+)(?:\s+country)?",
            r"(?:country|region|geographic)\s+(?:analysis|breakdown|distribution)(?:\s+for\s+(?P<location>[\w\s]+))?",
        ],
        "drug_comparison": [
            r"compare\s+(?P<drug_a>[\w]+)\s+(?:and|vs\.?|versus|with)\s+(?P<drug_b>[\w]+)",
            r"(?P<drug_a>[\w]+)\s+vs\.?\s+(?P<drug_b>[\w]+)",
        ],
        "summary": [
            r"summar(?:y|ise|ize)",
            r"overview",
            r"statistics?\s+(?:summary)?",
            r"how\s+many\s+(?:total\s+)?reports?",
        ],
    }

    def query(self, question: str) -> dict:
        """
        Process a natural language question and return a structured answer.

        Returns
        -------
        dict with keys: intent, answer, data, confidence
        """
        question_clean = question.strip().lower()
        logger.debug("Processing query: {!r}", question_clean)

        # Detect intent
        intent, params = self._detect_intent(question_clean)
        logger.debug("Detected intent: {} with params: {}", intent, params)

        # Route to handler
        handlers = {
            "adverse_events_for_drug": self._handle_adverse_events_for_drug,
            "signals_for_drug": self._handle_signals_for_drug,
            "event_count": self._handle_event_count,
            "top_drugs": self._handle_top_drugs,
            "top_events": self._handle_top_events,
            "all_signals": self._handle_all_signals,
            "seriousness_rate": self._handle_seriousness_rate,
            "drug_class_events": self._handle_drug_class_events,
            "date_range_analysis": self._handle_date_range_analysis,
            "geographic_analysis": self._handle_geographic_analysis,
            "drug_comparison": self._handle_drug_comparison,
            "summary": self._handle_summary,
            "unknown": self._handle_unknown,
        }

        handler = handlers.get(intent, handlers["unknown"])
        result = handler(params, question)
        result["intent"] = intent
        result["question"] = question
        result["timestamp"] = datetime.utcnow().isoformat()
        return result

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, question: str) -> Tuple[str, dict]:
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    params = match.groupdict()
                    # Clean up extracted params
                    params = {k: v.strip() if v else None for k, v in params.items()}
                    return intent, params
        return "unknown", {}

    # ------------------------------------------------------------------
    # Query handlers
    # ------------------------------------------------------------------

    def _handle_adverse_events_for_drug(self, params: dict, _question: str) -> dict:
        drug = params.get("drug", "").strip()
        if not drug:
            return self._error("Could not identify a drug name in your query.")

        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.adverse_event,
                AdverseEventReport.severity,
            ).filter(
                AdverseEventReport.drug_name.ilike(f"%{drug}%"),
                AdverseEventReport.is_duplicate == False,
            ).all()

        if not rows:
            return {
                "answer": f"No adverse event reports found for drug matching '{drug}'.",
                "data": [],
                "confidence": 0.3,
            }

        df = pd.DataFrame(rows, columns=["adverse_event", "severity"])
        top_events = (
            df.groupby("adverse_event")
            .agg(count=("adverse_event", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
            .head(10)
        )

        events_list = top_events.to_dict("records")
        summary_text = ", ".join(
            f"{r['adverse_event']} ({r['count']} reports)" for r in events_list[:5]
        )

        return {
            "answer": (
                f"The most common adverse events for drugs matching '{drug}' are: {summary_text}. "
                f"Total reports analysed: {len(df)}."
            ),
            "data": events_list,
            "confidence": 0.95,
        }

    def _handle_signals_for_drug(self, params: dict, _question: str) -> dict:
        drug = params.get("drug", "").strip()
        if not drug:
            return self._error("Could not identify a drug name.")

        with SessionLocal() as session:
            signals = session.query(RiskSignal).filter(
                RiskSignal.drug_name.ilike(f"%{drug}%")
            ).order_by(RiskSignal.severity_score.desc()).limit(10).all()

        if not signals:
            return {
                "answer": f"No risk signals detected for drug matching '{drug}'.",
                "data": [],
                "confidence": 0.8,
            }

        data = [{
            "adverse_event": s.adverse_event,
            "prr": s.prr,
            "ror": s.ror,
            "report_count": s.report_count,
            "severity_score": s.severity_score,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
        } for s in signals]

        top = data[0]
        answer = (
            f"Found {len(signals)} risk signal(s) for '{drug}'. "
            f"The most significant is '{top['adverse_event']}' with PRR={top.get('prr', 'N/A')}, "
            f"based on {top['report_count']} reports."
        )

        return {"answer": answer, "data": data, "confidence": 0.93}

    def _handle_event_count(self, params: dict, _question: str) -> dict:
        event = params.get("event", "").strip()
        if not event:
            return self._error("Could not identify the adverse event in your query.")

        with SessionLocal() as session:
            count = session.query(AdverseEventReport).filter(
                AdverseEventReport.adverse_event.ilike(f"%{event}%"),
                AdverseEventReport.is_duplicate == False,
            ).count()

        return {
            "answer": f"There are {count:,} report(s) of adverse events matching '{event}'.",
            "data": [{"event": event, "count": count}],
            "confidence": 0.9,
        }

    def _handle_top_drugs(self, params: dict, _question: str) -> dict:
        n = int(params.get("n") or 10)

        with SessionLocal() as session:
            rows = session.query(AdverseEventReport.drug_name).filter(
                AdverseEventReport.is_duplicate == False
            ).all()

        df = pd.DataFrame(rows, columns=["drug_name"])
        top = df["drug_name"].value_counts().head(n).reset_index()
        top.columns = ["drug_name", "report_count"]
        data = top.to_dict("records")

        summary = ", ".join(f"{r['drug_name']} ({r['report_count']})" for r in data[:5])
        return {
            "answer": f"Top {n} most reported drugs: {summary}.",
            "data": data,
            "confidence": 0.95,
        }

    def _handle_top_events(self, params: dict, _question: str) -> dict:
        n = int(params.get("n") or 10)

        with SessionLocal() as session:
            rows = session.query(AdverseEventReport.adverse_event).filter(
                AdverseEventReport.is_duplicate == False
            ).all()

        df = pd.DataFrame(rows, columns=["adverse_event"])
        top = df["adverse_event"].value_counts().head(n).reset_index()
        top.columns = ["adverse_event", "report_count"]
        data = top.to_dict("records")

        summary = ", ".join(f"{r['adverse_event']} ({r['report_count']})" for r in data[:5])
        return {
            "answer": f"Top {n} most common adverse events: {summary}.",
            "data": data,
            "confidence": 0.95,
        }

    def _handle_all_signals(self, _params: dict, _question: str) -> dict:
        with SessionLocal() as session:
            signals = session.query(RiskSignal).order_by(
                RiskSignal.severity_score.desc()
            ).limit(20).all()

        if not signals:
            return {
                "answer": "No risk signals have been detected yet. Run the risk detection pipeline first.",
                "data": [],
                "confidence": 0.8,
            }

        data = [{
            "drug_name": s.drug_name,
            "adverse_event": s.adverse_event,
            "prr": s.prr,
            "severity_score": s.severity_score,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
        } for s in signals]

        return {
            "answer": f"Showing top {len(signals)} detected risk signals, ordered by severity score.",
            "data": data,
            "confidence": 0.95,
        }

    def _handle_seriousness_rate(self, params: dict, _question: str) -> dict:
        drug = params.get("drug", "").strip()
        if not drug:
            return self._error("Could not identify a drug name.")

        with SessionLocal() as session:
            total = session.query(AdverseEventReport).filter(
                AdverseEventReport.drug_name.ilike(f"%{drug}%"),
                AdverseEventReport.is_duplicate == False,
            ).count()
            serious = session.query(AdverseEventReport).filter(
                AdverseEventReport.drug_name.ilike(f"%{drug}%"),
                AdverseEventReport.is_serious == True,
                AdverseEventReport.is_duplicate == False,
            ).count()

        if total == 0:
            return {"answer": f"No reports found for drug matching '{drug}'.", "data": [], "confidence": 0.3}

        rate = serious / total * 100
        return {
            "answer": (
                f"Drug '{drug}' has {serious:,} serious reports out of {total:,} total "
                f"({rate:.1f}% seriousness rate)."
            ),
            "data": [{"drug": drug, "total": total, "serious": serious, "seriousness_rate_pct": round(rate, 2)}],
            "confidence": 0.95,
        }

    def _handle_summary(self, _params: dict, _question: str) -> dict:
        with SessionLocal() as session:
            total = session.query(AdverseEventReport).count()
            serious = session.query(AdverseEventReport).filter(
                AdverseEventReport.is_serious == True
            ).count()
            drugs = session.query(AdverseEventReport.drug_name).distinct().count()
            signals = session.query(RiskSignal).count()

        rate = serious / max(total, 1) * 100
        return {
            "answer": (
                f"Platform summary: {total:,} total adverse event reports, "
                f"{serious:,} serious ({rate:.1f}%), covering {drugs} drugs. "
                f"{signals} risk signals detected."
            ),
            "data": {
                "total_reports": total,
                "serious_reports": serious,
                "seriousness_rate": round(rate, 2),
                "unique_drugs": drugs,
                "total_signals": signals,
            },
            "confidence": 0.99,
        }

    def _handle_drug_class_events(self, params: dict, _question: str) -> dict:
        drug_class = params.get("drug_class", "").strip()
        if not drug_class:
            return self._error("Could not identify a drug class in your query.")

        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.adverse_event,
                AdverseEventReport.drug_name,
                AdverseEventReport.severity,
            ).filter(
                AdverseEventReport.drug_class.ilike(f"%{drug_class}%"),
                AdverseEventReport.is_duplicate == False,
            ).all()

        if not rows:
            return {
                "answer": f"No adverse event reports found for drug class matching '{drug_class}'.",
                "data": [],
                "confidence": 0.3,
            }

        df = pd.DataFrame(rows, columns=["adverse_event", "drug_name", "severity"])
        top_events = (
            df.groupby("adverse_event")
            .agg(count=("adverse_event", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
            .head(10)
        )
        n_drugs = df["drug_name"].nunique()
        events_list = top_events.to_dict("records")
        summary_text = ", ".join(
            f"{r['adverse_event']} ({r['count']} reports)" for r in events_list[:5]
        )

        return {
            "answer": (
                f"For drugs in the '{drug_class}' class ({n_drugs} drugs, {len(df)} reports), "
                f"the most common adverse events are: {summary_text}."
            ),
            "data": events_list,
            "confidence": 0.90,
        }

    def _handle_date_range_analysis(self, params: dict, _question: str) -> dict:
        date_expr = params.get("date_expr", "").strip()

        with SessionLocal() as session:
            rows = session.query(
                AdverseEventReport.report_date,
                AdverseEventReport.drug_name,
                AdverseEventReport.adverse_event,
                AdverseEventReport.is_serious,
            ).filter(
                AdverseEventReport.is_duplicate == False,
                AdverseEventReport.report_date.isnot(None),
            ).all()

        if not rows:
            return {"answer": "No reports with date information found.", "data": [], "confidence": 0.3}

        df = pd.DataFrame(rows, columns=["report_date", "drug_name", "adverse_event", "is_serious"])
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["year_month"] = df["report_date"].dt.to_period("M").astype(str)

        # Compute monthly summary
        monthly = df.groupby("year_month").agg(
            total=("report_date", "count"),
            serious=("is_serious", "sum"),
        ).reset_index().sort_values("year_month")

        recent = monthly.tail(6)
        data = recent.to_dict("records")
        latest = data[-1] if data else {}

        return {
            "answer": (
                f"Reporting trend analysis (last 6 months): most recent month "
                f"'{latest.get('year_month', 'N/A')}' had {latest.get('total', 0)} reports "
                f"({latest.get('serious', 0)} serious). "
                f"Total reports across all months: {len(df):,}. "
                f"Query context: '{date_expr}'."
            ),
            "data": data,
            "confidence": 0.80,
        }

    def _handle_geographic_analysis(self, params: dict, _question: str) -> dict:
        location = params.get("location", "").strip() if params.get("location") else None

        with SessionLocal() as session:
            query = session.query(
                AdverseEventReport.country,
                AdverseEventReport.region,
                AdverseEventReport.is_serious,
            ).filter(AdverseEventReport.is_duplicate == False)

            if location:
                query = query.filter(
                    (AdverseEventReport.country.ilike(f"%{location}%")) |
                    (AdverseEventReport.region.ilike(f"%{location}%"))
                )

            rows = query.all()

        if not rows:
            msg = f"No reports found for location '{location}'." if location else "No reports with location data."
            return {"answer": msg, "data": [], "confidence": 0.3}

        df = pd.DataFrame(rows, columns=["country", "region", "is_serious"])
        geo = df.groupby("country").agg(
            total=("country", "count"),
            serious=("is_serious", "sum"),
        ).reset_index().sort_values("total", ascending=False)

        geo["seriousness_rate_pct"] = (geo["serious"] / geo["total"] * 100).round(2)
        data = geo.head(10).to_dict("records")
        top = data[0] if data else {}

        if location:
            answer = (
                f"Geographic analysis for '{location}': {len(df):,} reports from "
                f"{geo['country'].nunique()} country/countries. "
                f"Top country: {top.get('country', 'N/A')} ({top.get('total', 0)} reports, "
                f"{top.get('seriousness_rate_pct', 0)}% serious)."
            )
        else:
            answer = (
                f"Top reporting countries: "
                + ", ".join(f"{r['country']} ({r['total']})" for r in data[:5])
                + f". Total: {len(df):,} reports from {geo['country'].nunique()} countries."
            )

        return {"answer": answer, "data": data, "confidence": 0.88}

    def _handle_drug_comparison(self, params: dict, _question: str) -> dict:
        drug_a = params.get("drug_a", "").strip()
        drug_b = params.get("drug_b", "").strip()
        if not drug_a or not drug_b:
            return self._error("Could not identify two drugs to compare.")

        results = {}
        with SessionLocal() as session:
            for drug in [drug_a, drug_b]:
                total = session.query(AdverseEventReport).filter(
                    AdverseEventReport.drug_name.ilike(f"%{drug}%"),
                    AdverseEventReport.is_duplicate == False,
                ).count()
                serious = session.query(AdverseEventReport).filter(
                    AdverseEventReport.drug_name.ilike(f"%{drug}%"),
                    AdverseEventReport.is_serious == True,
                    AdverseEventReport.is_duplicate == False,
                ).count()
                signals_count = session.query(RiskSignal).filter(
                    RiskSignal.drug_name.ilike(f"%{drug}%")
                ).count()
                results[drug] = {
                    "drug": drug,
                    "total_reports": total,
                    "serious_reports": serious,
                    "seriousness_rate_pct": round(serious / max(total, 1) * 100, 2),
                    "risk_signals": signals_count,
                }

        a_data = results[drug_a]
        b_data = results[drug_b]

        return {
            "answer": (
                f"Comparison of '{drug_a}' vs '{drug_b}': "
                f"'{drug_a}' has {a_data['total_reports']} reports "
                f"({a_data['seriousness_rate_pct']}% serious, {a_data['risk_signals']} signals). "
                f"'{drug_b}' has {b_data['total_reports']} reports "
                f"({b_data['seriousness_rate_pct']}% serious, {b_data['risk_signals']} signals)."
            ),
            "data": [a_data, b_data],
            "confidence": 0.92,
        }

    def _handle_unknown(self, _params: dict, question: str) -> dict:
        return {
            "answer": (
                "I couldn't understand that query. Try asking:\n"
                "  - 'What are the adverse events for Metformin?'\n"
                "  - 'Show risk signals for Warfarin'\n"
                "  - 'Top 10 adverse events'\n"
                "  - 'How many reports of nausea?'\n"
                "  - 'Give me a summary'"
            ),
            "data": [],
            "confidence": 0.1,
        }

    @staticmethod
    def _error(message: str) -> dict:
        return {"answer": message, "data": [], "confidence": 0.0}


# ---------------------------------------------------------------------------
# LLM-enhanced Query Engine (requires OpenAI key)
# ---------------------------------------------------------------------------

class LLMQueryEngine:
    """
    LangChain-powered query engine that uses an LLM to interpret and
    respond to natural language pharmacovigilance queries.

    Falls back to LocalQueryEngine if LangChain/OpenAI is not configured.
    """

    SYSTEM_PROMPT = """You are PharmAI, an expert AI assistant specialised in pharmacovigilance
and drug safety monitoring. You help analyse adverse event data, interpret risk signals,
and provide evidence-based safety assessments.

You have access to tools to query the pharmacovigilance database.
Always provide precise, scientifically accurate answers. When discussing risk signals,
refer to standard disproportionality measures (PRR, ROR, IC). Always recommend
professional review of any detected safety signals before regulatory action.

Be concise but comprehensive. Use medical terminology appropriately."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> None:
        self._local = LocalQueryEngine()
        self._llm_available = False
        self._chain = None

        if api_key and api_key != "your-openai-api-key-here":
            self._init_langchain(api_key, model)

    def _init_langchain(self, api_key: str, model: str) -> None:
        try:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                from langchain.chat_models import ChatOpenAI  # type: ignore[no-redef]

            self._llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model,
                temperature=0.1,
            )
            self._llm_available = True
            logger.info("LangChain LLM initialised with model: {}", model)
        except Exception as exc:
            logger.warning("LangChain init failed ({}). Using local query engine.", exc)

    def query(self, question: str) -> dict:
        # First get structured data from local engine
        local_result = self._local.query(question)

        if not self._llm_available:
            return local_result

        try:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
            except ImportError:
                from langchain.schema import SystemMessage, HumanMessage  # type: ignore[no-redef]

            context = (
                f"Database query result: {local_result['answer']}\n"
                f"Data: {local_result['data']}"
            )
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"User question: {question}\n\nContext from database:\n{context}\n\n"
                                     "Provide a comprehensive pharmacovigilance expert response:"),
            ]
            response = self._llm.invoke(messages)
            enhanced_answer = response.content

            return {
                **local_result,
                "answer": enhanced_answer,
                "llm_enhanced": True,
                "local_answer": local_result["answer"],
            }
        except Exception as exc:
            logger.warning("LLM query enhancement failed: {}", exc)
            return local_result
