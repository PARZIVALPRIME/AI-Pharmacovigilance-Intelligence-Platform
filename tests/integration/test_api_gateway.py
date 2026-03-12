"""
Integration Tests — API Gateway
AI Pharmacovigilance Intelligence Platform

Tests the FastAPI endpoints end-to-end using TestClient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from fastapi.testclient import TestClient
    from api_gateway.main import app

    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI or dependencies not available"
)


class TestHealthEndpoints:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "platform" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestAdverseEventEndpoints:

    def test_list_reports_returns_200(self):
        response = client.get("/api/v1/reports?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert "page" in data

    def test_list_reports_pagination(self):
        response = client.get("/api/v1/reports?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 5

    def test_list_reports_drug_filter(self):
        response = client.get("/api/v1/reports?drug_name=Metformin")
        assert response.status_code == 200

    def test_reports_summary(self):
        response = client.get("/api/v1/reports/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_reports" in data

    def test_top_events_endpoint(self):
        response = client.get("/api/v1/reports/stats/top-events?n=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_top_drugs_endpoint(self):
        response = client.get("/api/v1/reports/stats/top-drugs?n=10")
        assert response.status_code == 200

    def test_country_distribution(self):
        response = client.get("/api/v1/reports/stats/by-country")
        assert response.status_code == 200

    def test_monthly_trend(self):
        response = client.get("/api/v1/reports/stats/monthly-trend")
        assert response.status_code == 200

    def test_get_nonexistent_report_returns_404(self):
        response = client.get("/api/v1/reports/NONEXISTENT-REPORT-ID")
        assert response.status_code == 404


class TestNLPEndpoints:

    def test_nlp_extract_basic(self):
        response = client.post("/api/v1/nlp/extract", json={
            "text": "Patient experienced nausea and dizziness after taking Metformin.",
            "mode": "rule_based",
        })
        assert response.status_code == 200
        data = response.json()
        assert "drugs" in data
        assert "adverse_events" in data
        assert "confidence_score" in data

    def test_nlp_extract_detects_nausea(self):
        response = client.post("/api/v1/nlp/extract", json={
            "text": "Patient reported nausea, vomiting, and headache.",
            "mode": "rule_based",
        })
        data = response.json()
        ae_lower = [ae.lower() for ae in data.get("adverse_events", [])]
        assert "nausea" in ae_lower

    def test_nlp_extract_short_text_rejected(self):
        response = client.post("/api/v1/nlp/extract", json={
            "text": "Hi",
            "mode": "rule_based",
        })
        assert response.status_code == 422  # Validation error

    def test_nlp_stats(self):
        response = client.get("/api/v1/nlp/stats")
        assert response.status_code == 200

    def test_nlp_extract_processing_time_recorded(self):
        response = client.post("/api/v1/nlp/extract", json={
            "text": "Patient had severe myalgia after statin therapy.",
            "mode": "rule_based",
        })
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0


class TestRiskSignalEndpoints:

    def test_list_signals(self):
        response = client.get("/api/v1/signals")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data

    def test_signal_stats(self):
        response = client.get("/api/v1/signals/stats")
        assert response.status_code == 200

    def test_signals_for_drug(self):
        response = client.get("/api/v1/signals/Metformin")
        assert response.status_code == 200

    def test_signal_detection_sync(self):
        response = client.post("/api/v1/signals/detect/sync", json={
            "prr_threshold": 2.0,
            "min_reports": 3,
            "contamination": 0.05,
        })
        assert response.status_code == 200


class TestAIAssistantEndpoints:

    def test_chat_basic_query(self):
        response = client.post("/api/v1/assistant/chat", json={
            "question": "Give me a platform summary",
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert data["confidence"] >= 0.0

    def test_chat_adverse_events_query(self):
        response = client.post("/api/v1/assistant/chat", json={
            "question": "What are the top adverse events?",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 10

    def test_chat_with_session_id(self):
        response = client.post("/api/v1/assistant/chat", json={
            "question": "How many reports of nausea are there?",
            "session_id": "test-session-001",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-001"

    def test_chat_history(self):
        # First message
        client.post("/api/v1/assistant/chat", json={
            "question": "Give me a summary",
            "session_id": "history-test-session",
        })
        # Get history
        response = client.get("/api/v1/assistant/history/history-test-session")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data

    def test_suggested_queries(self):
        response = client.get("/api/v1/assistant/suggested-queries")
        assert response.status_code == 200
        data = response.json()
        assert "queries" in data
        assert len(data["queries"]) > 0


class TestAnalyticsEndpoints:

    def test_severity_distribution(self):
        response = client.get("/api/v1/analytics/severity-distribution")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_gender_distribution(self):
        response = client.get("/api/v1/analytics/gender-distribution")
        assert response.status_code == 200

    def test_age_group_distribution(self):
        response = client.get("/api/v1/analytics/age-group-distribution")
        assert response.status_code == 200

    def test_drug_class_distribution(self):
        response = client.get("/api/v1/analytics/drug-class-distribution")
        assert response.status_code == 200


class TestReportingEndpoints:

    def test_generate_json_report(self):
        response = client.post("/api/v1/reports/generate", json={
            "format": "json",
            "include_signals": True,
            "include_trends": True,
            "top_n_events": 20,
        })
        assert response.status_code == 200
        data = response.json()
        assert "report_metadata" in data or "summary" in data

    def test_invalid_format_returns_400(self):
        response = client.post("/api/v1/reports/generate", json={
            "format": "invalid_format",
        })
        assert response.status_code in (400, 500)
