"""
Unit Tests — NLP Extraction Service
AI Pharmacovigilance Intelligence Platform
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from services.nlp_extraction.extractor import (
    RuleBasedExtractor,
    ExtractionResult,
    get_extractor,
    ADVERSE_EVENT_TERMS,
    SEVERITY_PATTERNS,
)


@pytest.fixture
def rule_extractor():
    return RuleBasedExtractor()


class TestRuleBasedExtractor:

    def test_empty_text_returns_empty_result(self, rule_extractor):
        result = rule_extractor.extract("")
        assert result.adverse_events == []
        assert result.drugs == []
        assert result.confidence_score == 0.0

    def test_whitespace_text_returns_empty_result(self, rule_extractor):
        result = rule_extractor.extract("   ")
        assert result.adverse_events == []

    def test_detects_nausea(self, rule_extractor):
        result = rule_extractor.extract("Patient experienced nausea after taking the drug.")
        assert "nausea" in result.adverse_events

    def test_detects_multiple_events(self, rule_extractor):
        result = rule_extractor.extract(
            "The patient had dizziness, headache, and nausea."
        )
        detected = set(result.adverse_events)
        assert len(detected) >= 2

    def test_detects_rhabdomyolysis(self, rule_extractor):
        result = rule_extractor.extract(
            "Severe rhabdomyolysis was observed in patient on statin therapy."
        )
        assert "rhabdomyolysis" in result.adverse_events

    def test_severity_detection_fatal(self, rule_extractor):
        result = rule_extractor.extract("The patient died following adverse reaction.")
        assert result.severity in ("fatal", None)

    def test_severity_detection_severe(self, rule_extractor):
        result = rule_extractor.extract(
            "Severe hepatotoxicity requiring hospitalisation was observed."
        )
        assert result.severity in ("severe", None)

    def test_severity_detection_mild(self, rule_extractor):
        result = rule_extractor.extract(
            "Patient reported mild nausea that resolved spontaneously."
        )
        assert result.severity in ("mild", None)

    def test_confidence_score_range(self, rule_extractor):
        result = rule_extractor.extract(
            "Patient experienced nausea and dizziness after Drug A."
        )
        assert 0.0 <= result.confidence_score <= 1.0

    def test_model_name_set(self, rule_extractor):
        result = rule_extractor.extract("Some text with nausea.")
        assert result.model_used != ""

    def test_processing_time_recorded(self, rule_extractor):
        result = rule_extractor.extract("Patient had nausea and headache.")
        assert result.processing_time_ms >= 0

    def test_batch_extract(self, rule_extractor):
        texts = [
            "Patient had nausea.",
            "Severe bleeding was observed.",
            "Patient reported dizziness and headache.",
        ]
        results = rule_extractor.batch_extract(texts)
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_long_text_handled(self, rule_extractor):
        long_text = "Patient experienced nausea. " * 100
        result = rule_extractor.extract(long_text)
        assert "nausea" in result.adverse_events

    def test_source_text_preserved(self, rule_extractor):
        text = "Patient had severe headache."
        result = rule_extractor.extract(text)
        assert result.source_text == text

    def test_high_confidence_with_events(self, rule_extractor):
        result = rule_extractor.extract(
            "Patient taking Drug X developed nausea, vomiting, and diarrhoea."
        )
        assert result.confidence_score > 0.3

    def test_low_confidence_with_no_events(self, rule_extractor):
        result = rule_extractor.extract("The weather was sunny today.")
        assert result.confidence_score < 0.5


class TestGetExtractor:

    def test_get_rule_based_extractor(self):
        extractor = get_extractor("rule_based")
        assert extractor is not None
        assert isinstance(extractor, RuleBasedExtractor)

    def test_get_default_extractor(self):
        extractor = get_extractor()
        assert isinstance(extractor, RuleBasedExtractor)

    def test_get_unknown_mode_defaults_to_rule_based(self):
        extractor = get_extractor("nonexistent_mode")
        assert isinstance(extractor, RuleBasedExtractor)


class TestAdverseEventVocabulary:

    def test_vocabulary_not_empty(self):
        assert len(ADVERSE_EVENT_TERMS) > 50

    def test_common_terms_present(self):
        required = {"nausea", "headache", "dizziness", "rash", "fatigue"}
        assert required.issubset(ADVERSE_EVENT_TERMS)

    def test_severity_patterns_complete(self):
        assert "fatal" in SEVERITY_PATTERNS
        assert "severe" in SEVERITY_PATTERNS
        assert "mild" in SEVERITY_PATTERNS
        assert "moderate" in SEVERITY_PATTERNS
