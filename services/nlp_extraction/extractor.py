"""
NLP Adverse Event Extraction Service
AI Pharmacovigilance Intelligence Platform

Provides multi-model NLP entity extraction for adverse drug event detection:
  - spaCy-based rule/pattern matching (fast, offline)
  - Transformer-based NER (high-accuracy biomedical model)
  - Ensemble confidence scoring

Architecture: Strategy pattern allows swapping NLP backends at runtime.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """A single named entity extracted from text."""
    text: str
    label: str          # DRUG, ADVERSE_EVENT, SYMPTOM, SEVERITY, etc.
    start: int
    end: int
    confidence: float = 1.0
    normalised: Optional[str] = None


@dataclass
class ExtractionResult:
    """Complete extraction result for one input text."""
    source_text: str
    drugs: List[str] = field(default_factory=list)
    adverse_events: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    severity: Optional[str] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    confidence_score: float = 0.0
    model_used: str = ""
    processing_time_ms: float = 0.0
    raw_entities: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known adverse event vocabulary (domain knowledge)
# ---------------------------------------------------------------------------

ADVERSE_EVENT_TERMS = {
    # Neurological
    "dizziness", "headache", "seizure", "stroke", "tremor", "neuropathy",
    "peripheral neuropathy", "ataxia", "somnolence", "insomnia", "confusion",
    "cognitive impairment", "depression", "anxiety", "suicidal ideation",
    # Cardiovascular
    "tachycardia", "bradycardia", "palpitations", "arrhythmia", "hypertension",
    "hypotension", "myocardial infarction", "heart failure", "oedema", "oedema",
    "angina", "qt prolongation", "thrombosis", "embolism", "stroke",
    # Gastrointestinal
    "nausea", "vomiting", "diarrhoea", "diarrhea", "constipation",
    "abdominal pain", "hepatotoxicity", "pancreatitis", "bleeding",
    "gastrointestinal haemorrhage", "peptic ulcer", "colitis",
    # Respiratory
    "cough", "dry cough", "dyspnoea", "bronchospasm", "pneumonitis",
    "pneumonia", "respiratory failure", "apnoea",
    # Musculoskeletal
    "myalgia", "myopathy", "rhabdomyolysis", "arthralgia", "tendinitis",
    "tendon rupture", "fracture", "osteoporosis",
    # Dermatological
    "rash", "urticaria", "angioedema", "alopecia", "photosensitivity",
    "skin necrosis", "bullous pemphigoid", "erythema",
    # Haematological
    "anaemia", "thrombocytopenia", "neutropenia", "leucopenia",
    "agranulocytosis", "haematoma", "bruising",
    # Metabolic
    "hypoglycaemia", "hyperglycaemia", "hypokalaemia", "hyperkalaemia",
    "hyponatraemia", "lactic acidosis", "diabetic ketoacidosis",
    # Immunological
    "anaphylaxis", "hypersensitivity reaction", "injection site reaction",
    "infusion reaction", "serotonin syndrome",
    # Renal
    "renal impairment", "acute kidney injury", "haematuria", "proteinuria",
    # Endocrine
    "adrenal suppression", "hypothyroidism", "hyperthyroidism", "endocrinopathy",
    # Infectious
    "infection", "serious infection", "tuberculosis", "pneumonia",
    "clostridium difficile", "genital mycotic infection", "urinary tract infection",
    # Oncological
    "lymphoma", "tumour", "carcinoma",
    # Other
    "fatigue", "weight gain", "weight loss", "fever", "flushing",
    "hearing loss", "blurred vision", "sexual dysfunction",
    "gynaecomastia", "peripheral oedema", "lipohypertrophy",
}

DRUG_INDICATORS = {
    "drug", "medication", "medicine", "treatment", "therapy", "agent",
    "compound", "substance", "pill", "tablet", "injection", "infusion",
    "administered", "prescribed", "taking", "received",
}

SEVERITY_PATTERNS = {
    "fatal": ["fatal", "death", "died", "mortality"],
    "life_threatening": ["life-threatening", "life threatening", "critical", "icu", "intensive care"],
    "severe": ["severe", "serious", "hospitalised", "hospitalized", "emergency"],
    "moderate": ["moderate", "significant", "notable"],
    "mild": ["mild", "minor", "slight", "minimal"],
}


# ---------------------------------------------------------------------------
# Abstract base extractor
# ---------------------------------------------------------------------------

class BaseExtractor(ABC):
    """Abstract base class for NLP extraction backends."""

    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract adverse events from free text."""
        ...

    @abstractmethod
    def batch_extract(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract from multiple texts efficiently."""
        ...

    def _detect_severity(self, text: str) -> Optional[str]:
        """Rule-based severity detection from text."""
        text_lower = text.lower()
        for severity, patterns in SEVERITY_PATTERNS.items():
            for p in patterns:
                if p in text_lower:
                    return severity
        return None

    def _compute_confidence(self, drugs: List[str], events: List[str]) -> float:
        """Heuristic confidence score based on extraction richness."""
        score = 0.0
        if drugs:
            score += 0.4
        if events:
            score += 0.4 + min(0.1, len(events) * 0.02)
        if not drugs and not events:
            score = 0.1
        return min(round(score, 4), 1.0)


# ---------------------------------------------------------------------------
# Rule-based extractor (spaCy + custom patterns)
# ---------------------------------------------------------------------------

class RuleBasedExtractor(BaseExtractor):
    """
    Fast rule-based adverse event extractor using spaCy NLP + domain patterns.
    Falls back to pure regex if spaCy is not installed.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.model_name = "rule_based_spacy"
        self._nlp = None
        self._spacy_available = False
        self._load_spacy(spacy_model)

    def _load_spacy(self, model_name: str) -> None:
        try:
            import spacy
            try:
                self._nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info("spaCy model '{}' loaded successfully.", model_name)
            except OSError:
                logger.warning(
                    "spaCy model '{}' not found. Downloading...", model_name
                )
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", model_name],
                    check=True, capture_output=True,
                )
                self._nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info("spaCy model '{}' downloaded and loaded.", model_name)
        except Exception as exc:
            logger.warning("spaCy unavailable ({}). Using regex fallback.", exc)
            self._spacy_available = False

    def extract(self, text: str) -> ExtractionResult:
        t0 = time.perf_counter()
        result = ExtractionResult(source_text=text, model_used=self.model_name)

        if not text or not text.strip():
            return result

        if self._spacy_available:
            result = self._extract_with_spacy(text, result)
        else:
            result = self._extract_with_regex(text, result)

        result.severity = result.severity or self._detect_severity(text)
        result.confidence_score = self._compute_confidence(result.drugs, result.adverse_events)
        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    def _extract_with_spacy(self, text: str, result: ExtractionResult) -> ExtractionResult:
        doc = self._nlp(text)
        entities = []

        # Extract named entities recognised by spaCy
        for ent in doc.ents:
            label_map = {
                "ORG": "DRUG",           # Organisations often = pharma companies/drugs
                "PRODUCT": "DRUG",
                "CHEMICAL": "DRUG",
                "DISEASE": "ADVERSE_EVENT",
                "GPE": None,
                "PERSON": None,
            }
            mapped = label_map.get(ent.label_)
            if mapped:
                entities.append(ExtractedEntity(
                    text=ent.text, label=mapped,
                    start=ent.start_char, end=ent.end_char, confidence=0.7,
                ))

        # Domain vocabulary matching
        text_lower = text.lower()
        for ae_term in ADVERSE_EVENT_TERMS:
            pattern = r"\b" + re.escape(ae_term) + r"\b"
            for m in re.finditer(pattern, text_lower):
                entities.append(ExtractedEntity(
                    text=ae_term, label="ADVERSE_EVENT",
                    start=m.start(), end=m.end(), confidence=0.9,
                ))

        result.entities = entities
        result.drugs = list({e.text for e in entities if e.label == "DRUG"})
        result.adverse_events = list({e.text for e in entities if e.label == "ADVERSE_EVENT"})
        result.symptoms = list({e.text for e in entities if e.label == "SYMPTOM"})
        return result

    def _extract_with_regex(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Pure regex fallback extractor."""
        text_lower = text.lower()
        ae_found = []
        for ae_term in ADVERSE_EVENT_TERMS:
            pattern = r"\b" + re.escape(ae_term) + r"\b"
            if re.search(pattern, text_lower):
                ae_found.append(ae_term)

        # Simple drug name extraction: capitalised words near drug indicators
        drug_pattern = r"(?:Drug|drug|medication|medicine|treatment)\s+([A-Z][a-zA-Z0-9\-]+)"
        drugs = re.findall(drug_pattern, text)

        # Also look for capitalised multi-word proper nouns before "administration" / "therapy"
        drug_pattern2 = r"([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)?)\s+(?:administration|therapy|treatment|use)"
        drugs += re.findall(drug_pattern2, text)

        result.drugs = list(set(drugs))
        result.adverse_events = ae_found
        return result

    def batch_extract(self, texts: List[str]) -> List[ExtractionResult]:
        return [self.extract(t) for t in texts]


# ---------------------------------------------------------------------------
# Transformer-based extractor (HuggingFace)
# ---------------------------------------------------------------------------

class TransformerExtractor(BaseExtractor):
    """
    High-accuracy transformer-based biomedical NER extractor.
    Uses d4data/biomedical-ner-all or similar HuggingFace model.
    Falls back to rule-based if transformers not available.
    """

    DEFAULT_MODEL = "d4data/biomedical-ner-all"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = None
        self._available = False
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            logger.info("Loading transformer NER model: {}", self.model_name)
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
            self._available = True
            logger.info("Transformer NER pipeline ready.")
        except Exception as exc:
            logger.warning(
                "Transformer model unavailable ({}). Falling back to rule-based.", exc
            )
            self._available = False
            self._fallback = RuleBasedExtractor()

    def extract(self, text: str) -> ExtractionResult:
        if not self._available:
            return self._fallback.extract(text)

        t0 = time.perf_counter()
        result = ExtractionResult(source_text=text, model_used=self.model_name)

        if not text or not text.strip():
            return result

        try:
            # Truncate to model max length
            truncated = text[:512]
            ner_output = self._pipeline(truncated)

            entities = []
            for ent in ner_output:
                label = self._map_label(ent.get("entity_group", ""))
                if label:
                    entities.append(ExtractedEntity(
                        text=ent["word"].strip(),
                        label=label,
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                        confidence=round(float(ent.get("score", 0.0)), 4),
                    ))

            # Also run domain vocab matching
            text_lower = text.lower()
            for ae_term in ADVERSE_EVENT_TERMS:
                pattern = r"\b" + re.escape(ae_term) + r"\b"
                if re.search(pattern, text_lower):
                    entities.append(ExtractedEntity(
                        text=ae_term, label="ADVERSE_EVENT",
                        start=0, end=0, confidence=0.85,
                    ))

            result.entities = entities
            result.drugs = list({e.text for e in entities if e.label == "DRUG"})
            result.adverse_events = list({e.text for e in entities if e.label in ("ADVERSE_EVENT", "DISEASE", "SIGN_SYMPTOM")})
            result.symptoms = list({e.text for e in entities if e.label == "SYMPTOM"})
            result.severity = self._detect_severity(text)
            result.confidence_score = self._compute_confidence(result.drugs, result.adverse_events)

        except Exception as exc:
            logger.error("Transformer extraction error: {}", exc)
            return self._fallback.extract(text) if hasattr(self, "_fallback") else result

        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    def batch_extract(self, texts: List[str]) -> List[ExtractionResult]:
        if not self._available:
            return self._fallback.batch_extract(texts)

        results = []
        BATCH = 8
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            for text in batch:
                results.append(self.extract(text))
        return results

    @staticmethod
    def _map_label(raw_label: str) -> Optional[str]:
        """Map biomedical NER labels to internal labels."""
        mapping = {
            "DRUG": "DRUG",
            "MEDICATION": "DRUG",
            "CHEMICAL": "DRUG",
            "DISEASE": "ADVERSE_EVENT",
            "DISORDER": "ADVERSE_EVENT",
            "SIGN_SYMPTOM": "ADVERSE_EVENT",
            "ADVERSE_EFFECT": "ADVERSE_EVENT",
            "SYMPTOM": "SYMPTOM",
            "FINDING": "SYMPTOM",
            "ANATOMY": None,
            "ORGANISM": None,
            "PERSON": None,
            "LOCATION": None,
        }
        return mapping.get(raw_label.upper())


# ---------------------------------------------------------------------------
# Ensemble extractor (combines both models)
# ---------------------------------------------------------------------------

class EnsembleExtractor(BaseExtractor):
    """
    Combines rule-based and transformer extractors with confidence-weighted voting.
    Provides the highest accuracy at the cost of some additional latency.
    """

    def __init__(self) -> None:
        self.model_name = "ensemble"
        self._rule_extractor = RuleBasedExtractor()
        self._transformer_extractor = TransformerExtractor()

    def extract(self, text: str) -> ExtractionResult:
        t0 = time.perf_counter()

        rule_result = self._rule_extractor.extract(text)
        transformer_result = self._transformer_extractor.extract(text)

        # Merge results — union of findings
        merged_drugs = list(set(rule_result.drugs) | set(transformer_result.drugs))
        merged_events = list(set(rule_result.adverse_events) | set(transformer_result.adverse_events))
        merged_symptoms = list(set(rule_result.symptoms) | set(transformer_result.symptoms))

        # Severity: prefer transformer result if available
        severity = transformer_result.severity or rule_result.severity

        # Confidence: weighted average
        confidence = round(
            0.4 * rule_result.confidence_score + 0.6 * transformer_result.confidence_score, 4
        )

        result = ExtractionResult(
            source_text=text,
            drugs=merged_drugs,
            adverse_events=merged_events,
            symptoms=merged_symptoms,
            severity=severity,
            entities=rule_result.entities + transformer_result.entities,
            confidence_score=confidence,
            model_used=self.model_name,
            processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return result

    def batch_extract(self, texts: List[str]) -> List[ExtractionResult]:
        return [self.extract(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_extractor(mode: str = "rule_based") -> BaseExtractor:
    """
    Factory function to get the appropriate extractor.

    Parameters
    ----------
    mode : str
        One of "rule_based", "transformer", "ensemble"
    """
    if mode == "transformer":
        return TransformerExtractor()
    elif mode == "ensemble":
        return EnsembleExtractor()
    else:
        return RuleBasedExtractor()
