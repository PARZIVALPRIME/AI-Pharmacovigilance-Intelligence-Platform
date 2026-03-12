"""
Synthetic Dataset Generator
AI Pharmacovigilance Intelligence Platform

Generates a realistic synthetic pharmacovigilance dataset with 10,000+
records mimicking FAERS / EudraVigilance data patterns.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference vocabulary
# ---------------------------------------------------------------------------

DRUGS = [
    ("Atorvastatin", "Lipitor", "Statins"),
    ("Metformin", "Glucophage", "Biguanides"),
    ("Lisinopril", "Zestril", "ACE Inhibitors"),
    ("Amlodipine", "Norvasc", "Calcium Channel Blockers"),
    ("Omeprazole", "Prilosec", "Proton Pump Inhibitors"),
    ("Metoprolol", "Lopressor", "Beta Blockers"),
    ("Losartan", "Cozaar", "ARBs"),
    ("Albuterol", "Ventolin", "Beta-2 Agonists"),
    ("Gabapentin", "Neurontin", "Anticonvulsants"),
    ("Sertraline", "Zoloft", "SSRIs"),
    ("Amoxicillin", "Amoxil", "Penicillins"),
    ("Azithromycin", "Zithromax", "Macrolides"),
    ("Ciprofloxacin", "Cipro", "Fluoroquinolones"),
    ("Warfarin", "Coumadin", "Anticoagulants"),
    ("Apixaban", "Eliquis", "Factor Xa Inhibitors"),
    ("Clopidogrel", "Plavix", "Antiplatelets"),
    ("Adalimumab", "Humira", "TNF Inhibitors"),
    ("Infliximab", "Remicade", "TNF Inhibitors"),
    ("Pembrolizumab", "Keytruda", "PD-1 Inhibitors"),
    ("Nivolumab", "Opdivo", "PD-1 Inhibitors"),
    ("Ibuprofen", "Advil", "NSAIDs"),
    ("Naproxen", "Aleve", "NSAIDs"),
    ("Prednisone", "Deltasone", "Corticosteroids"),
    ("Levothyroxine", "Synthroid", "Thyroid Hormones"),
    ("Insulin Glargine", "Lantus", "Insulins"),
    ("Empagliflozin", "Jardiance", "SGLT2 Inhibitors"),
    ("Sitagliptin", "Januvia", "DPP-4 Inhibitors"),
    ("Rosuvastatin", "Crestor", "Statins"),
    ("Furosemide", "Lasix", "Loop Diuretics"),
    ("Spironolactone", "Aldactone", "Aldosterone Antagonists"),
]

ADVERSE_EVENTS = {
    "Statins": [
        ("myalgia", 0.18), ("rhabdomyolysis", 0.03), ("hepatotoxicity", 0.05),
        ("peripheral neuropathy", 0.04), ("myopathy", 0.07), ("headache", 0.10),
        ("nausea", 0.09), ("dizziness", 0.08),
    ],
    "Biguanides": [
        ("lactic acidosis", 0.02), ("nausea", 0.20), ("diarrhoea", 0.18),
        ("vomiting", 0.12), ("abdominal pain", 0.10), ("vitamin B12 deficiency", 0.08),
        ("anorexia", 0.07), ("metallic taste", 0.06),
    ],
    "ACE Inhibitors": [
        ("dry cough", 0.25), ("hyperkalaemia", 0.12), ("hypotension", 0.10),
        ("angioedema", 0.04), ("renal impairment", 0.08), ("rash", 0.06),
        ("dizziness", 0.09), ("fatigue", 0.08),
    ],
    "Calcium Channel Blockers": [
        ("peripheral oedema", 0.22), ("flushing", 0.15), ("headache", 0.14),
        ("dizziness", 0.11), ("palpitations", 0.09), ("nausea", 0.07),
        ("constipation", 0.08), ("bradycardia", 0.04),
    ],
    "Proton Pump Inhibitors": [
        ("headache", 0.15), ("diarrhoea", 0.14), ("nausea", 0.12),
        ("hypomagnesaemia", 0.08), ("Clostridium difficile", 0.05),
        ("bone fracture", 0.04), ("vitamin B12 deficiency", 0.06),
        ("community-acquired pneumonia", 0.05),
    ],
    "Beta Blockers": [
        ("fatigue", 0.22), ("bradycardia", 0.16), ("cold extremities", 0.12),
        ("depression", 0.10), ("bronchospasm", 0.08), ("hypotension", 0.09),
        ("sexual dysfunction", 0.07), ("weight gain", 0.08),
    ],
    "ARBs": [
        ("hyperkalaemia", 0.10), ("hypotension", 0.09), ("dizziness", 0.12),
        ("renal impairment", 0.07), ("fatigue", 0.09), ("headache", 0.11),
        ("back pain", 0.06), ("upper respiratory infection", 0.08),
    ],
    "Beta-2 Agonists": [
        ("tachycardia", 0.18), ("tremor", 0.16), ("hypokalaemia", 0.10),
        ("headache", 0.12), ("palpitations", 0.14), ("anxiety", 0.08),
        ("paradoxical bronchospasm", 0.03), ("hypertension", 0.05),
    ],
    "Anticonvulsants": [
        ("dizziness", 0.22), ("somnolence", 0.20), ("ataxia", 0.12),
        ("peripheral oedema", 0.10), ("weight gain", 0.12), ("blurred vision", 0.08),
        ("cognitive impairment", 0.07), ("suicidal ideation", 0.03),
    ],
    "SSRIs": [
        ("nausea", 0.22), ("insomnia", 0.18), ("sexual dysfunction", 0.16),
        ("weight gain", 0.12), ("agitation", 0.09), ("serotonin syndrome", 0.03),
        ("suicidal ideation", 0.04), ("diarrhoea", 0.08),
    ],
    "Penicillins": [
        ("diarrhoea", 0.22), ("rash", 0.14), ("anaphylaxis", 0.03),
        ("nausea", 0.16), ("vomiting", 0.10), ("urticaria", 0.09),
        ("Clostridium difficile", 0.06), ("hypersensitivity reaction", 0.08),
    ],
    "Macrolides": [
        ("nausea", 0.20), ("diarrhoea", 0.18), ("abdominal pain", 0.14),
        ("QT prolongation", 0.06), ("hepatotoxicity", 0.04), ("hearing loss", 0.03),
        ("vomiting", 0.12), ("rash", 0.07),
    ],
    "Fluoroquinolones": [
        ("tendinitis", 0.08), ("tendon rupture", 0.04), ("nausea", 0.16),
        ("diarrhoea", 0.14), ("QT prolongation", 0.06), ("peripheral neuropathy", 0.05),
        ("CNS toxicity", 0.04), ("photosensitivity", 0.07),
    ],
    "Anticoagulants": [
        ("bleeding", 0.25), ("haematoma", 0.12), ("bruising", 0.18),
        ("gastrointestinal haemorrhage", 0.08), ("intracranial haemorrhage", 0.03),
        ("skin necrosis", 0.02), ("anaemia", 0.09), ("nausea", 0.06),
    ],
    "Factor Xa Inhibitors": [
        ("bleeding", 0.20), ("bruising", 0.16), ("haematuria", 0.08),
        ("gastrointestinal haemorrhage", 0.07), ("anaemia", 0.08),
        ("dizziness", 0.07), ("nausea", 0.06), ("fatigue", 0.05),
    ],
    "Antiplatelets": [
        ("bleeding", 0.18), ("bruising", 0.14), ("gastrointestinal upset", 0.12),
        ("rash", 0.06), ("thrombocytopenia", 0.04), ("headache", 0.08),
        ("diarrhoea", 0.09), ("neutropenia", 0.03),
    ],
    "TNF Inhibitors": [
        ("injection site reaction", 0.22), ("upper respiratory infection", 0.18),
        ("serious infection", 0.08), ("reactivation tuberculosis", 0.03),
        ("congestive heart failure", 0.04), ("demyelinating disease", 0.02),
        ("lymphoma", 0.02), ("hepatotoxicity", 0.05),
    ],
    "PD-1 Inhibitors": [
        ("fatigue", 0.28), ("immune-related pneumonitis", 0.08),
        ("immune-related colitis", 0.07), ("immune-related hepatitis", 0.05),
        ("rash", 0.18), ("endocrinopathy", 0.10), ("infusion reaction", 0.06),
        ("immune-related nephritis", 0.03),
    ],
    "NSAIDs": [
        ("gastrointestinal upset", 0.24), ("peptic ulcer", 0.08),
        ("renal impairment", 0.07), ("hypertension", 0.08), ("oedema", 0.09),
        ("cardiovascular event", 0.05), ("hepatotoxicity", 0.04), ("rash", 0.06),
    ],
    "Corticosteroids": [
        ("hyperglycaemia", 0.20), ("weight gain", 0.18), ("osteoporosis", 0.10),
        ("adrenal suppression", 0.08), ("Cushingoid features", 0.07),
        ("infection", 0.10), ("insomnia", 0.09), ("hypertension", 0.08),
    ],
    "Thyroid Hormones": [
        ("palpitations", 0.18), ("tachycardia", 0.14), ("tremor", 0.12),
        ("insomnia", 0.12), ("weight loss", 0.10), ("heat intolerance", 0.09),
        ("anxiety", 0.08), ("osteoporosis", 0.06),
    ],
    "Insulins": [
        ("hypoglycaemia", 0.30), ("weight gain", 0.18), ("injection site reaction", 0.14),
        ("lipohypertrophy", 0.08), ("oedema", 0.06), ("hypokalaemia", 0.05),
        ("anaphylaxis", 0.02), ("hypoglycaemic coma", 0.03),
    ],
    "SGLT2 Inhibitors": [
        ("genital mycotic infection", 0.20), ("urinary tract infection", 0.16),
        ("diabetic ketoacidosis", 0.04), ("polyuria", 0.12), ("hypotension", 0.08),
        ("Fournier gangrene", 0.01), ("fracture", 0.05), ("acute kidney injury", 0.05),
    ],
    "DPP-4 Inhibitors": [
        ("nasopharyngitis", 0.15), ("headache", 0.12), ("pancreatitis", 0.03),
        ("urinary tract infection", 0.10), ("arthralgia", 0.08),
        ("bullous pemphigoid", 0.02), ("heart failure", 0.04), ("nausea", 0.08),
    ],
    "Loop Diuretics": [
        ("hypokalaemia", 0.22), ("dehydration", 0.14), ("hypotension", 0.12),
        ("hearing loss", 0.06), ("hyperuricaemia", 0.10), ("hyponatraemia", 0.08),
        ("metabolic alkalosis", 0.07), ("azotaemia", 0.06),
    ],
    "Aldosterone Antagonists": [
        ("hyperkalaemia", 0.24), ("gynaecomastia", 0.14), ("menstrual irregularity", 0.08),
        ("renal impairment", 0.08), ("dizziness", 0.10), ("headache", 0.09),
        ("hypotension", 0.08), ("fatigue", 0.07),
    ],
}

COUNTRIES = [
    ("United States", "North America", 0.30),
    ("Germany", "Europe", 0.10),
    ("United Kingdom", "Europe", 0.09),
    ("France", "Europe", 0.07),
    ("Japan", "Asia-Pacific", 0.08),
    ("Canada", "North America", 0.05),
    ("Australia", "Asia-Pacific", 0.04),
    ("Brazil", "Latin America", 0.04),
    ("Italy", "Europe", 0.04),
    ("Spain", "Europe", 0.03),
    ("Netherlands", "Europe", 0.02),
    ("Switzerland", "Europe", 0.02),
    ("India", "Asia-Pacific", 0.03),
    ("South Korea", "Asia-Pacific", 0.02),
    ("Mexico", "Latin America", 0.02),
    ("Sweden", "Europe", 0.01),
    ("Belgium", "Europe", 0.01),
    ("Denmark", "Europe", 0.01),
    ("Norway", "Europe", 0.01),
    ("Other", "Other", 0.01),
]

SEVERITY_DIST = {
    "mild": 0.35,
    "moderate": 0.30,
    "severe": 0.20,
    "life_threatening": 0.10,
    "fatal": 0.05,
}

OUTCOME_DIST = {
    "recovered": 0.45,
    "recovering": 0.25,
    "not_recovered": 0.15,
    "fatal": 0.05,
    "unknown": 0.10,
}

CLINICAL_PHASES = ["phase_1", "phase_2", "phase_3", "phase_4", "post_market"]
CLINICAL_PHASE_DIST = [0.03, 0.07, 0.15, 0.25, 0.50]

SOURCE_TEXTS_TEMPLATES = [
    "Patient reported {event} after initiating {drug}. Severity was classified as {severity}.",
    "{drug} administration was associated with onset of {event} in a {age}-year-old {gender} patient.",
    "Following {drug} therapy, the patient experienced {event}. The reaction was considered {severity}.",
    "Adverse reaction report: {event} observed in patient taking {drug} for {indication}.",
    "Post-marketing surveillance identified {event} in patient receiving {drug}.",
    "Clinical trial participant developed {event} while on {drug} {phase}.",
    "Patient {age} y/o presented with {event}, possibly related to {drug} use.",
    "Spontaneous report of {event} in {gender} patient taking {drug}.",
]

INDICATIONS = [
    "hypertension", "type 2 diabetes", "hyperlipidaemia", "heart failure",
    "atrial fibrillation", "GERD", "asthma", "epilepsy", "depression",
    "rheumatoid arthritis", "cancer treatment", "pain management",
    "anticoagulation", "infection", "thyroid disorder",
]


def _weighted_choice(items: list, weights: list) -> any:
    """Select from items based on weights."""
    return random.choices(items, weights=weights, k=1)[0]


def _generate_source_text(drug: str, event: str, severity: str, age: int, gender: str, phase: str) -> str:
    template = random.choice(SOURCE_TEXTS_TEMPLATES)
    indication = random.choice(INDICATIONS)
    return template.format(
        drug=drug,
        event=event,
        severity=severity,
        age=age,
        gender=gender,
        indication=indication,
        phase=phase.replace("_", " "),
    )


def generate_synthetic_dataset(n_records: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic pharmacovigilance dataset.

    Parameters
    ----------
    n_records : int
        Number of records to generate (default 10,000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Full synthetic dataset ready for ingestion.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Pre-build weighted lists
    country_names = [c[0] for c in COUNTRIES]
    country_regions = {c[0]: c[1] for c in COUNTRIES}
    country_weights = [c[2] for c in COUNTRIES]

    severity_keys = list(SEVERITY_DIST.keys())
    severity_weights = list(SEVERITY_DIST.values())

    outcome_keys = list(OUTCOME_DIST.keys())
    outcome_weights = list(OUTCOME_DIST.values())

    # Date range: last 5 years
    end_date = date.today()
    start_date = end_date - timedelta(days=5 * 365)
    date_range_days = (end_date - start_date).days

    records = []
    for i in range(n_records):
        # Drug selection
        drug_tuple = random.choice(DRUGS)
        drug_name, brand_name, drug_class = drug_tuple

        # Adverse event for this drug class
        ae_pool = ADVERSE_EVENTS.get(drug_class, [("adverse reaction", 1.0)])
        ae_names = [a[0] for a in ae_pool]
        ae_weights = [a[1] for a in ae_pool]
        # Normalise weights
        total_w = sum(ae_weights)
        ae_weights = [w / total_w for w in ae_weights]
        adverse_event = _weighted_choice(ae_names, ae_weights)

        # Demographics
        age = max(18, min(95, int(np.random.normal(58, 18))))
        gender = _weighted_choice(["male", "female", "other", "unknown"], [0.46, 0.46, 0.04, 0.04])

        if age < 18:
            age_group = "paediatric"
        elif age < 45:
            age_group = "young_adult"
        elif age < 65:
            age_group = "middle_aged"
        elif age < 75:
            age_group = "elderly"
        else:
            age_group = "very_elderly"

        # Geography
        country = _weighted_choice(country_names, country_weights)
        region = country_regions[country]

        # Dates
        report_offset = random.randint(0, date_range_days)
        report_date = start_date + timedelta(days=report_offset)
        receipt_date = report_date + timedelta(days=random.randint(1, 30))

        # Severity & outcome
        severity = _weighted_choice(severity_keys, severity_weights)
        outcome = _weighted_choice(outcome_keys, outcome_weights)

        # Clinical phase
        clinical_phase = _weighted_choice(CLINICAL_PHASES, CLINICAL_PHASE_DIST)

        # Seriousness
        is_serious = severity in ("severe", "life_threatening", "fatal")

        # Generate source text
        source_text = _generate_source_text(
            drug=drug_name,
            event=adverse_event,
            severity=severity,
            age=age,
            gender=gender,
            phase=clinical_phase,
        )

        # Confidence score (simulated NLP confidence)
        confidence = round(random.uniform(0.65, 0.99), 4)

        records.append({
            "report_id": f"PVR-{uuid.uuid4().hex[:12].upper()}",
            "drug_name": drug_name,
            "brand_name": brand_name,
            "drug_class": drug_class,
            "adverse_event": adverse_event,
            "severity": severity,
            "outcome": outcome,
            "patient_age": age,
            "patient_age_group": age_group,
            "gender": gender,
            "country": country,
            "region": region,
            "report_date": report_date.isoformat(),
            "receipt_date": receipt_date.isoformat(),
            "clinical_phase": clinical_phase,
            "is_serious": is_serious,
            "source_text": source_text,
            "source_type": _weighted_choice(
                ["spontaneous", "clinical_trial", "literature", "health_authority", "patient"],
                [0.55, 0.20, 0.12, 0.08, 0.05],
            ),
            "confidence_score": confidence,
        })

    df = pd.DataFrame(records)

    # Add derived columns
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["receipt_date"] = pd.to_datetime(df["receipt_date"])
    df["report_year"] = df["report_date"].dt.year
    df["report_quarter"] = df["report_date"].dt.to_period("Q").astype(str)
    df["report_month"] = df["report_date"].dt.to_period("M").astype(str)
    df["days_to_receipt"] = (df["receipt_date"] - df["report_date"]).dt.days

    return df


def download_faers_dataset(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    Attempt to download a real FAERS dataset from FDA.
    Returns None if download fails (network unreachable / rate-limited).
    """
    import urllib.request

    faers_url = (
        "https://fis.fda.gov/content/Exports/faers_ascii_2023Q3.zip"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / "faers_2023Q3.zip"
        print(f"Attempting FAERS download from {faers_url} ...")
        urllib.request.urlretrieve(faers_url, zip_path)
        print("Download successful.")
        return None  # Would parse ZIP here in extended implementation
    except Exception as exc:
        print(f"FAERS download skipped ({exc}). Using synthetic dataset.")
        return None
