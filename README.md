# AI Pharmacovigilance Intelligence Platform (v2.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.2-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
A production-grade AI-powered Pharmacovigilance (PV) platform designed to automate and enhance **Aggregate Reports & Risk Management (AR&RM)** workflows. 

This platform simulates real-world pharmaceutical safety operations, specifically aligned with **Novartis Patient Safety and Pharmacovigilance Global Team** requirements. It integrates advanced NLP for case extraction, disproportionality analysis for signal detection, and automated regulatory reporting (PSUR/PBRER).

---

## ✨ Key Features (JD-Aligned)

### 📊 Aggregate Reports & Risk Management (AR&RM)
- **Automated Report Generation**: One-click generation of **PSUR (Periodic Safety Update Reports)**, **PBRER**, and **DSUR** in PDF, Excel, and JSON formats.
- **Data Ingestion & Cleaning**: Automated pipelines to retrieve and clean data from global databases (simulated FAERS).
- **Quality Control (QC)**: Built-in QC metrics tracking for completeness, consistency, and regulatory compliance.

### ⚠️ Risk Signal Detection & Workflow
- **Statistical Algorithms**: Implementation of EMA/FDA standards including **PRR**, **ROR**, **IC (Information Component)**, and **Chi-Square**.
- **ML Anomaly Detection**: Isolation Forest algorithms to detect unusual reporting patterns and emerging safety signals.
- **Signal Lifecycle Management**: Full workflow implementation (Detected → Under Review → Confirmed → Closed) with reviewer notes.

### ⚖️ Compliance & Audit Readiness
- **HA Submission Tracker**: Real-time tracking of Health Authority (HA) AR&RM deliverables and deadlines.
- **CAPA & Quality Incidents**: Integrated RCA/CAPA module for tracking investigations and preventive actions.
- **Comprehensive Audit Log**: Regulatory-compliant audit trail capturing all system actions, report generations, and signal assessments.

### 🤖 AI-Powered Assistant (PharmAI)
- **Natural Language Querying**: Query the database using plain English (e.g., *"Compare safety profiles of Warfarin vs Metformin"*).
- **NLP Extraction**: Biomedical NER to extract adverse events, drugs, and symptoms from free-text medical narratives.

---

## 🏗️ Architecture
The system follows a modular, service-oriented architecture:
- **API Gateway**: FastAPI REST API providing 40+ endpoints.
- **Dashboard**: Streamlit-based interactive analytics and case management.
- **Intelligence Services**: Modular Python services for Signal Detection, NLP, Reporting, and Compliance.
- **Persistence Layer**: SQLAlchemy ORM with support for PostgreSQL/SQLite.
- **Containerization**: Fully Dockerized for seamless deployment.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (Optional)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PARZIVALPRIME/AI-Pharmacovigilance-Intelligence-Platform.git
   cd AI-Pharmacovigilance-Intelligence-Platform
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment:
   ```bash
   python scripts/setup_environment.py
   ```

### Running the Platform
- **Start All Services**:
  ```bash
  docker-compose up --build
  ```
- **Run Backend API**:
  ```bash
  uvicorn api_gateway.main:app --reload --port 8000
  ```
- **Run Analytics Dashboard**:
  ```bash
  streamlit run dashboard/app.py
  ```

---

## 📋 Regulatory Compliance Note
This platform is designed to support compliance with **ICH E2C (R2)** and **ICH E2E** guidelines. It ensures data integrity through unique constraints on signals and a non-destructive audit trail of all safety assessments.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, Pydantic, SQLAlchemy
- **Frontend**: Streamlit, Plotly
- **AI/NLP**: LangChain, HuggingFace Transformers, spaCy, Scikit-learn
- **Data**: Pandas, NumPy, SciPy
- **DevOps**: Docker, Loguru, Pytest

---

Developed by the PharmaAI Team.
