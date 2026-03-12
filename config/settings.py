"""
AI Pharmacovigilance Intelligence Platform
Application Configuration

Centralised settings management using Pydantic BaseSettings.
All values can be overridden via environment variables or .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Base directory resolution
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """Core application settings."""

    APP_NAME: str = "AI Pharmacovigilance Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production-must-be-at-least-32-chars!!"

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class DatabaseSettings(BaseSettings):
    DATABASE_URL: str = f"sqlite:///{ROOT_DIR}/data/pharmacovigilance.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_ECHO: bool = False

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class APISettings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
    ]

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class NLPSettings(BaseSettings):
    SPACY_MODEL: str = "en_core_web_sm"
    NLP_BATCH_SIZE: int = 32
    NLP_MAX_TEXT_LENGTH: int = 10_000
    HF_MODEL_NAME: str = "d4data/biomedical-ner-all"

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class RiskDetectionSettings(BaseSettings):
    SIGNAL_THRESHOLD: float = 2.0
    ANOMALY_CONTAMINATION: float = 0.05
    MIN_REPORTS_FOR_SIGNAL: int = 3
    PRR_THRESHOLD: float = 2.0
    ROR_THRESHOLD: float = 2.0
    CHI_SQUARE_SIGNIFICANCE: float = 0.05

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class AIAssistantSettings(BaseSettings):
    OPENAI_API_KEY: str = "your-openai-api-key-here"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1000

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class ReportingSettings(BaseSettings):
    REPORT_OUTPUT_DIR: Path = ROOT_DIR / "data" / "exports"
    MAX_REPORT_ROWS: int = 50_000

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class LoggingSettings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = ROOT_DIR / "logs"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class DataSettings(BaseSettings):
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
    SYNTHETIC_DATASET_SIZE: int = 10_000

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


class Settings(BaseSettings):
    """Master settings aggregating all sub-settings."""

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    risk: RiskDetectionSettings = Field(default_factory=RiskDetectionSettings)
    ai_assistant: AIAssistantSettings = Field(default_factory=AIAssistantSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()


# Convenience export
settings = get_settings()
