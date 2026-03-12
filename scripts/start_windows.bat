@echo off
REM ============================================================
REM AI Pharmacovigilance Intelligence Platform
REM Windows Quick Start Script
REM ============================================================

echo.
echo ============================================================
echo  AI Pharmacovigilance Intelligence Platform
echo  Quick Start for Windows
echo ============================================================
echo.

REM Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Create virtual environment
if not exist venv\ (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)

REM Activate venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed.

REM Download spaCy model
echo [INFO] Downloading spaCy model...
python -m spacy download en_core_web_sm 2>NUL
if errorlevel 1 (
    echo [WARN] spaCy model download failed. Regex fallback will be used.
) else (
    echo [OK] spaCy model ready.
)

REM Create .env if missing
if not exist .env (
    if exist .env.example (
        copy .env.example .env >NUL
        echo [OK] .env created from .env.example
    )
)

REM Create directories
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\exports mkdir data\exports
if not exist logs mkdir logs
echo [OK] Directories ready.

REM Initialise database
echo [INFO] Initialising database...
python -c "from database.connection import create_all_tables; create_all_tables(); print('[OK] Database schema created.')"

REM Run pipeline
echo.
echo ============================================================
echo  Running Initial Data Pipeline
echo ============================================================
python pipelines/pipeline_orchestrator.py full --n-records 10000 --skip-nlp

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo  To start the API:
echo    uvicorn api_gateway.main:app --reload --port 8000
echo.
echo  To start the Dashboard:
echo    streamlit run dashboard/app.py
echo.
echo  API Docs: http://localhost:8000/docs
echo  Dashboard: http://localhost:8501
echo.
pause
