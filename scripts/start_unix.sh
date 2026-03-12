#!/usr/bin/env bash
# ============================================================
# AI Pharmacovigilance Intelligence Platform
# Unix/Linux/macOS Quick Start Script
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()  { echo -e "\n${BOLD}${BLUE}══ $1 ══${NC}"; }

echo ""
echo "============================================================"
echo "  AI Pharmacovigilance Intelligence Platform"
echo "  Quick Start — Unix/Linux/macOS"
echo "============================================================"

# Python check
step "Python Version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    error "Python 3.10+ required. Found $PYTHON_VERSION"
fi
ok "Python $PYTHON_VERSION"

# Virtual environment
step "Virtual Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Virtual environment created at venv/"
else
    warn "venv/ already exists. Skipping."
fi

source venv/bin/activate

# Dependencies
step "Python Dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Dependencies installed"

# spaCy model
step "spaCy Language Model"
if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    ok "spaCy en_core_web_sm already installed"
else
    info "Downloading spaCy en_core_web_sm..."
    python -m spacy download en_core_web_sm && ok "spaCy model installed" || warn "spaCy download failed. Regex fallback will be used."
fi

# .env file
step "Environment Configuration"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        ok ".env created from .env.example"
    else
        cat > .env << 'EOF'
DATABASE_URL=sqlite:///./data/pharmacovigilance.db
APP_ENV=development
DEBUG=true
EOF
        ok "Minimal .env created"
    fi
else
    warn ".env already exists. Skipping."
fi

# Directories
step "Creating Directories"
mkdir -p data/raw data/processed data/exports logs
ok "Directories ready"

# Database
step "Database Initialisation"
python -c "from database.connection import create_all_tables; create_all_tables(); print('Schema created.')"

# Initial pipeline
step "Running Initial Data Pipeline"
info "Generating 10,000 synthetic records and running full pipeline..."
python pipelines/pipeline_orchestrator.py full --n-records 10000

echo ""
echo "============================================================"
echo -e "${GREEN}${BOLD}  Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "  To start the API:"
echo "    uvicorn api_gateway.main:app --reload --port 8000"
echo ""
echo "  To start the Dashboard:"
echo "    streamlit run dashboard/app.py"
echo ""
echo "  API Docs:   http://localhost:8000/docs"
echo "  Dashboard:  http://localhost:8501"
echo ""
echo "  To run tests:"
echo "    pytest tests/"
echo ""
