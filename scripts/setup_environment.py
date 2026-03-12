"""
Environment Initialisation Script
AI Pharmacovigilance Intelligence Platform

Sets up the virtual environment, installs dependencies,
downloads spaCy models, and initialises the database.
Run once after cloning the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

def info(msg: str):  print(f"{BLUE}[INFO]{RESET}  {msg}")
def ok(msg: str):    print(f"{GREEN}[OK]{RESET}    {msg}")
def warn(msg: str):  print(f"{YELLOW}[WARN]{RESET}  {msg}")
def error(msg: str): print(f"{RED}[ERROR]{RESET} {msg}")
def step(msg: str):  print(f"\n{BOLD}{BLUE}══ {msg} ══{RESET}")


def run(cmd: list, cwd: Path = ROOT_DIR, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command."""
    info(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if check and result.returncode != 0:
        error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def check_python_version():
    step("Python Version Check")
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) < (3, 10):
        error(f"Python 3.10+ required. Found {major}.{minor}")
        sys.exit(1)
    ok(f"Python {major}.{minor} detected")


def create_venv():
    step("Virtual Environment")
    venv_dir = ROOT_DIR / "venv"
    if venv_dir.exists():
        warn("venv/ already exists. Skipping creation.")
    else:
        run([sys.executable, "-m", "venv", "venv"])
        ok("Virtual environment created at venv/")

    # Return the python executable path
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_dependencies(python_exe: Path):
    step("Installing Python Dependencies")
    req_file = ROOT_DIR / "requirements.txt"
    if not req_file.exists():
        error("requirements.txt not found")
        sys.exit(1)

    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_exe), "-m", "pip", "install", "-r", str(req_file)])
    ok("All dependencies installed")


def install_spacy_model(python_exe: Path):
    step("spaCy Language Model")
    result = subprocess.run(
        [str(python_exe), "-c", "import spacy; spacy.load('en_core_web_sm')"],
        capture_output=True,
    )
    if result.returncode == 0:
        ok("spaCy en_core_web_sm already installed")
    else:
        info("Downloading spaCy en_core_web_sm...")
        run([str(python_exe), "-m", "spacy", "download", "en_core_web_sm"])
        ok("spaCy model installed")


def create_env_file():
    step(".env Configuration")
    env_file = ROOT_DIR / ".env"
    example_file = ROOT_DIR / ".env.example"

    if env_file.exists():
        warn(".env already exists. Skipping.")
    elif example_file.exists():
        import shutil
        shutil.copy(example_file, env_file)
        ok(".env created from .env.example")
    else:
        warn(".env.example not found. Creating minimal .env")
        env_file.write_text(
            "DATABASE_URL=sqlite:///./data/pharmacovigilance.db\n"
            "APP_ENV=development\n"
            "DEBUG=true\n"
        )


def create_directories():
    step("Directory Structure")
    dirs = [
        ROOT_DIR / "data" / "raw",
        ROOT_DIR / "data" / "processed",
        ROOT_DIR / "data" / "exports",
        ROOT_DIR / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"Created: {d.relative_to(ROOT_DIR)}")


def initialise_database(python_exe: Path):
    step("Database Initialisation")
    init_script = """
import sys
sys.path.insert(0, '.')
from database.connection import create_all_tables
create_all_tables()
print("Database schema created successfully.")
"""
    result = subprocess.run(
        [str(python_exe), "-c", init_script],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("Database schema initialised")
        print(result.stdout)
    else:
        warn(f"Database init output: {result.stderr}")


def print_summary():
    print(f"\n{'=' * 60}")
    print(f"{BOLD}{GREEN}  Setup Complete!{RESET}")
    print(f"{'=' * 60}")
    print(f"""
{BOLD}Next Steps:{RESET}

  1. Activate the virtual environment:
     {YELLOW}Windows:{RESET}  venv\\Scripts\\activate
     {YELLOW}Linux/Mac:{RESET} source venv/bin/activate

  2. Copy and configure .env:
     {YELLOW}cp .env.example .env{RESET}

  3. Run the data pipeline:
     {YELLOW}python pipelines/pipeline_orchestrator.py full{RESET}

  4. Start the API:
     {YELLOW}uvicorn api_gateway.main:app --reload --port 8000{RESET}

  5. Start the dashboard:
     {YELLOW}streamlit run dashboard/app.py{RESET}

  6. Run tests:
     {YELLOW}pytest tests/{RESET}

{BOLD}URLs:{RESET}
  API:        http://localhost:8000
  API Docs:   http://localhost:8000/docs
  Dashboard:  http://localhost:8501
""")


if __name__ == "__main__":
    print(f"\n{BOLD}AI Pharmacovigilance Intelligence Platform — Setup{RESET}")
    print("=" * 60)

    check_python_version()
    python_exe = create_venv()
    install_dependencies(python_exe)
    install_spacy_model(python_exe)
    create_env_file()
    create_directories()
    initialise_database(python_exe)
    print_summary()
