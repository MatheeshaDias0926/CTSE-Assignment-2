"""
Central Configuration for Smart Factory MAS
============================================
All configurable parameters for the Multi-Agent System.
Runs entirely locally — zero external API calls.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Data file paths
PRODUCTION_CSV_PATH = os.path.join(DATA_DIR, "production_log.csv")
INVENTORY_DB_PATH = os.path.join(DATA_DIR, "inventory.db")

# ---------------------------------------------------------------------------
# Ollama / LLM Configuration (100 % Local)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.1          # Low for deterministic, factual outputs
MAX_TOKENS = 2048          # Reasonable context for 8B model
REQUEST_TIMEOUT = 120      # Seconds – local inference can be slow

# ---------------------------------------------------------------------------
# LangGraph Workflow Configuration
# ---------------------------------------------------------------------------
MAX_COORDINATOR_ITERATIONS = 10   # Safety limit on coordinator loop
RECURSION_LIMIT = 50              # LangGraph recursion guard

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_MODEL = OLLAMA_MODEL         # Same model used for LLM-as-Judge
EVAL_TEMPERATURE = 0.0            # Fully deterministic for evaluation

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
for _dir in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)
