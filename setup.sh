#!/usr/bin/env bash
# =============================================================================
#  docTypeID — one-shot setup script
#
#  What this script does:
#    1. Clone the repository
#    2. Create a Python virtual environment and install all requirements
#    3. Prompt for the Groq API key and write .env
#    4. Print how to run each component
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/IshaanGakhar/docTypeID.git"
REPO_DIR="docTypeID"
PYTHON="${PYTHON:-python3}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
section() { echo -e "\n${BOLD}══════════════════════════════════════════${RESET}"; echo -e "${BOLD}  $*${RESET}"; echo -e "${BOLD}══════════════════════════════════════════${RESET}"; }

# =============================================================================
# 1. Clone
# =============================================================================
section "1 / 4  Clone repository"

if [ -d "$REPO_DIR" ]; then
    warn "Directory '$REPO_DIR' already exists — skipping clone."
    warn "To get a fresh copy:  rm -rf $REPO_DIR  and re-run this script."
else
    info "Cloning $REPO_URL …"
    git clone "$REPO_URL" "$REPO_DIR"
    ok "Cloned into ./$REPO_DIR"
fi

cd "$REPO_DIR"

# =============================================================================
# 2. Python virtual environment + dependencies
# =============================================================================
section "2 / 4  Install requirements"

if ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}[ERR]${RESET}  '$PYTHON' not found. Install Python 3.10+ and retry."
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using $PYTHON  (version $PY_VER)"

if [ ! -d ".venv" ]; then
    info "Creating virtual environment (.venv) …"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment activated."

info "Upgrading pip …"
pip install --quiet --upgrade pip

info "Installing requirements.txt …"
pip install --quiet -r requirements.txt

# LoRA / LLM extras not in requirements.txt for lighter installs
info "Installing LoRA / LLM extras (transformers, peft, trl, bitsandbytes) …"
pip install --quiet -U \
    "transformers>=4.47.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "peft>=0.11.0" \
    "trl>=0.9.0" \
    "bitsandbytes>=0.43.0" \
    "groq>=0.9.0" \
    "tensorboard>=2.16.0" \
    "matplotlib>=3.8.0" \
    "json-repair>=0.30.0" \
    "python-dotenv>=1.0.0"

ok "All dependencies installed."

# Optional system tools for .doc support
if ! command -v antiword &>/dev/null && ! command -v catdoc &>/dev/null; then
    warn "Neither 'antiword' nor 'catdoc' found — .doc (legacy Word) support will be limited."
    warn "Install with:  sudo apt install antiword   or   sudo apt install catdoc"
fi

# =============================================================================
# 3. Groq API key
# =============================================================================
section "3 / 4  Groq API Key"

ENV_FILE=".env"

if [ -f "$ENV_FILE" ] && grep -q "^GROQ_API_KEY=" "$ENV_FILE" 2>/dev/null; then
    EXISTING_KEY=$(grep "^GROQ_API_KEY=" "$ENV_FILE" | cut -d'=' -f2-)
    if [ -n "$EXISTING_KEY" ] && [ "$EXISTING_KEY" != "your_key_here" ]; then
        ok "Existing .env found with a GROQ_API_KEY — skipping prompt."
        SKIP_KEY=true
    fi
fi

if [ "${SKIP_KEY:-false}" = "false" ]; then
    echo
    echo -e "  ${YELLOW}A Groq API key is required to run${RESET}  scripts/extract_with_groq.py"
    echo -e "  Get one free at: ${CYAN}https://console.groq.com/keys${RESET}"
    echo
    read -rp "  Paste your Groq API key (press Enter to skip): " GROQ_KEY

    if [ -z "$GROQ_KEY" ]; then
        warn "No key entered. You can add it later by editing .env:"
        warn "  GROQ_API_KEY=your_key_here"
        GROQ_KEY="your_key_here"
    fi

    # Write/overwrite the key in .env
    if [ -f "$ENV_FILE" ]; then
        # Replace existing line if present, otherwise append
        if grep -q "^GROQ_API_KEY=" "$ENV_FILE"; then
            sed -i "s|^GROQ_API_KEY=.*|GROQ_API_KEY=${GROQ_KEY}|" "$ENV_FILE"
        else
            echo "GROQ_API_KEY=${GROQ_KEY}" >> "$ENV_FILE"
        fi
    else
        echo "GROQ_API_KEY=${GROQ_KEY}" > "$ENV_FILE"
    fi

    ok ".env written."
fi

# =============================================================================
# 4. Usage instructions
# =============================================================================
section "4 / 4  How to run"

VENV_ACTIVATE="source $(pwd)/.venv/bin/activate"

cat << EOF

${BOLD}Activate the environment (run this every new shell session):${RESET}
  ${CYAN}${VENV_ACTIVATE}${RESET}

──────────────────────────────────────────────────────────────
${BOLD}A) NLP Metadata Extraction Pipeline${RESET}
──────────────────────────────────────────────────────────────

  # Process a folder of documents (PDF/DOCX/DOC/TXT):
  ${CYAN}python pipeline/run_pipeline.py --folder /path/to/docs --output results.json${RESET}

  # With a cluster CSV for consensus + verification:
  ${CYAN}python pipeline/run_pipeline.py --folder /path/to/docs --csv cluster.csv --output results.json${RESET}

──────────────────────────────────────────────────────────────
${BOLD}B) Groq Ground-Truth Extraction  (scripts/extract_with_groq.py)${RESET}
──────────────────────────────────────────────────────────────

  ${CYAN}python scripts/extract_with_groq.py --folder /path/to/docs --output data/groq_labels/ground_truth.json${RESET}

  Options:
    --resume       Resume a partially completed run
    --workers N    Parallel workers (default: 4)

──────────────────────────────────────────────────────────────
${BOLD}C) LoRA Fine-tuning  (scripts/train_lora.py)${RESET}
──────────────────────────────────────────────────────────────

  # Edit DOC_DIR and LABELS_JSON at the top of the script first, then:
  ${CYAN}python scripts/train_lora.py${RESET}

  # Run inference on a single document:
  ${CYAN}python scripts/train_lora.py --infer /path/to/doc.pdf${RESET}

  # Run inference on a whole folder (saves to gemma3-270m-lora-adapter/inference_results.json):
  ${CYAN}python scripts/train_lora.py --infer-folder /path/to/docs/${RESET}

  # Inspect adapted layers:
  ${CYAN}python scripts/train_lora.py --inspect-layers${RESET}

  # Generate training-curve plots (after training):
  ${CYAN}python scripts/train_lora.py --visualize${RESET}
  ${CYAN}python scripts/train_lora.py --visualize --viz-out my_plots/${RESET}

  # Live TensorBoard dashboard:
  ${CYAN}tensorboard --logdir gemma3-270m-lora-adapter${RESET}
  # then open http://localhost:6006 in your browser

──────────────────────────────────────────────────────────────
${BOLD}D) Tests${RESET}
──────────────────────────────────────────────────────────────

  ${CYAN}pytest tests/${RESET}

──────────────────────────────────────────────────────────────

${GREEN}Setup complete. Happy extracting!${RESET}

EOF
