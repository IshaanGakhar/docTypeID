# docTypeID — Legal Document Metadata Extraction Pipeline

End-to-end NLP pipeline for extracting structured metadata from US legal documents (PDF, DOCX, DOC, TXT), with optional cluster-aware consensus, LLM-based ground-truth generation, and LoRA fine-tuning of Gemma 3 270M.

---

## Table of Contents

1. [Workflow Overview](#workflow-overview)
2. [Extracted Fields](#extracted-fields)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Quick Start](#quick-start)
6. [Manual Setup](#manual-setup)
7. [Running the Pipeline](#running-the-pipeline)
8. [Groq Ground-Truth Extraction](#groq-ground-truth-extraction)
9. [LoRA Fine-Tuning (Gemma 3 270M)](#lora-fine-tuning-gemma-3-270m)
10. [TensorBoard & Visualizations](#tensorboard--visualizations)
11. [Tests](#tests)
12. [Configuration Reference](#configuration-reference)
13. [Output Schema](#output-schema)

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  Folder of .pdf / .docx / .doc / .txt  +  (optional) CSV        │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   pdf_loader.py      │  unified document ingestion
          │  (PDF/DOCX/DOC/TXT) │  preserves table order in DOCX
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │    preprocess.py     │  page splitting · line indexing
          │  → DocumentZones     │  caption / title / body zones
          └──────────┬──────────┘
                     │
        ┌────────────┴────────────────────────────────────┐
        │            Extraction modules (parallel)         │
        │                                                  │
        │  title_extractor      → title                   │
        │  doc_type_classifier  → document_types          │
        │  uid_extractor        → nuid (docket/case no.)  │
        │  court_judge_extractor→ court_name, location,   │
        │                          judge_name              │
        │  date_extractor       → filing_date             │
        │  party_extractor      → plaintiffs, defendants… │
        │  clause_extractor     → US Constitutional refs  │
        │  crf_ner              → CRF-based NER assist    │
        └────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  cluster_ingestion   │  (only when CSV supplied)
          │  · drop singletons   │  consensus across cluster docs
          │  · majority vote     │  three-way CSV verification
          │  · union parties     │  propagate filing_date
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │     results.json     │  one record per document
          │   (+ stdout JSONL)   │  with evidence & confidence
          └─────────────────────┘

  ── optional LLM track ──────────────────────────────────────────

  extract_with_groq.py   →  ground_truth.json   (Groq / gpt-oss-120b)
                                  │
  train_lora.py          ←────────┘   LoRA fine-tune Gemma 3 270M
                          →  gemma3-270m-lora-adapter/
                          →  TensorBoard events + matplotlib plots
```

---

## Extracted Fields

Every output record contains the following fields, each accompanied by `evidence` (the matched text snippet) and `confidence` (0–1 float):

| Field | Description |
|---|---|
| `title` | Document's own title in ALL CAPS |
| `document_types` | Multi-label list (e.g. `["Motion to Dismiss", "Memorandum"]`) |
| `nuid` | Normalised docket / case number |
| `court_name` | Full court name |
| `court_location` | District or county |
| `judge_name` | Presiding judge (without "Hon.") |
| `filing_date` | ISO-8601 date from signature block |
| `parties` | `{plaintiffs, defendants, petitioners, respondents}` |
| `clauses` | US Constitutional citations (amendments, articles, named clauses) |

---

## Project Structure

```
docTypeID/
├── pipeline/
│   ├── run_pipeline.py          # CLI orchestrator (folder & cluster modes)
│   ├── pdf_loader.py            # Unified document ingestion
│   ├── preprocess.py            # Text cleaning, zoning, line indexing
│   ├── config.py                # All regex patterns & thresholds
│   ├── title_extractor.py
│   ├── doc_type_classifier.py   # TF-IDF + rule-based classification
│   ├── uid_extractor.py         # Docket / case number extraction
│   ├── court_judge_extractor.py
│   ├── date_extractor.py        # Tiered date extraction
│   ├── party_extractor.py
│   ├── clause_extractor.py      # US Constitutional citation finder
│   ├── crf_ner.py               # CRF NER assistant
│   └── cluster_ingestion.py     # Cluster consensus & CSV verification
│
├── scripts/
│   ├── extract_with_groq.py     # Groq API → ground_truth.json
│   ├── train_lora.py            # LoRA fine-tuning + inference + plots
│   ├── train_crf.py             # CRF NER model training
│   └── train_doctype.py         # Doc-type classifier training
│
├── tests/
│   ├── test_smoke.py
│   └── test_regex_cases.py
│
├── data/
│   └── groq_labels/
│       └── ground_truth.json    # generated by extract_with_groq.py
│
├── requirements.txt
├── setup.sh                     # one-shot clone + install + key setup
└── .env                         # GROQ_API_KEY (not committed)
```

---

## Requirements

### Python packages

All packages are listed in `requirements.txt`. Key groups:

| Group | Packages |
|---|---|
| Document ingestion | `PyMuPDF`, `python-docx` |
| NLP / ML | `scikit-learn`, `sklearn-crfsuite`, `numpy`, `scipy`, `python-dateutil` |
| LLM / LoRA | `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl` |
| Groq API | `groq`, `python-dotenv`, `json-repair` |
| Monitoring | `tensorboard`, `matplotlib` |
| Testing | `pytest` |

### System dependencies (optional)

Legacy `.doc` (Word 97–2003) support requires one of:

```bash
sudo apt install antiword   # preferred
# or
sudo apt install catdoc
```

### Hardware

| Task | Minimum |
|---|---|
| NLP pipeline | Any CPU, ~512 MB RAM per worker |
| Groq extraction | Internet connection + Groq API key |
| LoRA training | CPU only (configured); ~16 GB RAM recommended for float32 |

---

## Quick Start

The fastest way to get going from scratch:

```bash
curl -fsSL https://raw.githubusercontent.com/IshaanGakhar/docTypeID/main/setup.sh | bash
```

Or clone first and run locally:

```bash
git clone https://github.com/IshaanGakhar/docTypeID.git
cd docTypeID
bash setup.sh
```

`setup.sh` will:
1. Clone the repository (if run via `curl`)
2. Create `.venv` and install all requirements
3. Prompt for your Groq API key and write `.env`
4. Print run instructions for every component

---

## Manual Setup

```bash
git clone https://github.com/IshaanGakhar/docTypeID.git
cd docTypeID

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Create .env with your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

---

## Running the Pipeline

### Activate the environment first

```bash
source .venv/bin/activate
```

### Folder mode (no cluster information)

Scan a folder and extract metadata from every supported document:

```bash
python pipeline/run_pipeline.py \
    --folder /path/to/docs/ \
    --output results.json
```

Each record is also printed as JSONL to stdout.

### Cluster mode (recommended)

When you have a CSV mapping documents to clusters (same legal matter), the pipeline:
- Drops singleton clusters (irrelevant isolated documents)
- Builds cluster-level consensus for court, parties, and dates
- Uses CSV fields (`court`, `location`, `all_entities`, `caption`, `docket_*`) to verify and augment per-document extraction

```bash
python pipeline/run_pipeline.py \
    --folder /path/to/docs/ \
    --csv    cluster_metadata.csv \
    --output results.json
```

Expected CSV columns include: `cluster_id`, `document_path`, `court`, `location`, `all_entities`, `caption`, `docket_primary_normalized`, `all_dockets_normalized`.

### Options

| Flag | Default | Description |
|---|---|---|
| `--folder PATH` | — | Directory to scan |
| `--csv PATH` | — | Cluster metadata CSV |
| `--output PATH` | `results.json` | Output file |
| `--workers N` | CPU count | Parallel worker processes |

---

## Groq Ground-Truth Extraction

Uses the Groq API (`openai/gpt-oss-120b`) to extract metadata from your existing documents, producing a `ground_truth.json` suitable for LoRA training.

```bash
python scripts/extract_with_groq.py \
    --folder /path/to/docs/ \
    --output data/groq_labels/ground_truth.json
```

The script uses only the first 2 and last 2 pages of each document to keep prompt size manageable.

### Options

| Flag | Default | Description |
|---|---|---|
| `--folder PATH` | — | Source documents folder |
| `--output PATH` | `ground_truth.json` | Output JSON file |
| `--resume` | off | Skip already-processed documents |
| `--workers N` | 4 | Concurrent API requests |

The API key is read from `.env` → `GROQ_API_KEY`.

---

## LoRA Fine-Tuning (Gemma 3 270M)

Fine-tunes `google/gemma-3-270m-it` with LoRA adapters on the ground-truth data generated above. Training runs entirely on CPU (float32).

### 1. Configure paths

Edit the top of `scripts/train_lora.py`:

```python
DOC_DIR     = Path("/path/to/source/docs")
LABELS_JSON = Path("data/groq_labels/ground_truth.json")
OUT_DIR     = Path("./gemma3-270m-lora-adapter")
```

### 2. Train

```bash
python scripts/train_lora.py
```

Training hyperparameters (set in `SFTConfig`):

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 1 (effective 16 via gradient accumulation) |
| Learning rate | 2e-4 with cosine schedule |
| Warmup steps | 20 |
| Max sequence length | 512 tokens |
| Loss | NLL (plain next-token prediction) |
| Gradient checkpointing | enabled |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | `q/k/v/o_proj`, `gate/up/down_proj` |

Loss and token accuracy are logged every 10 steps. Checkpoints are saved every 100 steps (last 2 kept).

### 3. Inference with the trained adapter

```bash
python scripts/train_lora.py --infer /path/to/doc.pdf
```

### 4. Inspect adapted layers

```bash
python scripts/train_lora.py --inspect-layers
```

---

## TensorBoard & Visualizations

Training metrics (loss, learning rate, token accuracy, gradient norm, entropy) are written to TensorBoard event files inside `gemma3-270m-lora-adapter/`.

### Live dashboard

```bash
tensorboard --logdir gemma3-270m-lora-adapter
# Open http://localhost:6006
```

### Static matplotlib plots

After training has finished, generate PNG charts:

```bash
# Save to default location (gemma3-270m-lora-adapter/plots/)
python scripts/train_lora.py --visualize

# Save to a custom directory
python scripts/train_lora.py --visualize --viz-out my_plots/
```

Generated files:

| File | Content |
|---|---|
| `loss.png` | Training loss over steps |
| `lr.png` | Learning rate schedule |
| `token_accuracy.png` | Mean token accuracy |
| `grad_norm.png` | Gradient norm |
| `entropy.png` | Per-token entropy |
| `summary.png` | Loss + LR side-by-side overview |

---

## Tests

```bash
pytest tests/
```

---

## Configuration Reference

All tuneable constants live in `pipeline/config.py`. Notable settings:

| Constant | Default | Effect |
|---|---|---|
| `MIN_CHARS` | 150 | Skip documents with fewer characters than this |
| `CAPTION_ZONE_LINES` | 40 | Lines from top treated as the caption zone |
| `TITLE_ZONE_LINES` | 60 | Lines from top searched for the document title |
| `DOCTYPE_THRESHOLD` | 0.5 | Minimum classifier probability to assign a type |
| `STRIP_REPEATED_LINES_IN_BODY` | `False` | Strip ECF running headers from body (keep off — headers contain extractable fields) |
| `CLASSIFIER_BACKEND` | `"lr"` | Classifier: `"lr"` (LogisticRegression), `"svm"`, or `"cnb"` |

`DOCTYPE_RULES` in `config.py` contains the full set of regex patterns used for rule-based document-type classification. Specific subtypes are listed before generic fallbacks so the most precise match wins.

---

## Output Schema

Each record in `results.json` follows this structure:

```json
{
  "file_path": "/abs/path/to/document.pdf",
  "title": {
    "value": "MOTION TO DISMISS",
    "confidence": 0.92,
    "evidence": "MOTION TO DISMISS"
  },
  "document_types": {
    "value": ["Motion to Dismiss", "Memorandum"],
    "confidence": 0.88,
    "evidence": "MOTION TO DISMISS … MEMORANDUM IN SUPPORT"
  },
  "nuid": {
    "value": "1:23-CV-04521-RJS",
    "confidence": 0.95,
    "evidence": "Case No. 1:23-CV-04521-RJS"
  },
  "court_name": {
    "value": "UNITED STATES DISTRICT COURT",
    "confidence": 0.95,
    "evidence": "UNITED STATES DISTRICT COURT"
  },
  "court_location": {
    "value": "Southern District of New York",
    "confidence": 0.90,
    "evidence": "FOR THE SOUTHERN DISTRICT OF NEW YORK"
  },
  "judge_name": {
    "value": "Robert J. Sullivan",
    "confidence": 0.85,
    "evidence": "Hon. Robert J. Sullivan"
  },
  "filing_date": {
    "value": "2023-09-28",
    "confidence": 0.95,
    "evidence": "Dated: September 28, 2023"
  },
  "parties": {
    "value": {
      "plaintiffs":  ["Alice Corp."],
      "defendants":  ["Bob Inc."],
      "petitioners": [],
      "respondents": []
    },
    "confidence": 0.80,
    "evidence": "Alice Corp. v. Bob Inc."
  },
  "clauses": {
    "value": ["Fourth Amendment", "Due Process Clause"],
    "confidence": 0.90,
    "evidence": "Fourth Amendment … Due Process Clause of the Fourteenth Amendment"
  }
}
```
