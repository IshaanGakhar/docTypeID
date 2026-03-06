"""
Generate synthetic legal documents for pipeline testing using Groq.

Usage:
    export GROQ_API_KEY=your_key_here
    python scripts/generate_synthetic_docs.py
    python scripts/generate_synthetic_docs.py --count 20 --out data/synthetic
    python scripts/generate_synthetic_docs.py --count 5 --start-at 21   # resume

Each call produces:
    {OUT_DIR}/doc_NNN.txt          — document text (feed directly to pipeline)
    {OUT_DIR}/ground_truth.json    — combined labels file for train_lora.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL    = "meta-llama/llama-4-maverick-17b-128e-instruct"  # change to your Groq model name e.g. "llama-3.3-70b-versatile"
OUT_DIR  = Path("data/synthetic")
N_DOCS   = 20
DELAY_S  = 1.5      # seconds between requests (stay under rate limit)

# Explicit rotation so every generation gets a different document type
DOC_TYPES = [
    "Motion to Dismiss",
    "Complaint",
    "Class Action Complaint",
    "Motion for Summary Judgment",
    "Answer",
    "Memorandum of Law",
    "Notice of Settlement",
    "Reply Brief",
    "Opposition Brief",
    "Notice of Removal",
    "Order",
    "Stipulation",
    "Declaration",
    "Petition",
    "Judgment",
    "Motion to Stay",
]

# ---------------------------------------------------------------------------
# System prompt — same one used for LoRA training
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a legal document generator. Generate realistic synthetic US federal civil litigation documents \
for testing an NLP metadata extraction pipeline.

The pipeline extracts these fields from each document:
  title, document_types, nuid (case/docket number), court_name, court_location,
  judge_name, filing_date, parties (plaintiffs/defendants), clauses (US Constitutional citations)

Each generated document must include a GROUND TRUTH block at the very end \
(after a "---END DOCUMENT---" separator) with the correct values for all fields in JSON format.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT TYPE (pick one per generation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose randomly from:
  Motion to Dismiss, Motion to Stay, Motion for Summary Judgment,
  Memorandum of Law, Complaint, Class Action Complaint, Answer,
  Reply Brief, Opposition Brief, Notice of Removal, Order,
  Stipulation, Declaration, Petition, Judgment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURAL REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CAPTION (top of page 1, as a two-column table — represent as pipe-separated text):
   - Left column: party block with plaintiff(s) "v." defendant(s), each labeled (Plaintiff/Defendant)
   - Right column: court name, district, case/docket number, judge name with "Hon." prefix
   - Vary the case number format across generations:
       Standard federal:   1:23-cv-04521-RJS
       Multi-district:     MDL No. 3:22-md-03047-YGR
       Two-digit year:     23-cv-00891-LTS
       State-style:        Index No. 2023/451892
   - Occasionally use "Case. No." (period after Case) to test parser robustness
   - Include the judge's initials as the docket suffix (e.g., judge "Robert J. Sullivan" → suffix -RJS)

2. TITLE BLOCK (after caption, in ALL CAPS):
   - Full descriptive title, e.g.:
       MEMORANDUM OF LAW IN SUPPORT OF DEFENDANT'S MOTION TO DISMISS
       PLAINTIFFS' CLASS ACTION COMPLAINT FOR VIOLATIONS OF SECURITIES LAW
   - Occasionally make it compound, spanning two lines joined by "AND" or "FOR"
   - ~15% of the time, add a filing stamp above the title:
       E-FILED
       [CLERK NAME IN ALL CAPS]
       [DATE]
       DISTRICT CLERK
     (The title must still appear AFTER the stamp on the same page)

3. TABLE OF CONTENTS (when doc type is Memorandum, Brief, Complaint — ~60% of time):
   - Include section entries with dotted leaders and page numbers, e.g.:
       I. FACTUAL BACKGROUND .............. 3
       II. ARGUMENT ........................ 7
   - Include a TABLE OF AUTHORITIES with case citations like:
       Smith v. Jones, No. 19-cv-04821 (S.D.N.Y. Mar. 15, 2019) .... 5
       Roe v. Corp., 2021 U.S. Dist. LEXIS 84729 (C.D. Cal. Apr. 2, 2021) .. 8
   - These cited dates must NOT be extracted as the filing date

4. BODY (2-5 paragraphs, legally plausible):
   - Reference at least one US Constitutional provision when relevant:
       Fourth Amendment (unreasonable search and seizure)
       Fifth Amendment Due Process Clause
       First Amendment (free speech)
       Fourteenth Amendment Equal Protection Clause
       Commerce Clause (Art. I, § 8, cl. 3)
       Supremacy Clause (Art. VI, cl. 2)
   - Include phrases like "Pursuant to Fed. R. Civ. P. 12(b)(6)" or "15 U.S.C. § 78j(b)"

5. SIGNATURE BLOCK (last page):
   - Format:
       Dated: [Month DD, YYYY]          Respectfully submitted,
       /s/ [Attorney Name]
       [LAW FIRM NAME LLP]
       [Street Address]
       [City, State ZIP]
       Tel: (XXX) XXX-XXXX
       Email: attorney@firm.com
       Counsel for [Plaintiff/Defendant]
   - The Dated: line is the TRUE filing date — it must appear on the LAST page

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIATION MATRIX — rotate through these across generations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Courts (pick one):
  UNITED STATES DISTRICT COURT, SOUTHERN DISTRICT OF NEW YORK
  UNITED STATES DISTRICT COURT, NORTHERN DISTRICT OF CALIFORNIA
  UNITED STATES DISTRICT COURT, CENTRAL DISTRICT OF CALIFORNIA
  UNITED STATES DISTRICT COURT, NORTHERN DISTRICT OF TEXAS, DALLAS DIVISION
  UNITED STATES DISTRICT COURT, DISTRICT OF DELAWARE

Party complexity (rotate):
  - Single plaintiff v. single defendant
  - Single plaintiff v. multiple defendants (3-5, comma-separated)
  - Class action: "[Name], individually and on behalf of all others similarly situated"
  - Multi-plaintiff: "[Name A] and [Name B], individually and on behalf of..."
  - Corporate defendants with suffixes: Inc., LLC, Corp., LLP, N.A., S.A.

Date placement traps (to test extractor robustness):
  - Put a year in the docket number (e.g., "23-cv-04521")
  - Put a date in a cited case inside the Table of Authorities (older than filing date)
  - Use the filing date in one of: "Filed: [date]", "Dated: [date]", "Date: [date]",
    "Filing Date - [date]", "Entered: [date]"
  - Occasionally use ISO format (2023-09-28) in the Dated: line

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND TRUTH (append after "---END DOCUMENT---"):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "title": "<exact ALL-CAPS title as it appears in the document>",
  "document_types": ["<primary type>", "<secondary if compound>"],
  "nuid": "<normalized docket, e.g. 1:23-CV-04521-RJS>",
  "court_name": "UNITED STATES DISTRICT COURT",
  "court_location": "<full district name, e.g. Southern District of New York>",
  "judge_name": "<full name without Hon., e.g. Robert J. Sullivan>",
  "filing_date": "<ISO-8601, e.g. 2023-09-28>",
  "parties": {
    "plaintiffs": ["<name as it appears in caption>"],
    "defendants": ["<name as it appears in caption>"]
  },
  "clauses": ["<Amendment or Article name if cited, else empty list>"]
}\
"""

_SEPARATOR = "---END DOCUMENT---"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str, doc_num: int, doc_path: Path) -> dict | None:
    """
    Split model output at the separator and extract ground-truth JSON.
    Returns the ground-truth dict with file_path injected, or None on failure.
    """
    if _SEPARATOR not in text:
        print(f"  [warn] doc {doc_num}: separator not found — saving raw text anyway")
        return None

    _doc_text, gt_part = text.split(_SEPARATOR, 1)

    # Extract the first JSON object from the ground-truth block
    match = re.search(r'\{.*\}', gt_part, re.DOTALL)
    if not match:
        print(f"  [warn] doc {doc_num}: no JSON found after separator")
        return None

    try:
        gt = json.loads(match.group())
    except json.JSONDecodeError as e:
        # Try a light repair pass
        try:
            import json_repair
            gt = json_repair.loads(match.group())
        except Exception:
            print(f"  [warn] doc {doc_num}: JSON parse failed — {e}")
            return None

    gt["file_path"] = str(doc_path.resolve())
    return gt


def _doc_text_only(text: str) -> str:
    """Return everything before the separator (the document body)."""
    if _SEPARATOR in text:
        return text.split(_SEPARATOR, 1)[0].strip()
    return text.strip()


def _save_pdf(doc_text: str, pdf_path: Path) -> None:
    """
    Render plain text as a multi-page PDF using ReportLab.

    Formatting rules:
      - ALL-CAPS lines              → bold, centered  (titles / headings)
      - Lines starting with /s/    → italic           (signature)
      - Pipe-separated lines (|)   → monospace        (caption tables)
      - Everything else            → normal body text
    """
    styles = getSampleStyleSheet()

    normal = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=10, leading=14,
        leftIndent=0, rightIndent=0, spaceAfter=4,
    )
    heading = ParagraphStyle(
        "heading", parent=normal,
        fontName="Times-Bold", fontSize=10, leading=14,
        alignment=TA_CENTER, spaceAfter=6,
    )
    mono = ParagraphStyle(
        "mono", parent=normal,
        fontName="Courier", fontSize=9, leading=12,
    )
    italic = ParagraphStyle(
        "italic", parent=normal,
        fontName="Times-Italic",
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=1.25 * inch, rightMargin=1.25 * inch,
        topMargin=1.0 * inch,   bottomMargin=1.0 * inch,
    )

    story = []
    for raw_line in doc_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        # Escape XML special chars for ReportLab
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if "|" in line:
            story.append(Paragraph(safe, mono))
        elif line.startswith("/s/"):
            story.append(Paragraph(safe, italic))
        elif line == line.upper() and len(line) > 4 and line[0].isalpha():
            story.append(Paragraph(safe, heading))
        else:
            story.append(Paragraph(safe, normal))

    doc.build(story)


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate(count: int, start_at: int, out_dir: Path) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found — set it in .env or as an environment variable")

    client = Groq(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_file = out_dir / "ground_truth.json"
    existing_gt: list[dict] = []
    if gt_file.exists():
        with open(gt_file) as f:
            existing_gt = json.load(f)

    new_gt: list[dict] = []
    succeeded = 0
    failed    = 0

    for i in range(count):
        doc_num   = start_at + i
        doc_type  = DOC_TYPES[(doc_num - 1) % len(DOC_TYPES)]
        doc_path  = out_dir / f"doc_{doc_num:03d}.pdf"

        print(f"Generating doc {doc_num:03d} [{doc_type}] ...", end=" ", flush=True)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": (
                        f"Generate document #{doc_num}. "
                        f"The document type MUST be: {doc_type}. "
                        "Vary the court, judge, and party structure from previous documents."
                    )},
                ],
                temperature=0.9,
                max_tokens=3000,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1
            time.sleep(DELAY_S * 2)
            continue

        doc_text = _doc_text_only(raw)

        # Save as PDF
        try:
            _save_pdf(doc_text, doc_path)
        except Exception as e:
            print(f"[pdf-warn] {e} — falling back to .txt")
            doc_path = doc_path.with_suffix(".txt")
            doc_path.write_text(doc_text, encoding="utf-8")

        gt = _parse_response(raw, doc_num, doc_path)
        if gt:
            new_gt.append(gt)
            print(f"OK")
        else:
            (out_dir / f"doc_{doc_num:03d}_raw.txt").write_text(raw, encoding="utf-8")
            print("partial (no ground truth)")

        succeeded += 1
        time.sleep(DELAY_S)

    # Merge with any existing ground truth and write
    all_gt = existing_gt + new_gt
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {succeeded} generated, {failed} failed.")
    print(f"Documents : {out_dir}/")
    print(f"Labels    : {gt_file}  ({len(all_gt)} total records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic legal docs via Groq")
    parser.add_argument("--count",    type=int, default=N_DOCS,
                        help=f"Number of documents to generate (default {N_DOCS})")
    parser.add_argument("--out",      type=Path, default=OUT_DIR,
                        help=f"Output folder (default {OUT_DIR})")
    parser.add_argument("--start-at", type=int, default=1, metavar="N",
                        help="Starting document number, useful for resuming (default 1)")
    parser.add_argument("--model",    default=MODEL,
                        help=f"Groq model name (default {MODEL})")
    args = parser.parse_args()

    MODEL = args.model
    generate(count=args.count, start_at=args.start_at, out_dir=args.out)
