"""
Train the CRF NER model for legal entities (COURT, JUDGE, DATE, CASE_NO, PARTY).

Input: a JSONL file where each line is:
  {"tokens": [str, ...], "labels": [str, ...]}

  Labels use BIO encoding:
    B-COURT, I-COURT, B-JUDGE, I-JUDGE, B-DATE, I-DATE,
    B-CASE_NO, I-CASE_NO, B-PARTY, I-PARTY, O

Output: saves CRF model to models/crf_ner.joblib

Usage:
    python scripts/train_crf.py --input data/ner_train.jsonl
    python scripts/train_crf.py --input data/ner_train.jsonl --c1 0.1 --c2 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import MODEL_CRF_PATH, CRF_ENTITY_LABELS
from pipeline.crf_ner import sent_to_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ner_jsonl(path: Path) -> tuple[list[list[dict]], list[list[str]]]:
    """
    Returns (X_features, y_labels) where each element is one sentence/document.
    """
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj.get("tokens", [])
            labels = obj.get("labels", [])
            if len(tokens) != len(labels):
                print(f"  WARNING: token/label length mismatch, skipping")
                continue
            if tokens:
                X.append(sent_to_features(tokens))
                y.append(labels)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_crf(
    jsonl_path: Path,
    c1: float = 0.1,
    c2: float = 0.01,
    max_iter: int = 100,
    output_dir: Path = Path("models"),
) -> None:
    try:
        import sklearn_crfsuite
    except ImportError:
        print("ERROR: sklearn-crfsuite not installed. Run: pip install sklearn-crfsuite")
        sys.exit(1)

    import joblib

    print(f"Loading training data from '{jsonl_path}'…")
    X_train, y_train = load_ner_jsonl(jsonl_path)

    if not X_train:
        print("ERROR: No training examples found.")
        sys.exit(1)

    print(f"  {len(X_train)} sentences loaded.")

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iter,
        all_possible_transitions=True,
    )

    print("Training CRF…")
    crf.fit(X_train, y_train)
    print("  Done.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, MODEL_CRF_PATH)
    print(f"Saved CRF model to '{MODEL_CRF_PATH}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRF NER model for legal entities")
    parser.add_argument("--input",    required=True, help="JSONL training file")
    parser.add_argument("--c1",       type=float, default=0.1,  help="L1 regularization (default: 0.1)")
    parser.add_argument("--c2",       type=float, default=0.01, help="L2 regularization (default: 0.01)")
    parser.add_argument("--max-iter", type=int,   default=100,  help="Max CRF iterations (default: 100)")
    parser.add_argument("--output",   default="models",         help="Output directory (default: models)")
    args = parser.parse_args()

    train_crf(
        jsonl_path=Path(args.input),
        c1=args.c1,
        c2=args.c2,
        max_iter=args.max_iter,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
