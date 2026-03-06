"""
Train the multi-label document type classifier.

Input: a JSONL or JSON file with records containing:
  {"text": str, "labels": [str, ...]}

  OR a directory of .txt / .pdf files with a companion labels.json:
  {"filename.txt": ["Motion", "Brief"], ...}

Output: saves model files to models/ directory.

Usage:
    python scripts/train_doctype.py --input data/train.jsonl
    python scripts/train_doctype.py --input data/docs/ --labels data/labels.json
    python scripts/train_doctype.py --input data/train.jsonl --backend svm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    MODEL_DOCTYPE_PATH,
    LABEL_BINARIZER_PATH,
    VECTORIZER_WORD_PATH,
    VECTORIZER_CHAR_PATH,
    DOCUMENT_TYPE_LABELS,
    CLASSIFIER_BACKEND,
    DOCTYPE_THRESHOLD,
)
from pipeline.doc_type_classifier import compute_binary_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[tuple[str, list[str]]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text   = obj.get("text", "")
            labels = obj.get("labels", [])
            if text:
                records.append((text, labels))
    return records


def load_json(path: Path) -> list[tuple[str, list[str]]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [(item["text"], item["labels"]) for item in data if "text" in item]
    return []


def load_directory(doc_dir: Path, labels_json: Path) -> list[tuple[str, list[str]]]:
    with open(labels_json) as f:
        label_map: dict[str, list[str]] = json.load(f)

    records = []
    for fname, labels in label_map.items():
        fpath = doc_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        text = fpath.read_text(encoding="utf-8", errors="replace")
        records.append((text, labels))
    return records


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(texts: list[str], vec_word, vec_char, fit: bool = False):
    if fit:
        X_word = vec_word.fit_transform(texts)
        X_char = vec_char.fit_transform(texts)
    else:
        X_word = vec_word.transform(texts)
        X_char = vec_char.transform(texts)

    X_bin_rows = [compute_binary_features(t) for t in texts]
    X_bin = csr_matrix(np.vstack(X_bin_rows))

    return hstack([X_word, X_char, X_bin])


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def make_classifier(backend: str):
    from sklearn.multiclass import OneVsRestClassifier

    if backend == "lr":
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )
    elif backend == "svm":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)
        )
    elif backend == "cnb":
        from sklearn.naive_bayes import ComplementNB
        base = ComplementNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: lr, svm, cnb")

    return OneVsRestClassifier(base)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    records: list[tuple[str, list[str]]],
    backend: str = CLASSIFIER_BACKEND,
    output_dir: Path = Path("models"),
) -> None:
    if not records:
        print("ERROR: no training records found.")
        sys.exit(1)

    print(f"Training on {len(records)} documents with backend='{backend}'")

    texts  = [r[0] for r in records]
    labels = [r[1] for r in records]

    # Label binarizer
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=DOCUMENT_TYPE_LABELS)
    Y   = mlb.fit_transform(labels)

    print(f"  Label classes: {list(mlb.classes_)}")
    print(f"  Label matrix shape: {Y.shape}")

    # Vectorizers
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.95,
        sublinear_tf=True, analyzer="word",
    )
    vec_char = TfidfVectorizer(
        ngram_range=(3, 5), min_df=1, max_df=0.95,
        sublinear_tf=True, analyzer="char_wb",
    )

    X = build_features(texts, vec_word, vec_char, fit=True)
    print(f"  Feature matrix shape: {X.shape}")

    clf = make_classifier(backend)
    clf.fit(X, Y)
    print("  Classifier trained.")

    # Save artifacts
    import joblib
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf,      MODEL_DOCTYPE_PATH)
    joblib.dump(mlb,      LABEL_BINARIZER_PATH)
    joblib.dump(vec_word, VECTORIZER_WORD_PATH)
    joblib.dump(vec_char, VECTORIZER_CHAR_PATH)

    print(f"\nSaved model artifacts to '{output_dir}/':")
    for p in [MODEL_DOCTYPE_PATH, LABEL_BINARIZER_PATH, VECTORIZER_WORD_PATH, VECTORIZER_CHAR_PATH]:
        print(f"  {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-label document type classifier")
    parser.add_argument("--input",   required=True, help="JSONL/JSON training file or document directory")
    parser.add_argument("--labels",  help="Labels JSON file (required when --input is a directory)")
    parser.add_argument("--backend", default=CLASSIFIER_BACKEND, choices=["lr", "svm", "cnb"],
                        help="Classifier backend (default: lr)")
    parser.add_argument("--output",  default="models", help="Output directory for model files")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        if not args.labels:
            print("ERROR: --labels required when --input is a directory.")
            sys.exit(1)
        records = load_directory(input_path, Path(args.labels))
    elif input_path.suffix == ".jsonl":
        records = load_jsonl(input_path)
    else:
        records = load_json(input_path)

    train(records, backend=args.backend, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
