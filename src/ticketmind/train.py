"""Çalıştır: python -m ticketmind.train. Test verisi ayar seçiminde kullanılmaz."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ticketmind import __version__
from ticketmind.data import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_REPORT, LABELS, SPLITS, load_records
from ticketmind.text import normalize_text


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_text, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
        )),
        ("classifier", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def evaluate(model, rows: list[dict]) -> dict:
    expected = [row["label"] for row in rows]
    predicted = model.predict([row["text"] for row in rows]).tolist()
    return {
        "n": len(rows),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, labels=list(LABELS), average="macro", zero_division=0),
        "per_class": classification_report(expected, predicted, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(LABELS)).tolist(),
        "confusion_labels": list(LABELS),
        "predictions": [{"id": row["id"], "expected": row["label"], "predicted": prediction}
                        for row, prediction in zip(rows, predicted)],
    }


def train(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL, report_path: Path = DEFAULT_REPORT) -> dict:
    rows = load_records(data_path)
    subsets = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    model = make_pipeline()
    # Pipeline.fit yalnız train satırlarını görür: sözlük/IDF dahil.
    model.fit([row["text"] for row in subsets["train"]], [row["label"] for row in subsets["train"]])
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([row["text"] for row in subsets["train"]], [row["label"] for row in subsets["train"]])
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata = {
        "artifact_schema": 1,
        "model_version": f"tfidf-lr-{__version__}-{data_sha[:12]}",
        "dataset_type": "synthetic_demo",
        "data_sha256": data_sha,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "labels": list(LABELS),
        "review_policy": "always_review_demo",
        "random_state": 42,
    }
    report = {
        **metadata,
        "model": "TF-IDF word (1,2) + LogisticRegression(C=2, class_weight=balanced)",
        "selection_policy": "Fixed educational configuration; no test-driven tuning; no OOD detector or calibrated review threshold.",
        "limitations": [
            "AI-authored synthetic data; scores do not estimate real customer performance.",
            "Scenario groups are split-disjoint; this does not guarantee semantic independence.",
            "All predictions require human review; model scores are uncalibrated.",
            "Entity rules are not trained NER and have no held-out extraction score yet.",
        ],
        "split_summary": {split: {"n": len(items), "groups": len({row['group_id'] for row in items}),
                                   "labels": dict(Counter(row["label"] for row in items))}
                          for split, items in subsets.items()},
        "validation": evaluate(model, subsets["validation"]),
        "test": evaluate(model, subsets["test"]),
        "dummy_test": evaluate(dummy, subsets["test"]),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "pipeline": model}, model_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    per_class = report["test"]["per_class"]
    lines = ["# Başlangıç modeli: sentetik demo değerlendirmesi", "",
             "Bu sonuçlar AI ile yazılmış küçük demo verisine aittir; gerçek müşteri mesajlarında başarı iddiası taşımaz.", "",
             f"Veri SHA-256: `{data_sha}`", "", "| Bölme | Mesaj | Grup |", "|---|---:|---:|"]
    lines += [f"| {split} | {report['split_summary'][split]['n']} | {report['split_summary'][split]['groups']} |" for split in SPLITS]
    lines += ["", "| Model | Test macro F1 | Test accuracy |", "|---|---:|---:|",
              f"| Most-frequent dummy | {report['dummy_test']['macro_f1']:.3f} | {report['dummy_test']['accuracy']:.3f} |",
              f"| TF-IDF + LR | {report['test']['macro_f1']:.3f} | {report['test']['accuracy']:.3f} |", "",
              f"Doğrulama macro F1: {report['validation']['macro_f1']:.3f}. Ayarlar sabittir; testten eşik veya hiperparametre seçilmedi.", "",
              "| Etiket | Precision | Recall | F1 | Test örneği |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {label} | {per_class[label]['precision']:.3f} | {per_class[label]['recall']:.3f} | {per_class[label]['f1-score']:.3f} | {int(per_class[label]['support'])} |" for label in LABELS]
    lines += ["", "## Hata matrisi", "", "Satır gerçek, sütun tahmin edilen etikettir.", "",
              "| Gerçek / Tahmin | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)]
    lines += ["| " + label + " | " + " | ".join(str(number) for number in values) + " |"
              for label, values in zip(LABELS, report['test']['confusion_matrix'])]
    lines += ["", "## Yanlış sınıflandırılan test örnekleri", ""]
    lookup = {row["id"]: row["text"] for row in rows}
    mistakes = [row for row in report["test"]["predictions"] if row["expected"] != row["predicted"]]
    lines += [f"- `{row['id']}`: {row['expected']} → {row['predicted']}: {lookup[row['id']]}" for row in mistakes] or ["Bu küçük test kümesinde hata yok; bu, gerçek kullanım başarısını kanıtlamaz."]
    lines += ["", "## Sınırlar", ""] + ["- " + item for item in report["limitations"]]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketMind-TR başlangıç modelini eğit ve raporla.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = train(args.data, args.model, args.report)
    print(json.dumps({"model": str(args.model), "report": str(args.report),
                      "test_macro_f1": report["test"]["macro_f1"], "dataset_type": "synthetic_demo"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

