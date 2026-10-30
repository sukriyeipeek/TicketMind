"""Fixed word/character baselines and optional real TinyBERT comparison.

The inspected legacy test is regression evidence only. Selection reads validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ticketmind import __version__
from ticketmind.data import DEFAULT_DATA, LABELS, PROJECT_ROOT, load_records
from ticketmind.text import normalize_text
from ticketmind.train import make_pipeline


MODEL_ORDER = ("word", "char", "tinybert")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_artifact(path: Path, artifact: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(artifact, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_char_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize_text,
            lowercase=False, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])


def split_for_experiment(rows: list[dict]) -> dict[str, list[dict]]:
    """Keep original split membership explicit; never merge inspected dev files."""
    subsets = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    if any(not value for value in subsets.values()):
        raise ValueError("Karşılaştırma train, validation ve legacy test bölmelerini gerektirir.")
    return subsets


def evaluate_model(model, rows: list[dict]) -> dict:
    expected = [row["label"] for row in rows]
    started = perf_counter()
    predicted = model.predict([row["text"] for row in rows]).tolist()
    elapsed = (perf_counter() - started) * 1000
    predictions = [
        {"id": row["id"], "expected": row["label"], "predicted": prediction}
        for row, prediction in zip(rows, predicted)
    ]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "per_class": classification_report(
            expected, predicted, labels=list(LABELS), output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(LABELS)).tolist(),
        "confusion_labels": list(LABELS),
        "predictions": predictions,
        "errors": [
            {**prediction, "text": row["text"]}
            for prediction, row in zip(predictions, rows)
            if prediction["expected"] != prediction["predicted"]
        ],
        "predict_ms": elapsed,
        "predict_ms_per_message": elapsed / len(rows),
        "timing_scope": "one local batch, wall time; no load/download time; not a production latency benchmark",
    }


def select_model(results: dict[str, dict]) -> str:
    """Select solely by validation macro F1; fixed order breaks exact ties."""
    available = [name for name in MODEL_ORDER if name in results]
    if not available:
        raise ValueError("En az bir model sonucu gerekli.")
    return max(available, key=lambda name: results[name]["validation"]["macro_f1"])


def _metadata(name: str, rows: list[dict], data_sha: str) -> dict:
    from ticketmind.transformer_model import training_fingerprint

    return {
        "artifact_schema": 1,
        "model_kind": f"tfidf_{name}",
        "model_version": f"tfidf-{name}-{__version__}-{data_sha[:12]}",
        "dataset_type": "synthetic_demo",
        "review_policy": "always_review_demo",
        "data_sha256": data_sha,
        "train_sha256": training_fingerprint(rows),
        "labels": list(LABELS),
        "training_ids": [row["id"] for row in rows],
        "training_groups": sorted({row["group_id"] for row in rows}),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "random_state": 42,
        "hyperparameters": {
            "analyzer": "word" if name == "word" else "char_wb",
            "ngram_range": [1, 2] if name == "word" else [3, 5],
            "sublinear_tf": True, "C": 2.0, "class_weight": "balanced", "max_iter": 1000,
        },
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Sabit model karşılaştırması", "",
        "AI ile yazılmış sentetik demo verisi. Ölçümler gerçek müşteri başarısını göstermez.", "",
        "Seçim yalnız validation macro F1 ile yapılır. Eski test daha önce incelendiği için legacy regression olarak gösterilir; yeni holdout değildir.", "",
        f"Seçilen model: **{report['selection']['selected_model']}**. Eşitlik sırası: word, char, tinybert.", "",
        "| Model | Validation macro F1 | Legacy regression macro F1 | Eğitim (ms) | Validation batch (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        lines.append(
            f"| {name} | {result['validation']['macro_f1']:.3f} | "
            f"{result['legacy_regression']['macro_f1']:.3f} | {result['fit_ms']:.1f} | "
            f"{result['validation']['predict_ms']:.1f} |"
        )
    lines += ["", "Süreler tek yerel koşunun duvar saati ölçümüdür; indirme/yükleme eğitim süresine dahil değildir. Üretim gecikmesi garantisi değildir."]
    for name, result in report["models"].items():
        lines += ["", f"## {name}", "", f"Artifact: `{result['artifact_path']}`", ""]
        for split_name in ("validation", "legacy_regression"):
            evaluation = result[split_name]
            lines += [
                f"### {split_name}", "", "| Etiket | Precision | Recall | F1 | Örnek |",
                "|---|---:|---:|---:|---:|",
            ]
            for label in LABELS:
                scores = evaluation["per_class"][label]
                lines.append(
                    f"| {label} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
                    f"{scores['f1-score']:.3f} | {int(scores['support'])} |"
                )
            lines += ["", "Hata matrisi: satır gerçek, sütun tahmin.", "",
                      "| Gerçek / Tahmin | " + " | ".join(LABELS) + " |",
                      "|---|" + "---:|" * len(LABELS)]
            lines += [
                "| " + label + " | " + " | ".join(map(str, counts)) + " |"
                for label, counts in zip(LABELS, evaluation["confusion_matrix"])
            ]
            lines += ["", "Yanlış sınıflandırmalar:", ""]
            lines += [
                f"- `{error['id']}`: {error['expected']} → {error['predicted']}: {error['text']}"
                for error in evaluation["errors"]
            ] or ["Bu küçük bölmede yanlış sınıflandırma bulunmadı."]
    lines += ["", "## Sınırlar", ""] + ["- " + item for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def compare(
    data_path: Path = DEFAULT_DATA, output_dir: Path = PROJECT_ROOT, *,
    include_transformer: bool = False, epochs: int = 3,
    local_files_only: bool = False,
) -> dict:
    data_path, output_dir = Path(data_path), Path(output_dir)
    rows = load_records(data_path)
    subsets = split_for_experiment(rows)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, factory in (("word", make_pipeline), ("char", make_char_pipeline)):
        model = factory()
        started = perf_counter()
        model.fit([row["text"] for row in subsets["train"]], [row["label"] for row in subsets["train"]])
        fit_ms = (perf_counter() - started) * 1000
        metadata = _metadata(name, subsets["train"], data_sha)
        artifact_path = model_dir / f"{name}.joblib"
        _atomic_artifact(artifact_path, {"metadata": metadata, "pipeline": model})
        results[name] = {
            "artifact_path": f"models/{name}.joblib", "metadata": metadata, "fit_ms": fit_ms,
            "validation": evaluate_model(model, subsets["validation"]),
            "legacy_regression": evaluate_model(model, subsets["test"]),
        }
    if include_transformer:
        from ticketmind.transformer_model import train_transformer

        model, metadata = train_transformer(
            subsets["train"], model_dir / "tinybert", data_sha256=data_sha,
            epochs=epochs, local_files_only=local_files_only,
        )
        results["tinybert"] = {
            "artifact_path": "models/tinybert", "metadata": metadata,
            "fit_ms": metadata["timings_ms"]["fit"],
            "validation": evaluate_model(model, subsets["validation"]),
            "legacy_regression": evaluate_model(model, subsets["test"]),
        }
    selected = select_model(results)
    report = {
        "schema_version": 1,
        "dataset_type": "synthetic_demo",
        "data_sha256": data_sha,
        "training_policy": "original train only; development.jsonl is never loaded or merged",
        "split_summary": {
            split: {"n": len(items), "groups": len({row['group_id'] for row in items}),
                    "labels": dict(Counter(row["label"] for row in items))}
            for split, items in subsets.items()
        },
        "selection": {
            "metric": "validation.macro_f1", "selected_model": selected,
            "selected_artifact": results[selected]["artifact_path"],
            "tie_break_order": list(MODEL_ORDER),
            "hyperparameters_fixed_before_evaluation": True,
            "legacy_test_role": "inspected regression set, excluded from selection",
        },
        "models": results,
        "limitations": [
            "Artificial scenario data; real-world accuracy and robustness are unknown.",
            "Original validation is small and reused; selection can overfit it.",
            "Legacy test was inspected before these experiments; it is not independent holdout evidence.",
            "All outputs need human review; probabilities are not calibrated confidence.",
            "Transformer tokenization truncates at 96 tokens by default; this is recorded in metadata.",
        ],
    }
    report_path = output_dir / "reports" / "model_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(report_path.with_suffix(".md"), _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketMind word/char/TinyBERT karşılaştırması")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--include-transformer", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true", help="HF dosyaları yalnız cache'den yüklenir")
    args = parser.parse_args()
    report = compare(
        args.data, args.output_dir, include_transformer=args.include_transformer,
        epochs=args.epochs, local_files_only=args.local_files_only,
    )
    print(json.dumps({
        "selected_model": report["selection"]["selected_model"],
        "report": str(args.output_dir / "reports" / "model_comparison.json"),
        "models": list(report["models"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
