"""Kural çıkarımını tam alan+başlangıç+bitiş eşleşmesiyle değerlendir."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Callable

from ticketmind.data import PROJECT_ROOT
from ticketmind.entities import extract_entities
from ticketmind.workbench import ENTITY_FIELDS, validate_entity_spans

DEFAULT_GOLD = PROJECT_ROOT / "data" / "entities_gold.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "entities.json"


def _scores(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def load_gold(path: Path = DEFAULT_GOLD) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Alan değerlendirme korpusu boş olamaz.")
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Her alan değerlendirme kaydı benzersiz bir id içermelidir.")
        if row["id"] in seen_ids:
            raise ValueError("Tekrarlanan alan değerlendirme id değeri.")
        seen_ids.add(row["id"])
        if (not isinstance(row.get("text"), str) or not row["text"].strip()
                or row.get("source") != "ai_authored_synthetic"
                or row.get("annotation_status") != "ai_authored_silver"
                or row.get("split") not in {"evaluation", "regression"}):
            raise ValueError("Alan korpusunun metin/köken/bölme sözleşmesi geçersiz.")
        if row.get("entities") is None:
            raise ValueError("Altın alan listesi açıkça verilmelidir; boş liste geçerlidir.")
        validate_entity_spans(row["text"], row["entities"])
    return rows


def evaluate_entities(rows: list[dict], extractor: Callable[[str], list[dict]] = extract_entities) -> dict:
    counts = {field: Counter(tp=0, fp=0, fn=0) for field in sorted(ENTITY_FIELDS)}
    examples = []
    for row in rows:
        gold = validate_entity_spans(row["text"], row["entities"])
        raw_predictions = extractor(row["text"])
        predictions = [{key: entity[key] for key in ("field", "value", "start", "end")}
                       for entity in raw_predictions]
        validate_entity_spans(row["text"], predictions)
        expected = {(entity["field"], entity["start"], entity["end"]) for entity in gold}
        predicted = {(entity["field"], entity["start"], entity["end"]) for entity in predictions}
        tp, fp, fn = expected & predicted, predicted - expected, expected - predicted
        for field in counts:
            counts[field].update(tp=sum(item[0] == field for item in tp),
                                 fp=sum(item[0] == field for item in fp),
                                 fn=sum(item[0] == field for item in fn))
        examples.append({"id": row["id"], "expected": gold, "predicted": predictions,
                         "exact_match": expected == predicted,
                         "false_positives": [list(item) for item in sorted(fp)],
                         "false_negatives": [list(item) for item in sorted(fn)]})
    totals = Counter(tp=0, fp=0, fn=0)
    for count in counts.values():
        totals.update(count)
    return {
        "matching": "exact field + start + end; end-exclusive Python character offsets",
        "n": len(rows), "micro": _scores(**totals),
        "per_field": {field: _scores(**count) for field, count in counts.items()},
        "record_exact_match": sum(example["exact_match"] for example in examples) / len(rows) if rows else 0.0,
        "examples": examples,
    }


def write_report(gold_path: Path = DEFAULT_GOLD, report_path: Path = DEFAULT_REPORT) -> dict:
    rows = load_gold(gold_path)
    report = {
        "dataset": "AI-authored synthetic silver annotations; not independently human validated",
        "data_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "limitations": [
            "Rules and a small product dictionary, not a learned NER model.",
            "Silver labels were written for this project; results do not estimate customer performance.",
            "Unknown products and ambiguous words deliberately remain in this set; failures are retained.",
        ],
        **evaluate_entities(rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Alan çıkarımı: sentetik silver değerlendirmesi", "",
             "Bu etiketler AI tarafından elle seçilmiş aralıklardır; bağımsız insan doğrulaması yoktur.", "",
             f"Örnek sayısı: {report['n']}. Veri SHA-256: `{report['data_sha256']}`.", "",
             "Eşleşme: alan türü + başlangıç + bitiş konumunun tamamı aynı olmalıdır. Kısmi puan verilmez.", "",
             "| Alan | TP | FP | FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for field, score in {**report["per_field"], "micro": report["micro"]}.items():
        lines.append(f"| {field} | {score['tp']} | {score['fp']} | {score['fn']} | "
                     f"{score['precision']:.3f} | {score['recall']:.3f} | {score['f1']:.3f} |")
    lines += ["", "## Hata bulunan örnekler", ""]
    lookup = {row["id"]: row["text"] for row in rows}
    mistakes = [item for item in report["examples"] if not item["exact_match"]]
    for item in mistakes:
        lines.append(f"- `{item['id']}`: {lookup[item['id']]} "
                     f"Yanlış eklenen: `{item['false_positives']}`; kaçırılan: `{item['false_negatives']}`.")
    if not mistakes:
        lines.append("Bu küçük kümede tam eşleşme sağlandı; gerçek kullanım başarısını kanıtlamaz.")
    lines += ["", "## Sınırlar", ""] + ["- " + item for item in report["limitations"]]
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketMind alan çıkarımını değerlendir.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = write_report(args.gold, args.report)
    print(json.dumps({"report": str(args.report), "n": report["n"], "micro": report["micro"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
