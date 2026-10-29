"""Sabit grup bölmelerini ve veri sözleşmesini eğitimden önce doğrular."""

import json
from pathlib import Path

from ticketmind.text import normalize_text

LABELS = ("kargo", "odeme", "iade", "teknik")
SPLITS = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "tickets.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "baseline.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "baseline.json"


def load_records(path: Path = DEFAULT_DATA) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    validate_records(records)
    return records


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Veri seti boş olamaz.")
    ids, texts, groups = set(), {}, {}
    coverage = {split: set() for split in SPLITS}
    for row in records:
        for field in ("id", "text", "label", "group_id", "split", "source"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Eksik veya boş alan: {field}")
        if row["label"] not in LABELS or row["split"] not in SPLITS:
            raise ValueError("Bilinmeyen etiket veya veri bölmesi.")
        if row["source"] != "ai_authored_synthetic":
            raise ValueError("Bu başlangıç hattı yalnız belgelenmiş sentetik demo verisini kabul eder.")
        if len(row["text"]) > 4000:
            raise ValueError("Mesaj 4000 karakteri aşamaz.")
        if row["id"] in ids:
            raise ValueError(f"Tekrarlanan id: {row['id']}")
        ids.add(row["id"])
        normalized = normalize_text(row["text"])
        if normalized in texts:
            raise ValueError(f"Tekrarlanan normalize metin: {row['id']}")
        texts[normalized] = row["id"]
        group_signature = (row["split"], row["label"])
        if row["group_id"] in groups and groups[row["group_id"]] != group_signature:
            raise ValueError("Aynı senaryo grubu farklı bölme veya etikette bulunamaz.")
        groups[row["group_id"]] = group_signature
        coverage[row["split"]].add(row["label"])
    if any(labels != set(LABELS) for labels in coverage.values()):
        raise ValueError("Her bölme dört etiketi de içermeli.")

