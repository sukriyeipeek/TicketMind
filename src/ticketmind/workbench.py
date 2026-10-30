"""Yerel kullanıcı örnekleri; sabit eğitim/test korpusunu değiştirmez."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterable
from uuid import uuid4

from ticketmind.data import LABELS, PROJECT_ROOT
from ticketmind.text import normalize_text

ENTITY_FIELDS = {"order_id", "error_code", "product"}
WORKBENCH_LABELS = {*LABELS, "review"}


def validate_entity_spans(text: str, entities: list[dict] | None) -> list[dict] | None:
    """Seçilen alanları orijinal metne bağla; None=henüz etiketlenmedi."""
    if entities is None:
        return None
    if not isinstance(entities, list) or len(entities) > 100:
        raise ValueError("Alan etiketleri en fazla 100 elemanlı bir liste olmalıdır.")
    result, seen = [], set()
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {"field", "value", "start", "end"}:
            raise ValueError("Her alan field, value, start ve end içermelidir.")
        field, value = entity["field"], entity["value"]
        start, end = entity["start"], entity["end"]
        if not isinstance(field, str) or field not in ENTITY_FIELDS:
            raise ValueError("Bilinmeyen alan etiketi.")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("Alan konumları orijinal metnin sınırları içinde olmalıdır.")
        if not isinstance(value, str) or text[start:end] != value:
            raise ValueError("Alan değeri, orijinal metindeki seçili aralıkla eşleşmelidir.")
        signature = (field, start, end)
        if signature in seen:
            raise ValueError("Aynı alan aralığı iki kez etiketlenemez.")
        seen.add(signature)
        result.append({"field": field, "value": value, "start": start, "end": end})
    return sorted(result, key=lambda item: (item["start"], item["end"], item["field"]))


class WorkbenchStore:
    """SQLite bağlantıları çağrıya özeldir; metin yalnız yerel veritabanına yazılır."""

    def __init__(self, db_path: Path | None = None, protected_paths: Iterable[Path] | None = None):
        self.db_path = Path(db_path if db_path is not None else os.environ.get(
            "TICKETMIND_WORKBENCH_DB", str(PROJECT_ROOT / "data" / "local" / "reviews.sqlite3")
        ))
        # None dinamik varsayılandır: sonradan eklenen holdout dosyaları da korunur.
        self.protected_paths = None if protected_paths is None else tuple(Path(p) for p in protected_paths)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY, text TEXT NOT NULL,
                normalized_text TEXT NOT NULL UNIQUE, label TEXT NOT NULL,
                group_id TEXT NOT NULL, note TEXT NOT NULL,
                entities_json TEXT NOT NULL, created_at TEXT NOT NULL
            )""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _protected_values(self) -> tuple[set[str], set[str]]:
        paths = self.protected_paths
        if paths is None:
            paths = tuple((PROJECT_ROOT / "data").rglob("*.jsonl"))
        texts, groups = set(), set()
        for path in paths:
            if not path.is_file():
                continue
            # Geçersiz koruma korpusu sessizce atlanmaz: olası sızıntıyı gizleme.
            with path.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Korunan veri korpusu geçersiz.")
                    if isinstance(row.get("text"), str):
                        texts.add(normalize_text(row["text"]))
                    if isinstance(row.get("group_id"), str):
                        groups.add(row["group_id"])
        return texts, groups

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "text": row["text"], "label": row["label"],
            "group_id": row["group_id"], "note": row["note"],
            "entities": json.loads(row["entities_json"]),
            "created_at": row["created_at"], "source": "human_authored", "split": "development",
        }

    def add_record(self, *, text: str, label: str, group_id: str,
                   note: str = "", entities: list[dict] | None = None) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Mesaj 1–4000 karakter içermeli ve yalnız boşluk olmamalıdır.")
        if not isinstance(label, str) or label not in WORKBENCH_LABELS:
            raise ValueError("Etiket kargo, odeme, iade, teknik veya review olmalıdır.")
        if not isinstance(group_id, str) or not group_id.strip() or len(group_id) > 100:
            raise ValueError("Grup kimliği 1–100 karakter olmalıdır.")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Not en fazla 2000 karakter olmalıdır.")
        group_id = group_id.strip()
        checked_entities = validate_entity_spans(text, entities)
        normalized = normalize_text(text)
        protected_texts, protected_groups = self._protected_values()
        if normalized in protected_texts:
            raise ValueError("Bu mesaj korunan bir korpusta zaten var; geliştirme kaydı eklenmedi.")
        if group_id in protected_groups:
            raise ValueError("Bu grup korunan bir korpusa ait; yeni bir geliştirme grubu kullanın.")
        record_id = "human-" + uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT label FROM records WHERE group_id = ? LIMIT 1", (group_id,)
                ).fetchone()
                if existing is not None and existing["label"] != label:
                    raise ValueError("Aynı geliştirme grubunda çelişen etiketler bulunamaz.")
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (record_id, text, normalized, label, group_id, note,
                     json.dumps(checked_entities, ensure_ascii=False), created_at),
                )
                saved = connection.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
                return self._record(saved)
        except sqlite3.IntegrityError as error:
            raise ValueError("Bu mesajın normalize edilmiş bir kopyası zaten kayıtlı.") from error

    def list_records(self, limit: int = 100, offset: int = 0) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
            raise ValueError("limit 1–1000 aralığında, offset sıfır veya pozitif tam sayı olmalıdır.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM records ORDER BY created_at, id LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._record(row) for row in rows]

    def stats(self) -> dict:
        with self._connect() as connection:
            counts = connection.execute("SELECT label, COUNT(*) AS n FROM records GROUP BY label").fetchall()
            annotated = connection.execute(
                "SELECT COUNT(*) FROM records WHERE entities_json != 'null'"
            ).fetchone()[0]
            groups = connection.execute("SELECT COUNT(DISTINCT group_id) FROM records").fetchone()[0]
        by_label = {label: 0 for label in sorted(WORKBENCH_LABELS)}
        by_label.update({row["label"]: row["n"] for row in counts})
        return {"total": sum(by_label.values()), "groups": groups, "by_label": by_label,
                "entity_annotated": annotated, "source": "human_authored", "split": "development"}

    def export_jsonl(self) -> str:
        # Sayfalama sınırı dışa aktarımda kayıtları sessizce düşürmez.
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM records ORDER BY created_at, id").fetchall()
        return "".join(json.dumps(self._record(row), ensure_ascii=False) + "\n" for row in rows)
