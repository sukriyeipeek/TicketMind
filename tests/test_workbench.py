import json

import pytest

from ticketmind.data import PROJECT_ROOT
from ticketmind.text import normalize_text
from ticketmind.workbench import WorkbenchStore, validate_entity_spans


def store(tmp_path):
    return WorkbenchStore(tmp_path / "reviews.sqlite3", protected_paths=[])


def test_record_roundtrip_keeps_original_offsets_and_provenance(tmp_path):
    workbench = store(tmp_path)
    text = "  TM-7123 siparişini iptal edin."
    entity = {"field": "order_id", "value": "TM-7123", "start": 2, "end": 9}
    result = workbench.add_record(text=text, label="iade", group_id="user-cancel-1",
                                  note="Kullanıcının açık iptal isteği.", entities=[entity])
    assert result["text"] == text
    assert result["entities"] == [entity]
    assert result["source"] == "human_authored"
    assert result["split"] == "development"
    assert result["created_at"].endswith("+00:00")
    assert result == workbench.list_records()[0]
    exported = json.loads(workbench.export_jsonl())
    assert exported == result
    assert "normalized_text" not in exported
    assert workbench.stats()["entity_annotated"] == 1
    # Yeniden açıldığında veriler SQLite üzerinde kalır.
    assert WorkbenchStore(workbench.db_path, protected_paths=[]).stats()["total"] == 1


def test_duplicate_is_rejected_after_turkish_normalization(tmp_path):
    workbench = store(tmp_path)
    workbench.add_record(text="İADE  istiyorum", label="iade", group_id="user-1")
    with pytest.raises(ValueError, match="normalize"):
        workbench.add_record(text="iade istiyorum ", label="iade", group_id="user-2")
    assert workbench.stats()["total"] == 1


def test_protected_corpus_text_and_group_cannot_enter_development(tmp_path):
    corpus = tmp_path / "holdout.jsonl"
    corpus.write_text(json.dumps({"text": "Bu bağımsız test kaydı.", "group_id": "reserved-group"}), encoding="utf-8")
    workbench = WorkbenchStore(tmp_path / "reviews.sqlite3", protected_paths=[corpus])
    with pytest.raises(ValueError, match="korunan bir korpusta"):
        workbench.add_record(text="BU BAĞIMSIZ TEST KAYDI.", label="review", group_id="user-1")
    with pytest.raises(ValueError, match="grup korunan"):
        workbench.add_record(text="Farklı bir yeniden yazım.", label="review", group_id="reserved-group")
    assert workbench.stats()["total"] == 0


def test_group_label_conflict_rejected_without_dropping_previous_record(tmp_path):
    workbench = store(tmp_path)
    workbench.add_record(text="Kuryeye ulaşamıyorum.", label="kargo", group_id="same-case")
    with pytest.raises(ValueError, match="çelişen"):
        workbench.add_record(text="Ücrette sorun var.", label="odeme", group_id="same-case")
    assert workbench.stats()["by_label"]["kargo"] == 1
    assert workbench.stats()["total"] == 1


def test_review_export_and_pagination_do_not_change_training_data(tmp_path):
    before = (PROJECT_ROOT / "data" / "tickets.jsonl").read_bytes()
    workbench = store(tmp_path)
    workbench.add_record(text="Neyin bozulduğunu bilmiyorum.", label="review", group_id="review-1")
    workbench.add_record(text="Ürün çalışmıyor ama henüz ne isteyeceğime karar vermedim.",
                         label="review", group_id="review-2", entities=[])
    assert len(workbench.list_records(limit=1, offset=1)) == 1
    assert len(workbench.export_jsonl().splitlines()) == 2
    assert workbench.stats()["entity_annotated"] == 1  # None ile [] farklıdır.
    assert all(json.loads(line)["split"] == "development" for line in workbench.export_jsonl().splitlines())
    assert before == (PROJECT_ROOT / "data" / "tickets.jsonl").read_bytes()


@pytest.mark.parametrize("kwargs", [
    {"text": " "}, {"text": "a" * 4001}, {"label": "spam"},
    {"group_id": ""}, {"note": "a" * 2001},
])
def test_invalid_user_fields_rejected(tmp_path, kwargs):
    data = {"text": "Kısa bir yeni mesaj.", "label": "review", "group_id": "user-1"}
    data.update(kwargs)
    with pytest.raises(ValueError):
        store(tmp_path).add_record(**data)


@pytest.mark.parametrize("entity", [
    {"field": "product", "value": "telefon", "start": -1, "end": 6},
    {"field": "product", "value": "telefon", "start": 0, "end": 99},
    {"field": "product", "value": "başka", "start": 0, "end": 7},
    {"field": "secret", "value": "telefon", "start": 0, "end": 7},
    {"field": "product", "value": "telefon", "start": False, "end": 7},
])
def test_span_bounds_and_values_are_checked(entity):
    with pytest.raises(ValueError):
        validate_entity_spans("telefon", [entity])


def test_database_environment_override(tmp_path, monkeypatch):
    path = tmp_path / "user-workbench.sqlite3"
    monkeypatch.setenv("TICKETMIND_WORKBENCH_DB", str(path))
    assert WorkbenchStore(protected_paths=[]).db_path == path


def test_development_corpus_is_separate_and_balanced():
    rows = [json.loads(line) for line in (PROJECT_ROOT / "data" / "development.jsonl").read_text(encoding="utf-8").splitlines()]
    starter = [json.loads(line) for line in (PROJECT_ROOT / "data" / "tickets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 32
    assert {row["label"]: sum(item["label"] == row["label"] for item in rows) for row in rows} == {
        "kargo": 8, "odeme": 8, "iade": 8, "teknik": 8,
    }
    assert all(row["split"] == "development" and row["source"] == "ai_authored_synthetic" for row in rows)
    assert not {normalize_text(row["text"]) for row in starter} & {normalize_text(row["text"]) for row in rows}
    assert not {row["group_id"] for row in starter} & {row["group_id"] for row in rows}
