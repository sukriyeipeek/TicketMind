"""Learning-pipeline invariants, independent of the authored demo corpus."""

from copy import deepcopy
import json
import math
import unicodedata

import joblib
import numpy as np
import pytest

from ticketmind.data import validate_records
from ticketmind.entities import extract_entities
from ticketmind.service import TicketService
from ticketmind.train import train


@pytest.fixture
def demo_rows():
    train_messages = {
        "kargo": "Kargom gelmedi ve paket teslim edilmedi.",
        "odeme": "Ödeme sırasında kartımdan iki kez ücret çekildi.",
        "iade": "Aldığım ürünü iade etmek ve geri göndermek istiyorum.",
        "teknik": "Uygulamaya giriş yapamıyorum ve ekran hata veriyor.",
    }
    rows = []
    for label, text in train_messages.items():
        for split in ("train", "validation", "test"):
            message = text
            if split == "validation":
                message = f"{label} dogrulamagizlisözcük farklı doğrulama mesajı."
            elif split == "test":
                message = f"{label} testgizlisözcük farklı test mesajı."
            rows.append(
                {
                    "id": f"{label}-{split}-1",
                    "text": message,
                    "label": label,
                    "group_id": f"{label}-{split}",
                    "split": split,
                    "source": "ai_authored_synthetic",
                }
            )
    return rows


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_holdout_text_cannot_change_vocabulary_or_fitted_weights(demo_rows, tmp_path):
    first_data = tmp_path / "first.jsonl"
    second_data = tmp_path / "second.jsonl"
    write_rows(first_data, demo_rows)
    perturbed = deepcopy(demo_rows)
    for row in perturbed:
        if row["split"] != "train":
            row["text"] = f"{row['id']} başkaevrendesözcük yepyeni tutulmuş veri."
    write_rows(second_data, perturbed)
    first_model, second_model = tmp_path / "first.joblib", tmp_path / "second.joblib"
    train(first_data, first_model, tmp_path / "first-report.json")
    train(second_data, second_model, tmp_path / "second-report.json")
    first = joblib.load(first_model)["pipeline"]
    second = joblib.load(second_model)["pipeline"]
    vocabulary = first.named_steps["tfidf"].vocabulary_
    assert "dogrulamagizlisözcük" not in vocabulary
    assert "testgizlisözcük" not in vocabulary
    assert "başkaevrendesözcük" not in vocabulary
    assert vocabulary == second.named_steps["tfidf"].vocabulary_
    np.testing.assert_allclose(
        first.named_steps["tfidf"].idf_, second.named_steps["tfidf"].idf_,
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        first.named_steps["classifier"].coef_,
        second.named_steps["classifier"].coef_,
        rtol=0, atol=0,
    )


def test_scenario_group_cannot_cross_train_and_validation(demo_rows):
    conflicting = deepcopy(demo_rows)
    train_row = next(row for row in conflicting if row["split"] == "train")
    validation_row = next(
        row for row in conflicting
        if row["split"] == "validation" and row["label"] == train_row["label"]
    )
    validation_row["group_id"] = train_row["group_id"]
    with pytest.raises(ValueError):
        validate_records(conflicting)


def test_equivalent_unicode_text_is_rejected_as_duplicate(demo_rows):
    duplicate = deepcopy(demo_rows)
    original = next(
        row for row in duplicate if row["label"] == "iade" and row["split"] == "train"
    )
    target = next(
        row for row in duplicate
        if row["label"] == "iade" and row["split"] == "validation"
    )
    target["text"] = (
        "  " + unicodedata.normalize("NFD", original["text"]).replace(" ", "   ") + "  "
    )
    with pytest.raises(ValueError):
        validate_records(duplicate)


def test_entities_point_to_original_unicode_text():
    text = "🧾 İade için TM-1042: kulaklığım E42 hatası veriyor."
    entities = extract_entities(text)
    assert {entity["field"]: entity["value"] for entity in entities} == {
        "order_id": "TM-1042",
        "product": "kulaklığım",
        "error_code": "E42",
    }
    for entity in entities:
        assert text[entity["start"] : entity["end"]] == entity["value"]
    assert [entity["start"] for entity in entities] == sorted(
        entity["start"] for entity in entities
    )


def test_trained_artifact_roundtrip_and_unknown_text_require_review(demo_rows, tmp_path):
    data_path = tmp_path / "data.jsonl"
    model_path = tmp_path / "baseline.joblib"
    report_path = tmp_path / "report.json"
    write_rows(data_path, demo_rows)
    report = train(data_path, model_path, report_path)
    service = TicketService(model_path)
    result = service.analyze("Aldığım ürünü iade etmek ve geri göndermek istiyorum.")
    assert result["category"] == "iade"
    assert result["model_version"] == report["model_version"]
    assert result["needs_review"] is True
    assert result["score_type"] == "uncalibrated_model_score"
    assert "synthetic_demo_model" in result["review_reasons"]
    assert math.isfinite(result["model_score"])
    assert 0 <= result["model_score"] <= 1
    unknown = service.analyze("zzqxvvyyy 🚀")
    assert unknown["category"] is None
    assert unknown["needs_review"] is True
    assert "no_known_features" in unknown["review_reasons"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["split_summary"]["train"]["n"] == 4


def test_artifact_cannot_enable_automatic_decisions(demo_rows, tmp_path):
    data_path = tmp_path / "data.jsonl"
    model_path = tmp_path / "baseline.joblib"
    write_rows(data_path, demo_rows)
    train(data_path, model_path, tmp_path / "report.json")
    artifact = joblib.load(model_path)
    artifact["metadata"]["review_policy"] = "automatic"
    joblib.dump(artifact, model_path)
    with pytest.raises(ValueError):
        TicketService(model_path)
