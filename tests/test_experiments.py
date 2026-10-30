"""Model-selection invariants: training isolation and validation-only decisions."""

from copy import deepcopy
import json

import joblib
import numpy as np
import pytest

import ticketmind.experiments as experiments
from ticketmind.experiments import compare, select_model
from ticketmind.service import TicketService


def records():
    messages = {
        "kargo": "Kargom teslim edilmedi paketimi bekliyorum",
        "odeme": "Kartımdan iki kez ödeme ücreti çekildi",
        "iade": "Ürünü iade etmek geri göndermek istiyorum",
        "teknik": "Uygulama açılmıyor giriş hatası veriyor",
    }
    return [
        {
            "id": f"{label}-{split}",
            "text": text if split == "train" else f"{label} {split} benzersizbekletilensözcük",
            "label": label, "group_id": f"{label}-{split}", "split": split,
            "source": "ai_authored_synthetic",
        }
        for label, text in messages.items()
        for split in ("train", "validation", "test")
    ]


def write_data(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )


def test_selection_never_reads_legacy_regression_and_ties_are_fixed():
    results = {
        "char": {"validation": {"macro_f1": 0.4}, "legacy_regression": {"macro_f1": 1.0}},
        "word": {"validation": {"macro_f1": 0.8}, "legacy_regression": {"macro_f1": 0.0}},
    }
    assert select_model(results) == "word"
    results["char"]["validation"]["macro_f1"] = 0.8
    assert select_model(results) == "word"
    results["tinybert"] = {"validation": {"macro_f1": 0.9}}
    assert select_model(results) == "tinybert"


def test_heldout_changes_do_not_change_fitted_models_and_dev_is_ignored(tmp_path):
    first_rows = records()
    altered_rows = deepcopy(first_rows)
    for row in altered_rows:
        if row["split"] != "train":
            row["text"] = f"{row['id']} yepyenibekletilenveri değişmiş örnek"
    first_data, second_data = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_data(first_data, first_rows)
    write_data(second_data, altered_rows)
    first_output, second_output = tmp_path / "first", tmp_path / "second"
    (first_output / "data").mkdir(parents=True)
    (first_output / "data" / "development.jsonl").write_text("not valid JSON", encoding="utf-8")
    first_report = compare(first_data, first_output)
    compare(second_data, second_output)
    training_ids = {row["id"] for row in first_rows if row["split"] == "train"}
    for name in ("word", "char"):
        first = joblib.load(first_output / "models" / f"{name}.joblib")
        second = joblib.load(second_output / "models" / f"{name}.joblib")
        assert set(first["metadata"]["training_ids"]) == training_ids
        assert first["pipeline"].named_steps["tfidf"].vocabulary_ == second["pipeline"].named_steps["tfidf"].vocabulary_
        np.testing.assert_array_equal(
            first["pipeline"].named_steps["tfidf"].idf_,
            second["pipeline"].named_steps["tfidf"].idf_,
        )
        np.testing.assert_array_equal(
            first["pipeline"].named_steps["classifier"].coef_,
            second["pipeline"].named_steps["classifier"].coef_,
        )
        evaluation = first_report["models"][name]
        assert "legacy_regression" in evaluation and "test" not in evaluation
        assert evaluation["validation"]["n"] == 4
        assert evaluation["validation"]["predict_ms"] >= 0
        assert evaluation["fit_ms"] >= 0
        assert len(evaluation["validation"]["confusion_matrix"]) == 4


def test_report_and_saved_artifacts_are_real_service_inputs(tmp_path):
    data_path = tmp_path / "data.jsonl"
    write_data(data_path, records())
    report = compare(data_path, tmp_path)
    saved_report = json.loads((tmp_path / "reports" / "model_comparison.json").read_text(encoding="utf-8"))
    assert saved_report == report
    assert (tmp_path / "reports" / "model_comparison.md").is_file()
    assert report["selection"]["metric"] == "validation.macro_f1"
    for name in ("word", "char"):
        service = TicketService(tmp_path / report["models"][name]["artifact_path"])
        result = service.analyze("Ürünü iade etmek geri göndermek istiyorum")
        assert result["category"] == "iade"
        assert result["needs_review"] is True
        assert result["model_version"] == report["models"][name]["metadata"]["model_version"]


def test_failed_artifact_save_preserves_previous_file(monkeypatch, tmp_path):
    destination = tmp_path / "word.joblib"
    destination.write_bytes(b"previous complete artifact")

    def failed_dump(artifact, path):
        path.write_bytes(b"incomplete new artifact")
        raise OSError("injected serialization failure")

    monkeypatch.setattr(experiments.joblib, "dump", failed_dump)
    with pytest.raises(OSError):
        experiments._atomic_artifact(destination, {"not": "published"})
    assert destination.read_bytes() == b"previous complete artifact"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["word.joblib"]
