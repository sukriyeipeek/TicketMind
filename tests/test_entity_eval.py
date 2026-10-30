import json

import pytest

from ticketmind.entity_eval import DEFAULT_GOLD, evaluate_entities, load_gold, write_report


def test_exact_span_metric_counts_boundary_errors_as_fp_and_fn():
    text = "TM-123 kulaklık E42"
    gold = [{"field": "order_id", "value": "TM-123", "start": 0, "end": 6},
            {"field": "product", "value": "kulaklık", "start": 7, "end": 15},
            {"field": "error_code", "value": "E42", "start": 16, "end": 19}]

    def fake_extractor(_):
        return [{**gold[0], "method": "rule"},
                {"field": "product", "value": "ulaklık", "start": 8, "end": 15, "method": "demo_dictionary"}]

    result = evaluate_entities([{"id": "boundary", "text": text, "entities": gold}], fake_extractor)
    assert result["micro"]["tp"] == 1
    assert result["micro"]["fp"] == 1
    assert result["micro"]["fn"] == 2
    assert result["micro"]["precision"] == pytest.approx(0.5)
    assert result["micro"]["recall"] == pytest.approx(1 / 3)
    assert result["micro"]["f1"] == pytest.approx(0.4)
    assert result["record_exact_match"] == 0.0
    assert result["per_field"]["order_id"]["f1"] == 1.0


def test_wrong_field_does_not_match_same_span():
    entity = {"field": "order_id", "value": "TM-123", "start": 0, "end": 6}
    result = evaluate_entities([{"id": "field", "text": "TM-123", "entities": [entity]}],
                               lambda _: [{**entity, "field": "error_code"}])
    assert result["micro"]["tp"] == 0
    assert result["micro"]["fp"] == 1
    assert result["micro"]["fn"] == 1


def test_empty_entities_can_be_correct_but_no_positive_f1_claim():
    result = evaluate_entities([{"id": "empty", "text": "Teşekkürler.", "entities": []}], lambda _: [])
    assert result["record_exact_match"] == 1.0
    assert result["micro"]["f1"] == 0.0


def test_silver_corpus_spans_and_honest_rule_failures():
    rows = load_gold()
    assert len(rows) == 24
    assert all(row["source"] == "ai_authored_synthetic" for row in rows)
    assert all(row["annotation_status"] == "ai_authored_silver" for row in rows)
    result = evaluate_entities(rows)
    # Mevcut küçük sözlüğün sınırları gizlenmez; gerçek skor raporda ölçülür.
    assert result["per_field"]["product"]["fn"] > 0
    assert result["per_field"]["product"]["fp"] > 0


def test_report_json_and_markdown_are_generated(tmp_path):
    path = tmp_path / "entities.json"
    report = write_report(DEFAULT_GOLD, path)
    assert json.loads(path.read_text(encoding="utf-8"))["micro"] == report["micro"]
    assert "bağımsız insan doğrulaması yoktur" in path.with_suffix(".md").read_text(encoding="utf-8")


def test_unvalidated_or_invalid_gold_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"
    row = {"id": "bad", "text": "telefon", "source": "ai_authored_synthetic", "annotation_status": "ai_authored_silver",
           "split": "evaluation", "entities": [{"field": "product", "value": "telefon", "start": 0, "end": 999}]}
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="sınırları"):
        load_gold(path)
