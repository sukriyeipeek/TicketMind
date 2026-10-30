"""Offline checks; a tiny randomly initialized local BERT is never a quality test."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ticketmind.data import LABELS
from ticketmind.transformer_model import TransformerClassifier, train_transformer


def test_optional_model_module_import_does_not_load_torch_or_transformers():
    source_dir = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source_dir)!r}); "
        "import ticketmind.transformer_model; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_non_training_rows_are_rejected_before_optional_model_loading(tmp_path):
    with pytest.raises(ValueError, match="train"):
        train_transformer(
            [{"id": "validation-leak", "text": "iade", "label": "iade", "split": "validation"}],
            tmp_path, data_sha256="not-used",
        )
    assert not (tmp_path / "metadata.json").exists()


def test_invalid_local_metadata_is_rejected_without_optional_libraries(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"artifact_schema": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        TransformerClassifier(tmp_path)


def test_changed_local_files_are_rejected_before_loading_weights(tmp_path):
    (tmp_path / "config.json").write_text("changed config", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "artifact_schema": 1, "model_kind": "transformer", "dataset_type": "synthetic_demo",
        "review_policy": "always_review_demo", "labels": list(LABELS),
        "model_version": "hash-check-only", "hyperparameters": {"max_length": 96},
        "files_sha256": {"config.json": "0" * 64},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        TransformerClassifier(tmp_path)


def test_local_tiny_bert_roundtrip_without_download(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(42)
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=12, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=128,
            num_labels=4, id2label=dict(enumerate(LABELS)),
            label2id={label: index for index, label in enumerate(LABELS)},
        )
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\niade\nkargo\nödeme\nhata\nürün\ngiriş\npaket\n",
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "do_lower_case": False}), encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "artifact_schema": 1, "model_kind": "transformer", "dataset_type": "synthetic_demo",
        "review_policy": "always_review_demo", "labels": list(LABELS),
        "model_version": "local-random-test-only", "hyperparameters": {"max_length": 96},
    }), encoding="utf-8")
    adapter = TransformerClassifier(tmp_path)
    probabilities = adapter.predict_proba(["İADE", "Kargo paket"])
    assert probabilities.shape == (2, 4)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
    np.testing.assert_allclose(
        adapter.predict_proba(["İADE"]), probabilities[:1], rtol=1e-5, atol=1e-7,
    )
    assert adapter.predict_proba([]).shape == (0, 4)
    assert adapter.classes_.tolist() == list(LABELS)
    with pytest.raises(ValueError):
        adapter.predict_proba([""])
