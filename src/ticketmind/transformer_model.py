"""Optional, real CPU fine-tuning and local inference for a pinned Turkish BERT.

Importing this module never imports torch/transformers and never downloads files.
Only explicit ``train_transformer`` may download the immutable source checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from time import perf_counter

import numpy as np

from ticketmind import __version__
from ticketmind.data import LABELS
from ticketmind.text import normalize_text


SOURCE_MODEL = "ytu-ce-cosmos/turkish-tiny-bert-uncased"
SOURCE_REVISION = "208794046047fc7445f7a4179636423802691268"
DEFAULT_MAX_LENGTH = 96


def _libraries():
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer seçeneği için torch, transformers ve safetensors kurulmalı. "
            "README'deki isteğe bağlı transformer kurulumunu uygulayın."
        ) from exc
    return torch, transformers, AutoTokenizer, AutoModelForSequenceClassification


def training_fingerprint(rows: list[dict]) -> str:
    canonical = json.dumps(
        [{"id": row["id"], "text": row["text"], "label": row["label"]} for row in rows],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_training_rows(rows: list[dict]) -> None:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Transformer eğitimi yalnız train satırlarını kabul eder.")
    if {row.get("label") for row in rows} != set(LABELS):
        raise ValueError("Transformer eğitiminde dört etiket de bulunmalı.")


class TransformerClassifier:
    """A sklearn-like adapter that loads only locally saved safetensors."""

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if (
            self.metadata.get("artifact_schema") != 1
            or self.metadata.get("model_kind") != "transformer"
            or self.metadata.get("dataset_type") != "synthetic_demo"
            or self.metadata.get("review_policy") != "always_review_demo"
            or self.metadata.get("labels") != list(LABELS)
        ):
            raise ValueError("Transformer metadata sözleşmesi uyumsuz.")
        self.model_version = self.metadata["model_version"]
        self.max_length = int(self.metadata["hyperparameters"]["max_length"])
        if not 1 <= self.max_length <= 512:
            raise ValueError("Geçersiz transformer token bütçesi.")
        for name, expected_sha in self.metadata.get("files_sha256", {}).items():
            if Path(name).name != name:
                raise ValueError("Geçersiz yerel artifact dosya adı.")
            if hashlib.sha256((model_dir / name).read_bytes()).hexdigest() != expected_sha:
                raise ValueError("Transformer dosya hash'i metadata ile eşleşmiyor.")
        self.classes_ = np.asarray(LABELS)
        torch, _, AutoTokenizer, AutoModel = _libraries()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), do_lower_case=False, local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True,
        ).to("cpu")
        if (
            self.model.config.num_labels != len(LABELS)
            or self.model.config.id2label != dict(enumerate(LABELS))
        ):
            raise ValueError("Transformer etiket sırası metadata ile eşleşmiyor.")
        self.model.eval()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() or len(text) > 4000 for text in texts
        ):
            raise ValueError("Metinler boş olmayan, en fazla 4000 karakterlik string listesi olmalı.")
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        probabilities = []
        for start in range(0, len(texts), 8):
            encoded = self.tokenizer(
                [normalize_text(text) for text in texts[start : start + 8]],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities.append(self._torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, texts: list[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)] if len(texts) else self.classes_[:0]


def train_transformer(
    rows: list[dict], model_dir: Path, *, data_sha256: str,
    epochs: int = 3, batch_size: int = 8, max_length: int = DEFAULT_MAX_LENGTH,
    learning_rate: float = 5e-4, seed: int = 42, threads: int = 2,
    local_files_only: bool = False,
) -> tuple[TransformerClassifier, dict]:
    """Fine-tune the fixed checkpoint on TRAIN only; no validation/test gradients."""
    _require_training_rows(rows)
    if not 1 <= epochs <= 20 or not 1 <= batch_size <= 64 or not 4 <= max_length <= 512:
        raise ValueError("epochs 1–20, batch_size 1–64 ve max_length 4–512 olmalı.")
    if not 0 < learning_rate < 1 or not 1 <= threads <= 16:
        raise ValueError("Geçersiz öğrenme oranı veya CPU thread sayısı.")
    model_dir = Path(model_dir)
    cache_dir = model_dir.parent / "hf-cache"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    torch, transformers, AutoTokenizer, AutoModel = _libraries()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    load_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL, revision=SOURCE_REVISION, do_lower_case=False,
        trust_remote_code=False, local_files_only=local_files_only, cache_dir=str(cache_dir),
    )
    model = AutoModel.from_pretrained(
        SOURCE_MODEL, revision=SOURCE_REVISION, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        trust_remote_code=False, use_safetensors=True,
        local_files_only=local_files_only, cache_dir=str(cache_dir),
    ).to("cpu")
    download_load_ms = (perf_counter() - load_started) * 1000
    prepare_started = perf_counter()
    texts = [normalize_text(row["text"]) for row in rows]
    token_lengths = [len(item) for item in tokenizer(texts, truncation=False)["input_ids"]]
    encoded = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    keys = list(encoded)
    labels = torch.tensor([LABELS.index(row["label"]) for row in rows], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(*(encoded[key] for key in keys), labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    preprocessing_ms = (perf_counter() - prepare_started) * 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    fit_started = perf_counter()
    losses = []
    for _ in range(epochs):
        model.train()
        total_loss, total_examples = 0.0, 0
        for batch in loader:
            inputs = {key: value for key, value in zip(keys, batch[:-1])}
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=batch[-1])
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(output.loss.detach()) * len(batch[-1])
            total_examples += len(batch[-1])
        losses.append(total_loss / total_examples)
    fit_ms = (perf_counter() - fit_started) * 1000
    train_sha = training_fingerprint(rows)
    hyperparameters = {
        "epochs": epochs, "batch_size": batch_size, "max_length": max_length,
        "learning_rate": learning_rate, "optimizer": "AdamW", "weight_decay": 0.01,
        "gradient_clip_norm": 1.0, "seed": seed, "threads": threads, "device": "cpu",
        "truncation": True,
    }
    identity = hashlib.sha256(json.dumps({
        "source_revision": SOURCE_REVISION, "train_sha256": train_sha,
        "hyperparameters": hyperparameters,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    metadata = {
        "artifact_schema": 1,
        "model_kind": "transformer",
        "model_version": f"tinybert-{__version__}-{identity[:12]}",
        "dataset_type": "synthetic_demo",
        "review_policy": "always_review_demo",
        "labels": list(LABELS),
        "data_sha256": data_sha256,
        "train_sha256": train_sha,
        "training_ids": [row["id"] for row in rows],
        "training_groups": sorted({row["group_id"] for row in rows}),
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_weights": "model.safetensors",
        "trust_remote_code": False,
        "normalization": "ticketmind.text.normalize_text; tokenizer do_lower_case=False",
        "python_version": platform.python_version(),
        "libraries": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hyperparameters": hyperparameters,
        "train_truncated_examples": sum(length > max_length for length in token_lengths),
        "epoch_train_loss": losses,
        "timings_ms": {"download_load": download_load_ms, "preprocessing": preprocessing_ms, "fit": fit_ms},
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    save_started = perf_counter()
    model.eval()
    with tempfile.TemporaryDirectory(prefix=".tinybert-", dir=model_dir.parent) as temporary_dir:
        staging = Path(temporary_dir)
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        metadata["files_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in staging.iterdir() if path.is_file()
        }
        metadata["timings_ms"]["save"] = (perf_counter() - save_started) * 1000
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        # Publish metadata last. A concurrent new loader sees absent metadata
        # rather than a mixture of old config and new weights during replacement.
        (model_dir / "metadata.json").unlink(missing_ok=True)
        for path in staging.iterdir():
            if path.is_file() and path.name != "metadata.json":
                os.replace(path, model_dir / path.name)
        os.replace(staging / "metadata.json", model_dir / "metadata.json")
    return TransformerClassifier(model_dir), metadata
