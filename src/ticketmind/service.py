"""Eğitilmiş yerel modelle kullanım; bu modül eğitim başlatmaz."""

from pathlib import Path

import joblib
import numpy as np
import sklearn

from ticketmind.data import DEFAULT_MODEL, LABELS
from ticketmind.entities import extract_entities


class TicketService:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        model_path = Path(model_path)
        self.is_transformer = model_path.is_dir()
        if self.is_transformer:
            from ticketmind.transformer_model import TransformerClassifier
            self.pipeline = TransformerClassifier(model_path)
            self.metadata = self.pipeline.metadata
            return
        # joblib/pickle yalnız kendi ürettiğimiz güvenilir yerel artifact için kullanılır.
        artifact = joblib.load(model_path)
        self.metadata = artifact["metadata"]
        self.pipeline = artifact["pipeline"]
        if (self.metadata.get("artifact_schema") != 1
                or self.metadata.get("dataset_type") != "synthetic_demo"
                or self.metadata.get("review_policy") != "always_review_demo"
                or self.metadata.get("sklearn_version") != sklearn.__version__
                or set(self.pipeline.classes_) != set(LABELS)):
            raise ValueError("Model sözleşmesi/sürümü uyumsuz; mevcut ortamda yeniden eğitin.")

    def analyze(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Mesaj boş olamaz ve en fazla 4000 karakter içerebilir.")
        input_truncated = False
        if self.is_transformer:
            from ticketmind.text import normalize_text
            token_count = len(self.pipeline.tokenizer(normalize_text(text), truncation=False)["input_ids"])
            input_truncated = token_count > self.pipeline.max_length
            scores = self.pipeline.predict_proba([text])[0]
            no_features = False
        else:
            vector = self.pipeline.named_steps["tfidf"].transform([text])
            scores = self.pipeline.named_steps["classifier"].predict_proba(vector)[0]
            no_features = vector.nnz == 0
        if not np.isfinite(scores).all() or np.any(scores < 0) or not np.isclose(scores.sum(), 1, atol=1e-5):
            raise ValueError("Model geçerli kategori skorları üretmedi.")
        best = int(scores.argmax())
        # Sıfır özellik dışında kapsam dışını güvenilir biçimde saptadığımız iddia edilmez.
        reasons = ["synthetic_demo_model"]
        if no_features:
            reasons.append("no_known_features")
        if input_truncated:
            reasons.append("transformer_input_truncated")
        return {
            "category": None if no_features else str(self.pipeline.classes_[best]),
            "model_score": float(scores[best]),
            "category_scores": {str(label): float(score) for label, score in zip(self.pipeline.classes_, scores)},
            "score_type": "uncalibrated_model_score",
            "needs_review": True,
            "review_reasons": reasons,
            "entities": extract_entities(text),
            "model_version": self.metadata["model_version"],
            "input_truncated": input_truncated,
        }

