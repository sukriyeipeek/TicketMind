"""Eğitilmiş yerel modelle kullanım; bu modül eğitim başlatmaz."""

from pathlib import Path

import joblib
import sklearn

from ticketmind.data import DEFAULT_MODEL, LABELS
from ticketmind.entities import extract_entities


class TicketService:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
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
        vector = self.pipeline.named_steps["tfidf"].transform([text])
        scores = self.pipeline.named_steps["classifier"].predict_proba(vector)[0]
        best = int(scores.argmax())
        # Sıfır özellik dışında kapsam dışını güvenilir biçimde saptadığımız iddia edilmez.
        no_features = vector.nnz == 0
        reasons = ["synthetic_demo_model"]
        if no_features:
            reasons.append("no_known_features")
        return {
            "category": None if no_features else str(self.pipeline.classes_[best]),
            "model_score": float(scores[best]),
            "score_type": "uncalibrated_model_score",
            "needs_review": True,
            "review_reasons": reasons,
            "entities": extract_entities(text),
            "model_version": self.metadata["model_version"],
        }

