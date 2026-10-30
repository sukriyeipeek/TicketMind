"""Local HTTP API: classification, annotation, experiments and sourced drafts."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import pickle
import threading
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from ticketmind import __version__
from ticketmind.data import PROJECT_ROOT
from ticketmind.service import TicketService

Category = Literal["kargo", "odeme", "iade", "teknik"]
ModelKey = Literal["baseline", "word", "char", "tinybert"]
NOT_READY_MESSAGE = "Model hazır değil. Proje klasöründe `python -m ticketmind.train` komutunu çalıştırın ve API'yi yeniden başlatın."


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[StrictStr, Field(min_length=1, max_length=4000)]

    @field_validator("text")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Mesaj yalnızca boşluklardan oluşamaz.")
        return value


class AnalyzeRequest(TextRequest):
    model: ModelKey = "baseline"


class EntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["order_id", "error_code", "product"]
    value: Annotated[StrictStr, Field(min_length=1)]
    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(gt=0)]
    method: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def require_ordered_span(self):
        if self.end <= self.start:
            raise ValueError("end, start değerinden büyük olmalı.")
        return self


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category | None
    model_score: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    category_scores: dict[Category, StrictFloat]
    score_type: Literal["uncalibrated_model_score"]
    needs_review: Literal[True]
    review_reasons: list[StrictStr]
    entities: list[EntityResponse]
    model_version: StrictStr
    input_truncated: bool = False

    @model_validator(mode="after")
    def require_complete_scores(self):
        values = self.category_scores
        if set(values) != {"kargo", "odeme", "iade", "teknik"}:
            raise ValueError("Dört kategori skoru da bulunmalı.")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("Geçersiz model skoru.")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Kategori skorları toplamı 1 olmalı.")
        if not math.isclose(max(values.values()), self.model_score, abs_tol=1e-5):
            raise ValueError("En yüksek skor model_score ile eşleşmeli.")
        return self


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    dataset_type: Literal["synthetic_demo"]


class AnnotationSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["order_id", "error_code", "product"]
    value: Annotated[StrictStr, Field(min_length=1)]
    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(gt=0)]


class AnnotationRequest(TextRequest):
    label: Literal["kargo", "odeme", "iade", "teknik", "review"]
    group_id: Annotated[StrictStr, Field(min_length=1, max_length=100)]
    note: Annotated[StrictStr, Field(max_length=1000)] = ""
    author_confirmed: Literal[True]
    entities: Annotated[list[AnnotationSpan], Field(max_length=30)] | None = None


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_transformer: bool = False
    epochs: Annotated[StrictInt, Field(ge=1, le=5)] = 3


class DraftRequest(TextRequest):
    provider: Literal["tfidf", "ollama"] = "tfidf"
    generator: Literal["extractive", "ollama"] = "extractive"
    category: Category | None = None


def create_app(model_path: Path | None = None, *, workbench_path: Path | None = None,
               models_dir: Path | None = None, reports_dir: Path | None = None) -> FastAPI:
    models = Path(models_dir or PROJECT_ROOT / "models")
    reports = Path(reports_dir or PROJECT_ROOT / "reports")
    artifact_path = Path(model_path or os.environ.get("TICKETMIND_MODEL_PATH", models / "baseline.joblib"))
    artifacts = {"baseline": artifact_path, "word": models / "word.joblib", "char": models / "char.joblib", "tinybert": models / "tinybert"}
    load_errors = (OSError, ValueError, EOFError, TypeError, KeyError, pickle.UnpicklingError, ImportError, RuntimeError)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        try:
            application.state.service = TicketService(artifact_path)
        except load_errors:
            pass
        yield
        application.state.service = None

    application = FastAPI(title="TicketMind-TR", version=__version__, lifespan=lifespan,
                          description="Yerel Türkçe NLP öğrenme atölyesi; bütün tahminler ve taslaklar insan incelemesi gerektirir.")
    application.state.service = None
    application.state.model_cache = {}
    application.state.model_lock = threading.RLock()
    application.state.workbench = None
    application.state.workbench_lock = threading.Lock()
    application.state.rag = {}
    application.state.rag_lock = threading.Lock()
    application.state.training_lock = threading.Lock()
    application.state.jobs = {}
    application.state.jobs_lock = threading.Lock()

    def get_service(key: str):
        if key == "baseline":
            if application.state.service is None:
                raise HTTPException(status_code=503, detail=NOT_READY_MESSAGE)
            return application.state.service
        if application.state.training_lock.locked():
            raise HTTPException(status_code=503, detail="Deney modelleri güncelleniyor; başlangıç modelini kullanabilirsiniz.")
        path = artifacts[key]
        marker = path / "metadata.json" if key == "tinybert" else path
        try:
            stamp = marker.stat().st_mtime_ns
            with application.state.model_lock:
                cached = application.state.model_cache.get(key)
                if cached is None or cached[0] != stamp:
                    cached = (stamp, TicketService(path))
                    application.state.model_cache[key] = cached
            return cached[1]
        except load_errors as error:
            raise HTTPException(status_code=503, detail="Seçilen model hazır değil. Deneyler sekmesinden ilgili eğitimi çalıştırın; transformer için isteğe bağlı paketler gerekir.") from error

    def store():
        from ticketmind.workbench import WorkbenchStore
        with application.state.workbench_lock:
            if application.state.workbench is None:
                application.state.workbench = WorkbenchStore(db_path=workbench_path)
            return application.state.workbench

    def read_report(name: str):
        try:
            return json.loads((reports / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Rapor henüz üretilmedi; ilgili değerlendirme komutunu çalıştırın.") from error

    @application.get("/health", response_model=HealthResponse)
    def health():
        loaded = application.state.service is not None
        body = {"status": "ready" if loaded else "not_ready", "model_loaded": loaded, "dataset_type": "synthetic_demo"}
        return body if loaded else JSONResponse(status_code=503, content=body)

    @application.get("/models")
    def list_models():
        labels = {"baseline": "Başlangıç · kelime TF-IDF", "word": "Deney · kelime TF-IDF", "char": "Deney · karakter TF-IDF", "tinybert": "Türkçe TinyBERT · fine-tuned"}
        return {"models": [{"key": key, "label": labels[key], "available": (path / "metadata.json").exists() if key == "tinybert" else path.is_file()}
                           for key, path in artifacts.items()], "all_require_review": True}

    @application.post("/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest):
        return get_service(payload.model).analyze(payload.text)

    @application.get("/workbench")
    def workbench_list(limit: Annotated[int, Field(ge=1, le=200)] = 100, offset: Annotated[int, Field(ge=0)] = 0):
        return {"stats": store().stats(), "records": store().list_records(limit=limit, offset=offset)}

    @application.post("/workbench", status_code=201)
    def workbench_add(payload: AnnotationRequest):
        try:
            record = store().add_record(text=payload.text, label=payload.label, group_id=payload.group_id,
                                       note=payload.note, entities=None if payload.entities is None else [entity.model_dump() for entity in payload.entities])
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"record": record}

    @application.get("/workbench/export")
    def workbench_export():
        return Response(content=store().export_jsonl(), media_type="application/x-ndjson",
                        headers={"Content-Disposition": 'attachment; filename="ticketmind-development.jsonl"'})

    @application.get("/data/development")
    def development():
        rows = [json.loads(line) for line in (PROJECT_ROOT / "data/development.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"records": rows, "role": "inspected_synthetic_development_only", "training_merged": False}

    @application.get("/experiments")
    def experiment_report():
        return read_report("model_comparison.json")

    @application.post("/experiments/run", status_code=202)
    def experiment_run(payload: ExperimentRequest):
        if not application.state.training_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Bir deney zaten çalışıyor.")
        job_id = uuid4().hex
        job = {"id": job_id, "status": "running", "include_transformer": payload.include_transformer, "epochs": payload.epochs}
        with application.state.jobs_lock:
            if len(application.state.jobs) >= 20:
                finished = next((key for key, value in application.state.jobs.items() if value["status"] != "running"), None)
                if finished:
                    application.state.jobs.pop(finished)
            application.state.jobs[job_id] = job

        def run():
            try:
                from ticketmind.experiments import compare
                report = compare(include_transformer=payload.include_transformer, epochs=payload.epochs)
                update = {"status": "completed", "selected_model": report["selection"]["selected_model"]}
            except Exception as error:
                update = {"status": "failed", "error": str(error)[:500]}
            finally:
                application.state.training_lock.release()
            with application.state.jobs_lock:
                application.state.jobs[job_id].update(update)

        threading.Thread(target=run, name="ticketmind-experiment", daemon=True).start()
        return job

    @application.get("/experiments/jobs/{job_id}")
    def experiment_status(job_id: str):
        with application.state.jobs_lock:
            job = application.state.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Deney kimliği bulunamadı.")
            return dict(job)

    @application.get("/evaluation/entities")
    def entity_report():
        return read_report("entities.json")

    @application.get("/evaluation/rag")
    def rag_report():
        return read_report("rag.json")

    @application.get("/rag/config")
    def rag_config():
        return {"embedding_model": os.environ.get("TICKETMIND_EMBED_MODEL", "embeddinggemma"),
"generation_model": os.environ.get("TICKETMIND_LLM_MODEL", "qwen2.5:7b"),
                "knowledge_base": "fictional_demo_store", "needs_review": True}

    @application.post("/rag/draft")
    def rag_draft(payload: DraftRequest):
        from ticketmind.rag import RagService
        key = (payload.provider, payload.generator)
        with application.state.rag_lock:
            if key not in application.state.rag:
                application.state.rag[key] = RagService(provider=payload.provider, generator=payload.generator)
            rag = application.state.rag[key]
        return rag.draft(payload.text, category=payload.category)

    return application


app = create_app()
