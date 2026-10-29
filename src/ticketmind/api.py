"""HTTP boundary for the trained, local TicketMind model.

The API only loads an existing artifact. Training is an explicit offline step.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import pickle
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ticketmind.service import TicketService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOT_READY_MESSAGE = (
    "Model hazır değil. Proje klasöründe `python -m ticketmind.train` komutunu "
    "çalıştırın ve API'yi yeniden başlatın."
)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[StrictStr, Field(min_length=1, max_length=4000)]

    @field_validator("text")
    @classmethod
    def require_visible_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Mesaj yalnızca boşluklardan oluşamaz.")
        # Keep original text and offsets; normalization belongs to the model.
        return text


class EntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Literal["order_id", "error_code", "product"]
    value: Annotated[StrictStr, Field(min_length=1)]
    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(gt=0)]
    method: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def require_ordered_span(self) -> EntityResponse:
        if self.end <= self.start:
            raise ValueError("end, start değerinden büyük olmalıdır.")
        return self


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["kargo", "odeme", "iade", "teknik"] | None
    model_score: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
    score_type: Literal["uncalibrated_model_score"]
    needs_review: Literal[True]
    review_reasons: list[Annotated[StrictStr, Field(min_length=1)]]
    entities: list[EntityResponse]
    model_version: Annotated[StrictStr, Field(min_length=1)]


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    dataset_type: Literal["synthetic_demo"]


def create_app(model_path: Path | None = None) -> FastAPI:
    """Create an isolated app; an explicit model path also makes tests simple."""
    artifact_path = Path(
        model_path
        if model_path is not None
        else os.environ.get(
            "TICKETMIND_MODEL_PATH", str(PROJECT_ROOT / "models" / "baseline.joblib")
        )
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        try:
            application.state.service = TicketService(artifact_path)
        except (OSError, ValueError, EOFError, TypeError, KeyError, pickle.UnpicklingError):
            # Do not expose paths, model internals, or customer messages.
            # A missing or invalid artifact is an unavailable model, not training.
            application.state.service = None
        yield
        application.state.service = None

    application = FastAPI(
        title="TicketMind-TR",
        description=(
            "Türkçe destek talebi analizi. Sentetik veriyle hazırlanmış öğrenme "
            "demosudur; bütün sonuçlar insan incelemesi gerektirir."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.service = None

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Model hazır değil"}},
    )
    def health(request: Request):
        loaded = request.app.state.service is not None
        body = HealthResponse(
            status="ready" if loaded else "not_ready",
            model_loaded=loaded,
            dataset_type="synthetic_demo",
        )
        if not loaded:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @application.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={503: {"description": "Önce model eğitilmeli"}},
    )
    def analyze(payload: AnalyzeRequest, request: Request):
        service = request.app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail=NOT_READY_MESSAGE)
        return service.analyze(payload.text)

    return application


app = create_app()
