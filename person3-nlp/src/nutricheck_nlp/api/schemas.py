"""Pydantic request/response models (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..text_normalizer import MAX_TEXT_CHARS


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="English clinical note / OCR text", max_length=MAX_TEXT_CHARS)
    sex: Literal["M", "F"] | None = Field(None, description="Optional; auto-detected from text if omitted")
    ocr_confidence: float | None = Field(None, ge=0.0, le=1.0)
    use_biogpt: bool = Field(False, description="Add BioGPT second opinion (needs torch + model)")
    explain: bool = Field(False, description="Also generate a short BioGPT explanation")
    use_classifier: bool | None = Field(None, description="Trained context classifier as verifier: null = use it if available")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class ReportItem(BaseModel):
    label: str = Field("", max_length=120, description="e.g. 'January 2025'")
    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    sex: Literal["M", "F"] | None = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class RecordsRequest(BaseModel):
    reports: list[ReportItem] = Field(..., min_length=1, max_length=20)
    date_order: Literal["DMY", "MDY"] | None = Field(None, description="How to read numeric dates such as 03/04/2025 (never guessed if omitted)")


class ExplainRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    level: Literal["simple_en", "detailed_en", "medical_en", "te", "hi"] = "simple_en"
    sex: Literal["M", "F"] | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=2000, description="Question asked by the user")
    result: dict[str, Any] = Field(default_factory=dict, description="Active report analysis result")
    report_text: str = Field("", description="Original report text if available")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class VisitPrepRequest(RecordsRequest):
    language: Literal["en", "hi", "te"] = "en"
    generated_on: str | None = Field(None, max_length=40)


class MaskRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)


class VaultStoreRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)
    label: str = Field("", max_length=120)
    consent: bool = Field(False, description="Must be true: nothing is stored without explicit permission")
    sex: Literal["M", "F"] | None = None
    date_order: Literal["DMY", "MDY"] | None = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    use_biogpt: bool = False
    use_classifier: bool | None = None

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v: list[str]) -> list[str]:
        for t in v:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("every text must be a non-blank string")
            if len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"text longer than {MAX_TEXT_CHARS} characters")
        return v


class NutrientResult(BaseModel):
    key: str
    name: str
    status: Literal["deficient", "borderline", "suspected", "not_deficient", "not_assessed", "conflicting"]
    status_label: str
    confidence: float
    reliability: Literal["ok", "verify", "abstain"]
    abstain_reasons: list[str] = []
    lab_status: str
    text_status: str
    evidence: list[str]
    evidence_items: list[dict[str, Any]] = []
    detective: dict[str, Any] = {}
    notes: list[str]
    needs_review: bool = False
    review_reasons: list[str] = []
    history_of_deficiency: bool
    historical_values: list[dict[str, Any]] = []
    under_treatment: bool
    recommendation: str


class AnalysisResponse(BaseModel):
    version: str
    language: str
    language_info: dict[str, Any] = {}
    language_supported: bool = True
    document_profile: dict[str, Any] = {}
    patient: dict[str, Any]
    nutrients: list[NutrientResult]
    findings: list[NutrientResult]
    summary: dict[str, list[str]]
    labs: list[dict[str, Any]]
    rejected_values: list[dict[str, Any]]
    mentions: list[dict[str, Any]]
    supporting_symptoms: list[dict[str, Any]]
    symptom_statements: list[dict[str, Any]] = []
    risk_factors: list[dict[str, Any]]
    treatments: list[dict[str, Any]]
    treatment_statements: list[dict[str, Any]] = []
    dates: dict[str, Any] = {}
    suggestions: list[str]
    warnings: list[str]
    needs_human_review: bool
    quality: dict[str, Any] = {}
    security: dict[str, Any] = {}
    input_rejected: dict[str, Any] | None = None
    disclaimer: str
    biogpt: dict[str, Any] | None = None
    classifier: dict[str, Any] | None = None
    ocr: dict[str, Any] | None = None
    record_id: str | None = None
    document: dict[str, Any] | None = None
    processing_report: dict[str, Any] | None = None
    processing_ms: float


class BatchResponse(BaseModel):
    count: int
    results: list[AnalysisResponse]


class FileAnalysisResponse(BaseModel):
    document: dict[str, Any]
    count: int
    results: list[AnalysisResponse]
