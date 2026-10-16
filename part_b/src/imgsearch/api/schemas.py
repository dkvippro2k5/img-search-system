"""Schema request/response của API."""

from pydantic import BaseModel, Field, field_validator

from imgsearch.config import get_settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(30, ge=1, le=100)
    safe_search: bool = True

    @field_validator("query")
    @classmethod
    def _check_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query rỗng")
        if len(v) > get_settings().max_query_chars:
            raise ValueError("query quá dài")
        return v


class ImageHit(BaseModel):
    image_id: str
    thumb_url: str
    score: float
    person_ids: list[str] = []
    description: str | None = None
    source_url: str | None = None


class ResultGroup(BaseModel):
    """Một nhóm kết quả: theo ý định (multi-intent) hoặc theo người đích (multi-hop)."""

    label: str
    hits: list[ImageHit]
    total: int


class EvidenceStep(BaseModel):
    subj: str
    predicate: str
    obj: str
    source: str
    evidence: str
    evidence_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    query_type: str
    groups: list[ResultGroup]
    reasoning: list[EvidenceStep] = []
    messages: list[str] = []  # "không tìm thấy bằng chứng quan hệ", "đã bỏ điều kiện..."
    suggestions: list[str] = []  # "Bạn có muốn tìm..."
