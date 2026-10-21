"""Schema request/response của API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from imgsearch.config import get_settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(30, ge=1, le=100, description="Số ảnh tối đa mỗi nhóm")
    safe_search: bool | None = Field(None, description="Bỏ trống: theo SAFE_SEARCH_DEFAULT")

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
    # Ảnh cỡ gốc cho ô xem ảnh (thumb_url có thể là thumbnail 320 px trên S3); None: chỉ có thumb.
    image_url: str | None = None
    score: float
    person_ids: list[str] = []
    description: str | None = None  # tiêu đề bài báo nguồn, hiện để người dùng tự đánh giá (Q5)
    source_url: str | None = None


GroupKind = Literal["intent", "person", "co_occurrence", "relation_target"]


class ResultGroup(BaseModel):
    """Một nhóm kết quả.

    - intent: một ý định của truy vấn (single, mỗi vế của multi-intent)
    - person: một trong các người trùng tên (E2)
    - co_occurrence: ảnh có đủ mọi người (M1 nhóm phụ, M2, M7)
    - relation_target: người đích của truy vấn quan hệ (H1, H2)
    """

    label: str
    kind: GroupKind
    person_ids: list[str] = []
    hits: list[ImageHit]
    total: int  # số ảnh có trong nhóm trước khi cắt top_k


class EvidenceStep(BaseModel):
    """Một cạnh quan hệ: obj là <predicate> của subj."""

    subj: str
    subj_id: str
    predicate: str  # mã: child, spouse...
    predicate_label: str  # nhãn hiển thị: "con", "vợ/chồng"...
    obj: str
    obj_id: str
    source: str
    evidence: str
    evidence_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    query_type: str
    groups: list[ResultGroup]
    reasoning: list[EvidenceStep] = []
    messages: list[str] = []  # "không tìm thấy bằng chứng quan hệ", "không có trong dữ liệu"...
    suggestions: list[str] = []  # tên chuẩn cho "Bạn có muốn tìm..."
