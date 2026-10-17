"""Multi-hop qua đồ thị tri thức: người gốc -> cạnh có bằng chứng -> người đích.

Cạnh lưu một chiều "obj là <predicate> của subj". Hỏi "<predicate> của A" thì người đích là:
- obj của các cạnh (A, predicate, obj), và
- subj của các cạnh (subj, nghịch đảo(predicate), A), ví dụ hỏi "con của A" cũng lấy được
  từ cạnh (B, parent, A) "A là bố/mẹ của B".

Không có cạnh -> engine trả "không tìm thấy bằng chứng quan hệ", KHÔNG trả ảnh (H3).
Không bao giờ suy quan hệ từ tên (Phạm Nhật Quân Anh không tự thành con của Phạm Nhật Vượng).
"""

from dataclasses import dataclass

from imgsearch.retrieval.store import MetaStore

INVERSE = {"child": "parent", "parent": "child", "spouse": "spouse", "sibling": "sibling"}


@dataclass(frozen=True)
class Edge:
    target: str  # person_id người đích
    subj: str
    predicate: str
    obj: str
    source: str
    evidence: str
    evidence_url: str | None = None


def find_edges(store: MetaStore, anchor: str, predicate: str) -> list[Edge]:
    inverse = INVERSE.get(predicate)
    edges: dict[tuple, Edge] = {}
    for r in store.relations_touching(anchor):
        if r["subj_person"] == anchor and r["predicate"] == predicate:
            target = r["obj_person"]
        elif inverse and r["obj_person"] == anchor and r["predicate"] == inverse:
            target = r["subj_person"]
        else:
            continue
        if target == anchor:
            continue
        edge = Edge(
            target,
            r["subj_person"],
            r["predicate"],
            r["obj_person"],
            r["source"],
            r["evidence"],
            r["evidence_url"],
        )
        edges.setdefault((target, edge.evidence), edge)
    return list(edges.values())
