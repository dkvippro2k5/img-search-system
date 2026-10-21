"""Điều phối truy xuất lai (H2): phân tích -> liên kết person -> truy xuất -> nhóm kết quả.

Luồng xử lý: PLAN.md 4.4. Hành vi theo từng kịch bản được kiểm ở tests/test_engine_search.py
(tên test theo mã SCENARIOS.md). Mỗi bước ghi thời gian vào `timings` (ms) cho Server-Timing.

Truy xuất một ý định (`_retrieve`): BM25 + dense ảnh (nếu index có dense.npz) + dense văn bản
(nếu có text_dense.npz và cấu hình mentor) gộp bằng fusion.rrf có trọng số `dense_weight`,
`text_dense_weight`. Dense ảnh nhận phần mô tả đã bỏ tên người ("Hari Won mặc váy cưới" ->
"mặc váy cưới"): model ảnh không biết mặt người nổi tiếng, danh tính đến từ lọc person. Dense văn
bản nhận cả câu (tiêu đề có nhắc tên), gọi mentor song song với hai kênh kia, lỗi thì bỏ kênh.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from imgsearch.api.schemas import EvidenceStep, ImageHit, ResultGroup, SearchRequest, SearchResponse
from imgsearch.config import Settings
from imgsearch.retrieval import guard, multi_hop, multi_intent
from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.aliases import AliasIndex
from imgsearch.retrieval.dense import DenseIndex, OnnxTextEncoder, TextEncoder, min_cosine
from imgsearch.retrieval.fusion import rrf
from imgsearch.retrieval.lexical import LexicalIndex
from imgsearch.retrieval.mentor import MentorClient, MentorTextEncoder
from imgsearch.retrieval.query_parser import (
    Intent,
    ParsedQuery,
    QueryParser,
    QueryType,
    tokenize,
)
from imgsearch.retrieval.store import MetaStore
from imgsearch.storage import s3

log = logging.getLogger(__name__)

Ranked = list[tuple[str, float]]

MSG_NO_INDEX = "Chưa nạp index, hệ thống chưa có dữ liệu để tìm."
MSG_NO_RESULT = "Không tìm thấy ảnh phù hợp."
MSG_REFUSED = (
    "Hệ thống không hỗ trợ tìm ảnh nhạy cảm của người thật (khỏa thân, ảnh nóng, nội y, thi thể…)."
)
MSG_UNSUPPORTED = {
    "negation": "Truy vấn phủ định (“không …”) chưa được hỗ trợ; kết quả có thể vẫn chứa "
    "điều bạn muốn loại trừ.",
    "count": "Lọc theo số người trong ảnh chưa được hỗ trợ.",
}


@dataclass
class _Group:
    label: str
    kind: str
    ranked: Ranked
    person_ids: list[str] = field(default_factory=list)


@dataclass
class _Draft:
    groups: list[_Group] = field(default_factory=list)
    reasoning: list[EvidenceStep] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        text_encoder: TextEncoder | None = None,
        query_text_encoder: TextEncoder | None = None,
    ):
        """`text_encoder`: thay encoder ONNX của kênh ảnh; `query_text_encoder`: thay encoder
        mentor của kênh văn bản (test dùng encoder giả, không cần model hay mạng)."""
        self.settings = settings
        self._text_encoder = text_encoder
        self._query_text_encoder = query_text_encoder
        self.ready = False
        self.n_images = 0
        self.store: MetaStore | None = None
        self.lexical: LexicalIndex | None = None
        self.dense: DenseIndex | None = None
        self.text_dense: DenseIndex | None = None
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="text-dense")
        self.parser = QueryParser(AliasIndex([]))
        self.unsafe_ids: set[str] = set()
        self._s3 = None

    def load(self) -> None:
        """Nạp index/v{INDEX_VERSION}/ (tải từ S3 trước nếu INDEX_FROM_S3) và dựng từ điển
        person. Thiếu index: chạy với index rỗng, trừ khi REQUIRE_INDEX (khi đó không bao giờ
        `ready`, /healthz trả 503)."""
        if self.settings.index_from_s3:
            self._download_index()
        meta = self.settings.index_path / "meta.sqlite"
        if meta.exists():
            self.store = MetaStore(meta)
            self.lexical = LexicalIndex.load(self.settings.index_path / "lexical.npz", self.store)
            self.parser = QueryParser(AliasIndex(self.store.persons()))
            self.n_images = self.store.count_images()
            self.unsafe_ids = self.store.unsafe_image_ids(
                self.settings.nsfw_block_score, self.settings.unsafe_flags
            )
            self.dense = self._load_dense()
            self.text_dense = self._load_text_dense()
            log.info(
                "Đã nạp %s: %d ảnh, %d person, %d ảnh bị lọc an toàn",
                meta,
                self.n_images,
                len(self.parser.aliases),
                len(self.unsafe_ids),
            )
        else:
            if self.settings.require_index:
                log.error("Không thấy %s và REQUIRE_INDEX=true: không nhận truy vấn", meta)
                return
            log.warning("Không thấy %s, chạy với index rỗng", meta)
        self.ready = True

    def _download_index(self) -> None:
        """Lỗi tải không chặn khởi động: dùng bản index đã có trên đĩa (nếu có)."""
        from imgsearch.storage.s3 import download_index

        t = time.perf_counter()
        try:
            n = download_index(self.settings)
        except Exception as e:  # noqa: BLE001 — mạng, quyền, bucket rỗng: đều quay về bản local
            log.error("Không tải được index từ S3 (%s); dùng bản trên đĩa nếu có", e)
            return
        log.info("Tải %d file index từ S3 trong %.1f s", n, time.perf_counter() - t)

    def _load_dense(self) -> DenseIndex | None:
        """Lỗi ở đây không chặn khởi động: log cảnh báo và chạy chỉ BM25."""
        s = self.settings
        path = s.index_path / "dense.npz"
        if not s.dense_enabled or not path.exists():
            return None
        manifest = json.loads((s.index_path / "manifest.json").read_text(encoding="utf-8"))
        model = manifest.get("dense", {}).get("model")
        if model != s.dense_model:
            log.warning(
                "dense.npz dựng bằng %s nhưng DENSE_MODEL=%s: tắt dense", model, s.dense_model
            )
            return None
        try:
            encoder = self._text_encoder or OnnxTextEncoder(
                s.models_dir / model, s.dense_onnx_file, s.onnx_threads, s.query_cache_size
            )
            meta = getattr(encoder, "meta", {})
            dense = DenseIndex.load(
                path,
                self.lexical.image_ids,
                encoder,
                self.lexical.allowed_rows,
                min_cosine(meta, s.dense_min_prob),
            )
        except (OSError, ValueError) as e:
            log.warning("Không nạp được dense (%s): chạy chỉ BM25", e)
            return None
        log.info("Dense %s: %.1f%% ảnh có vector", model, 100 * dense.coverage)
        return dense

    def _load_text_dense(self) -> DenseIndex | None:
        """Kênh văn bản (P3): text_dense.npz + dịch vụ embedding của mentor. Thiếu file, thiếu
        cấu hình hoặc model khác manifest thì tắt kênh (log), không chặn khởi động."""
        s = self.settings
        path = s.index_path / "text_dense.npz"
        if not s.text_dense_enabled or not path.exists():
            return None
        manifest = json.loads((s.index_path / "manifest.json").read_text(encoding="utf-8"))
        model = manifest.get("text_dense", {}).get("model")
        if model != s.text_embed_model:
            log.warning("text_dense.npz là %s, TEXT_EMBED_MODEL=%s: tắt", model, s.text_embed_model)
            return None
        encoder = self._query_text_encoder
        if encoder is None:
            if not (s.mentor_api_key and s.text_embed_url):
                log.warning("Có text_dense.npz nhưng thiếu MENTOR_API_KEY/TEXT_EMBED_URL: tắt")
                return None
            client = MentorClient(
                s.mentor_api_key,
                s.text_embed_url,
                s.text_embed_model,
                timeout=s.mentor_timeout_seconds,
                cache_size=s.query_cache_size,
            )
            encoder = MentorTextEncoder(client, s.text_dense_dim, model)
        with np.load(path, allow_pickle=False) as z:
            vecs, has = z["vecs"][:, : encoder.dim].astype(np.float32), z["has_vec"]
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        try:
            dense = DenseIndex(
                vecs, has, self.lexical.image_ids, encoder, self.lexical.allowed_rows
            )
        except ValueError as e:
            log.warning("Không nạp được text_dense (%s): tắt kênh văn bản", e)
            return None
        pct = 100 * dense.coverage
        log.info("Dense văn bản %s (%d chiều): %.1f%% ảnh", model, encoder.dim, pct)
        return dense

    def info(self) -> dict:
        return {
            "index_version": self.settings.index_version,
            "index_loaded": self.store is not None,
            "n_images": self.n_images,
            "n_persons": len(self.parser.aliases),
            "dense": self.dense.encoder.name if self.dense else None,
            "text_dense": self.text_dense.encoder.name if self.text_dense else None,
            "allow_unsafe_search": self.settings.allow_unsafe_search,
        }

    def mentor_stats(self) -> dict | None:
        """Bộ đếm lời gọi mentor của kênh văn bản (cho /metrics); None khi kênh tắt."""
        client = getattr(self.text_dense and self.text_dense.encoder, "client", None)
        return dict(client.stats) if client is not None else None

    # --- điểm vào ---

    def search(self, req: SearchRequest) -> tuple[SearchResponse, dict[str, float]]:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        parsed = self.parser.parse(req.query)
        timings["parse"] = (time.perf_counter() - t) * 1000
        blocked = guard.check(req.query, has_person=bool(parsed.mentions))
        if blocked.blocked:  # X2: không truy xuất, không gọi mentor
            log.warning("Từ chối truy vấn nhạy cảm (X2) %s: %r", blocked.terms, req.query)
            refused = SearchResponse(
                query=req.query, query_type="refused", groups=[], messages=[MSG_REFUSED]
            )
            return refused, timings

        t = time.perf_counter()
        self._prefetch_text(parsed)
        draft = _Draft()
        if self.store is None:
            draft.messages.append(MSG_NO_INDEX)
        elif parsed.type == QueryType.MULTI_HOP:
            self._multi_hop(parsed, draft)
        elif parsed.type == QueryType.MULTI_INTENT:
            self._multi_intent(parsed, draft)
        else:
            self._single(parsed, draft)
        timings["retrieve"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        safe = self.settings.safe_search_default if req.safe_search is None else req.safe_search
        safe = safe or not self.settings.allow_unsafe_search  # tắt lọc chỉ khi server cho phép
        groups = self._hydrate(draft.groups, req.top_k, safe) if self.store else []
        timings["hydrate"] = (time.perf_counter() - t) * 1000

        messages = draft.messages + [MSG_UNSUPPORTED[u] for u in parsed.unsupported]
        if not groups and not draft.messages:
            messages.append(MSG_NO_RESULT)
        response = SearchResponse(
            query=req.query,
            query_type=parsed.type,
            groups=groups,
            reasoning=draft.reasoning,
            messages=_unique(messages),
            suggestions=parsed.suggestions,
        )
        return response, timings

    # --- theo loại truy vấn ---

    def _single(self, parsed: ParsedQuery, draft: _Draft) -> None:
        if not parsed.intents:
            return
        intent = parsed.intents[0]
        if not self._report_unknown(intent, draft):
            return
        if len(intent.person_ids) > 1 and parsed.ambiguous:
            # E2: tên trùng nhiều người -> mỗi người một nhóm, kèm thông báo.
            names = ", ".join(self._person_label(p) for p in intent.person_ids)
            draft.messages.append(
                f"“{', '.join(parsed.ambiguous)}” khớp nhiều người: {names}. "
                "Kết quả được chia theo từng người; thêm nghề trước tên để chọn đúng người."
            )
            for pid in intent.person_ids:
                draft.groups.append(
                    _Group(
                        self._person_label(pid),
                        "person",
                        self._retrieve(intent.text, [pid], mentions=intent.mentions),
                        [pid],
                    )
                )
            return
        draft.groups.append(
            _Group(
                self._intent_label(intent),
                "intent",
                self._retrieve(intent.text, intent.person_ids, intent.categories, intent.mentions),
                intent.person_ids,
            )
        )

    def _multi_intent(self, parsed: ParsedQuery, draft: _Draft) -> None:
        intents = [i for i in parsed.intents if self._report_unknown(i, draft)]
        with_people = [i for i in intents if i.person_ids]
        # Giao chỉ có nghĩa khi có >= 2 người khác nhau (M3 "Mỹ Tâm ... và Mỹ Tâm ..." thì không).
        several_people = len({tuple(i.person_ids) for i in with_people}) >= 2

        if parsed.co_occurrence and several_people:
            group = self._co_occurrence_group(parsed.raw, with_people)
            if group:
                draft.groups.append(group)
                return
            names = " và ".join(self._intent_label(i) for i in with_people)
            draft.messages.append(f"Không có ảnh nào có cả {names}; hiển thị ảnh của từng người.")

        for intent in intents:
            draft.groups.append(
                _Group(
                    self._intent_label(intent),
                    "intent",
                    self._retrieve(
                        intent.text, intent.person_ids, intent.categories, intent.mentions
                    ),
                    intent.person_ids,
                )
            )

        # M1: "A và B" hiểu là mỗi người một nhóm; có ảnh chung thì thêm nhóm phụ.
        if not parsed.co_occurrence and several_people:
            group = self._co_occurrence_group(parsed.raw, with_people)
            if group:
                group.label = f"Cùng xuất hiện: {group.label}"
                draft.groups.append(group)

    def _multi_hop(self, parsed: ParsedQuery, draft: _Draft) -> None:
        anchors = parsed.anchor_person_ids
        predicate = parsed.relation_path[0]
        relation = lx.PREDICATE_LABELS.get(predicate, predicate)

        if len(anchors) > 1:
            # H7: không chọn bừa người gốc.
            names = ", ".join(self._person_label(p) for p in anchors)
            draft.messages.append(
                f"“{', '.join(parsed.ambiguous)}” khớp nhiều người: {names}. "
                "Hãy ghi rõ người nào (ví dụ thêm nghề trước tên) để tra quan hệ."
            )
            return

        anchor = anchors[0]
        anchor_name = self._name(anchor)
        edges = multi_hop.find_edges(self.store, anchor, predicate)
        if not edges:
            draft.messages.append(
                f"Không tìm thấy bằng chứng quan hệ “{relation}” của {anchor_name} trong dữ liệu."
            )
            return

        by_target: dict[str, list[multi_hop.Edge]] = {}
        for edge in edges:
            by_target.setdefault(edge.target, []).append(edge)
        for target, target_edges in by_target.items():
            draft.reasoning += [self._evidence(e) for e in target_edges]
            ranked = [(i, 0.0) for i in self.store.images_of([target])]
            target_name = self._name(target)
            if not ranked:
                # H4: có bằng chứng nhưng người đích không có ảnh.
                draft.messages.append(
                    f"Có bằng chứng {target_name} là {relation} của {anchor_name}, "
                    f"nhưng dữ liệu không có ảnh của {target_name}."
                )
                continue
            draft.groups.append(
                _Group(
                    f"{target_name} ({relation} của {anchor_name})",
                    "relation_target",
                    ranked,
                    [target],
                )
            )

    # --- truy xuất ---

    def _retrieve(
        self,
        text: str,
        person_ids: list[str],
        categories: list[str] | None = None,
        mentions: list[str] | None = None,
    ) -> Ranked:
        """Xếp hạng ảnh cho một ý định.

        - Có person: ảnh khớp mô tả của person đó đứng trước (BM25 + dense trên phần mô tả đã bỏ
          tên), rồi mọi ảnh còn lại của person (không làm mất ảnh chỉ vì caption không nhắc tới
          mô tả, S1).
        - Không person: BM25 + dense toàn bộ (giới hạn theo nghề nếu có), đa dạng hóa theo người.
        """
        pool = self.settings.candidate_pool
        if person_ids:
            desc = _residual(text, mentions or [])
            # Kênh văn bản chỉ khi có phần mô tả (như kênh ảnh); gọi mạng song song với BM25.
            text_job = self._text_job(text, pool, person_ids=person_ids) if desc else None
            hits = self.lexical.search(text, limit=pool, person_ids=person_ids)
            img = []
            if desc and self.dense:
                img = self.dense.search(desc, pool, person_ids=person_ids)
            hits = self._fuse(hits, img, self._text_result(text_job))
            seen = {i for i, _ in hits}
            return hits + [(i, 0.0) for i in self.store.images_of(person_ids) if i not in seen]

        text_job = self._text_job(text, pool, categories=categories)
        hits = self.lexical.search(text, limit=pool, categories=categories) if categories else []
        if not hits:
            hits = self.lexical.search(text, limit=pool)
        img = []
        if self.dense:
            img = self.dense.search(text, pool, categories=categories) if categories else []
            img = img or self.dense.search(text, pool)
        hits = self._fuse(hits, img, self._text_result(text_job))
        persons = self.store.persons_of([i for i, _ in hits])
        return multi_intent.diversify(hits, persons, self.settings.per_person_cap)

    def prefetch(self, query: str) -> None:
        """Gửi trước embedding kênh văn bản cho `query` (API gọi trước khi xếp hàng semaphore).
        search() sau đó dùng lại lời gọi đang bay hoặc cache; không có kênh thì không làm gì."""
        if self.text_dense is None:
            return
        try:
            parsed = self.parser.parse(query)
            if not guard.check(query, has_person=bool(parsed.mentions)).blocked:
                self._prefetch_text(parsed)
        except Exception as e:  # noqa: BLE001 — chỉ là tối ưu, lỗi thì search() tự gọi
            log.warning("prefetch lỗi (%s)", e)

    def _prefetch_text(self, parsed: ParsedQuery) -> None:
        """Gửi trước embedding kênh văn bản của mọi vế sẽ dùng tới (cùng điều kiện _retrieve:
        vế không có người, hoặc có người kèm phần mô tả)."""
        prefetch = getattr(self.text_dense and self.text_dense.encoder, "prefetch", None)
        if prefetch is None or parsed.type == QueryType.MULTI_HOP:
            return
        prefetch(
            [
                i.text
                for i in parsed.intents
                if i.text.strip() and (not i.person_ids or _residual(i.text, i.mentions))
            ]
        )

    def _text_job(self, text: str, pool: int, **filters):
        if self.text_dense is None or not text.strip():
            return None
        return self._pool.submit(self.text_dense.search, text, pool, **filters)

    def _text_result(self, job) -> Ranked:
        """Kết quả kênh văn bản; lỗi bất kỳ (mạng đã có timeout trong client) -> bỏ kênh."""
        if job is None:
            return []
        try:
            return job.result(timeout=self.settings.mentor_timeout_seconds + 0.5)
        except Exception as e:  # noqa: BLE001 — kênh phụ, không được làm hỏng truy vấn
            log.warning("Kênh văn bản lỗi (%s): bỏ qua", e)
            return []

    def _fuse(self, lexical: Ranked, dense: Ranked, text: Ranked | None = None) -> Ranked:
        """RRF có trọng số: BM25 (1), ảnh (`dense_weight`), văn bản (`text_dense_weight`)."""
        lists = [(lexical, 1.0), (dense, self.settings.dense_weight)]
        lists.append((text or [], self.settings.text_dense_weight))
        lists = [(r, w) for r, w in lists if r]
        if len(lists) <= 1:
            return lists[0][0] if lists else lexical
        fused = rrf([[i for i, _ in r] for r, _ in lists], weights=[w for _, w in lists])
        return fused[: self.settings.candidate_pool]

    def _co_occurrence_group(self, raw: str, intents: list[Intent]) -> _Group | None:
        common = multi_intent.intersect(set(self.store.images_of(i.person_ids)) for i in intents)
        if not common:
            return None
        person_ids = _unique(p for i in intents for p in i.person_ids)
        hits = self.lexical.search(raw, limit=self.settings.candidate_pool, person_ids=person_ids)
        ranked = multi_intent.restrict(hits, common, self.store.images_of(person_ids))
        label = " + ".join(self._intent_label(i) for i in intents)
        return _Group(label, "co_occurrence", ranked, person_ids)

    # --- dựng kết quả ---

    def _hydrate(self, groups: list[_Group], top_k: int, safe: bool) -> list[ResultGroup]:
        unsafe = self.unsafe_ids if safe else set()
        trimmed = []
        for g in groups:
            first: dict[str, float] = {}  # E7: mỗi ảnh một lần, giữ vị trí và điểm lần đầu
            for i, s in g.ranked:
                if i not in unsafe:
                    first.setdefault(i, s)
            ranked = list(first.items())
            if ranked:
                trimmed.append((g, ranked[:top_k], len(ranked)))

        ids = _unique(i for _, top, _ in trimmed for i, _ in top)
        rows = self.store.get_images(ids)
        persons = self.store.persons_of(ids)
        return [
            ResultGroup(
                label=g.label,
                kind=g.kind,
                person_ids=g.person_ids,
                total=total,
                hits=[
                    ImageHit(
                        image_id=i,
                        thumb_url=self._thumb_url(rows[i]),
                        score=round(s, 4),
                        person_ids=sorted(persons.get(i, [])),
                        description=rows[i]["description"],
                        source_url=rows[i]["source_url"],
                    )
                    for i, s in top
                ],
            )
            for g, top, total in trimmed
        ]

    def _thumb_url(self, row: dict) -> str:
        if self.settings.local_images:
            return f"/img/{row['image_id']}"
        if self.settings.thumbs_from_s3:
            if self._s3 is None:
                self._s3 = s3.client(self.settings)
            return s3.presigned_thumb_url(self.settings, row["image_id"], self._s3)
        return row["image_url"] or ""

    # --- nhãn, thông báo ---

    def _report_unknown(self, intent: Intent, draft: _Draft) -> bool:
        """Báo tên lạ; trả False nếu vế này không còn gì để tìm (E6, M5, X1).

        Không tìm BM25 bằng tên lạ: sẽ ra người có tên gần giống, trái với E6.
        """
        for name in intent.unknown_names:
            draft.messages.append(f"Không tìm thấy “{name}” trong danh sách người của dữ liệu.")
        return bool(intent.person_ids or not intent.unknown_names)

    def _name(self, person_id: str) -> str:
        person = self.parser.aliases.persons.get(person_id)
        return person.name if person else person_id

    def _person_label(self, person_id: str) -> str:
        person = self.parser.aliases.persons.get(person_id)
        if person is None:
            return person_id
        category = lx.CATEGORY_LABELS.get(person.category or "")
        return f"{person.name} ({category})" if category else person.name

    def _intent_label(self, intent: Intent) -> str:
        """Tên chuẩn của người + phần mô tả còn lại: "den vau" -> "Đen Vâu",
        "Mỹ Tâm trên sân khấu" -> "Mỹ Tâm · xếp theo “trên sân khấu”". Mô tả ghi là tiêu chí xếp
        hạng, không ghép thẳng vào tên: nhóm gồm mọi ảnh của người đó (S1), nên "Phạm Nhật Vượng —
        bị bắt" sẽ đọc như hệ thống xác nhận một tin chưa có thật."""
        if not intent.person_ids:
            return intent.text
        names = " / ".join(self._name(p) for p in intent.person_ids)
        rest = _residual(intent.text, intent.mentions)
        return f"{names} · xếp theo “{rest}”" if rest else names

    def _evidence(self, edge: multi_hop.Edge) -> EvidenceStep:
        return EvidenceStep(
            subj=self._name(edge.subj),
            subj_id=edge.subj,
            predicate=edge.predicate,
            predicate_label=lx.PREDICATE_LABELS.get(edge.predicate, edge.predicate),
            obj=self._name(edge.obj),
            obj_id=edge.obj,
            source=edge.source,
            evidence=edge.evidence,
            evidence_url=edge.evidence_url,
        )


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def _residual(text: str, mentions: list[str]) -> str:
    """Phần chữ còn lại sau khi bỏ các tên người; rỗng nếu chỉ còn từ chức năng ("ảnh")."""
    words = [t for t in tokenize(text) if t.is_word]
    folds = [t.fold for t in words]
    keep = [True] * len(words)
    for mention in mentions:
        m = [t.fold for t in tokenize(mention) if t.is_word]
        for i in range(len(folds) - len(m) + 1):
            if m and folds[i : i + len(m)] == m and all(keep[i : i + len(m)]):
                keep[i : i + len(m)] = [False] * len(m)
                break
    rest = [t for t, k in zip(words, keep, strict=True) if k]
    if all(t.fold in lx.STOPWORDS for t in rest):
        return ""
    return " ".join(t.surface for t in rest)
