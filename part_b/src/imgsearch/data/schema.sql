-- Schema SQLite cho meta.sqlite (PLAN.md 7.1). Khóa ảnh = sha256.
-- Chỉ chứa ảnh đã vào index: approved và không phải bản gần trùng.

CREATE TABLE person (
    person_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    aliases       TEXT,              -- JSON list
    category      TEXT,
    wikidata_qid  TEXT
);

CREATE TABLE entity (
    entity_id       TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    person_id       TEXT REFERENCES person(person_id),
    merge_method    TEXT             -- name | face | wikidata | manual
);

CREATE TABLE image (
    image_id         TEXT PRIMARY KEY, -- sha256
    image_url        TEXT,
    thumb_path       TEXT,
    width            INTEGER,
    height           INTEGER,
    bytes            INTEGER,
    phash            TEXT,
    caption          TEXT,             -- VLM của mentor
    description      TEXT,             -- tiêu đề bài báo nguồn
    source_url       TEXT,
    quality_flags    TEXT,
    nsfw_score       REAL,
    watermark_score  REAL,
    sensitive_flag   TEXT,
    status           TEXT,             -- approved | rejected | review
    vec_row          INTEGER UNIQUE    -- vị trí trong FAISS (0..N-1)
);

-- Ảnh trùng giữa nhiều entity: quan hệ nhiều-nhiều, không xóa (E7).
CREATE TABLE image_entity (
    image_id       TEXT REFERENCES image(image_id),
    entity_id      TEXT REFERENCES entity(entity_id),
    human_flag     TEXT,
    reject_reason  TEXT,
    PRIMARY KEY (image_id, entity_id)
);

-- Vật hóa image -> person (qua entity) để lọc theo người nhanh. Dựng lúc build.
CREATE TABLE image_person (
    image_id   TEXT NOT NULL REFERENCES image(image_id),
    person_id  TEXT NOT NULL REFERENCES person(person_id),
    PRIMARY KEY (person_id, image_id)
) WITHOUT ROWID;

-- Mỗi cạnh BẮT BUỘC có source + evidence; cạnh không bằng chứng bị loại lúc build.
-- Chiều: obj là <predicate> của subj. Ví dụ (Phạm Nhật Vượng, child, Phạm Nhật Minh Hoàng)
-- = Minh Hoàng là con của Phạm Nhật Vượng. Lưu một chiều; tra ngược ở retrieval/multi_hop.py.
CREATE TABLE relation (
    subj_person   TEXT NOT NULL REFERENCES person(person_id),
    predicate     TEXT NOT NULL,     -- child, spouse, parent, sibling, member_of...
    obj_person    TEXT NOT NULL REFERENCES person(person_id),
    source        TEXT NOT NULL,     -- wikidata | article | manual
    evidence      TEXT NOT NULL,
    evidence_url  TEXT
);

CREATE INDEX idx_entity_person ON entity(person_id);
CREATE INDEX idx_image_entity_entity ON image_entity(entity_id);
CREATE INDEX idx_image_person_image ON image_person(image_id);
CREATE INDEX idx_relation_subj ON relation(subj_person, predicate);
CREATE INDEX idx_relation_obj ON relation(obj_person, predicate);

-- Chỉ mục BM25 không nằm trong SQLite mà ở lexical.npz (retrieval/bm25.py):
-- FTS5 quá chậm với 80k ảnh (100-150 ms/truy vấn, đo 17/10).
