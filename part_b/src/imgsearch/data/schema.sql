-- Schema SQLite cho meta.sqlite (PLAN.md 7.1). Khóa ảnh = sha256.

CREATE TABLE IF NOT EXISTS person (
    person_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    aliases       TEXT,              -- JSON, gồm dạng không dấu
    category      TEXT,
    wikidata_qid  TEXT
);

CREATE TABLE IF NOT EXISTS entity (
    entity_id       TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    person_id       TEXT REFERENCES person(person_id),
    merge_method    TEXT             -- name | face | wikidata | manual
);

CREATE TABLE IF NOT EXISTS image (
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
    vec_row          INTEGER           -- vị trí trong FAISS
);

-- Ảnh trùng giữa nhiều entity: quan hệ nhiều-nhiều, không xóa.
CREATE TABLE IF NOT EXISTS image_entity (
    image_id       TEXT REFERENCES image(image_id),
    entity_id      TEXT REFERENCES entity(entity_id),
    human_flag     TEXT,
    reject_reason  TEXT,
    PRIMARY KEY (image_id, entity_id)
);

-- Mỗi cạnh BẮT BUỘC có source + evidence; cạnh không bằng chứng không được dùng.
CREATE TABLE IF NOT EXISTS relation (
    subj_person   TEXT NOT NULL REFERENCES person(person_id),
    predicate     TEXT NOT NULL,     -- child, spouse, parent, sibling, member_of...
    obj_person    TEXT NOT NULL REFERENCES person(person_id),
    source        TEXT NOT NULL,     -- wikidata | article | manual
    evidence      TEXT NOT NULL,
    evidence_url  TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_person ON entity(person_id);
CREATE INDEX IF NOT EXISTS idx_image_entity_entity ON image_entity(entity_id);
CREATE INDEX IF NOT EXISTS idx_relation_subj ON relation(subj_person, predicate);

-- BM25 trên caption + tiêu đề + tên person; cột *_fold là dạng không dấu.
CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
    image_id UNINDEXED,
    text,
    text_fold,
    tokenize = 'unicode61 remove_diacritics 0'
);
