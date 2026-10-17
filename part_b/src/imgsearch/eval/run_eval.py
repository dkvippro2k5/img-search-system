"""Chấm hệ thống trên bộ golden (data/golden/gold/, định dạng trong README.md của bộ đó).

Ba phần:
1. Truy xuất: mỗi truy vấn trong queries.json chạy qua engine. Truy vấn mong đợi kết quả được
   tính P@10, Recall@50, nDCG@10 (độ liên quan 0/1/2), độ phủ nhóm@10; mọi truy vấn được chấm
   đúng/sai theo `expected_behavior`. Ảnh trong `possibly_relevant_unjudged` bị bỏ khỏi danh
   sách trước khi tính (không tính là sai khi chưa chấm tay).
2. Gộp entity: so cặp ảnh cùng người với persons.json (pairwise precision/recall).
3. Quan hệ: so cạnh trong meta.sqlite với relations.json.

Kết quả nhiều nhóm được trải thành một danh sách bằng cách xen kẽ từng nhóm (giống lưới
hiển thị theo nhóm). id ảnh so bằng 16 ký tự đầu của sha256.

Chạy (trong part_b/):
    python -m imgsearch.eval.run_eval --golden ../data/golden --index-version golden-auto
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.config import Settings
from imgsearch.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from imgsearch.retrieval.engine import SearchEngine
from imgsearch.retrieval.store import MetaStore

SHORT = 16


@dataclass
class GoldQuery:
    id: str
    type: str
    query: str
    expected: str
    scenarios: list[str] = field(default_factory=list)
    intent_mode: str | None = None
    relevant: dict[str, int] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    unjudged: set[str] = field(default_factory=set)
    note: str | None = None


def load_gold_queries(path: str | Path) -> list[GoldQuery]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldQuery(
            id=r["id"],
            type=r["type"],
            query=r["query"],
            expected=r["expected_behavior"],
            scenarios=r.get("scenarios") or [],
            intent_mode=r.get("intent_mode"),
            relevant={k[:SHORT]: v for k, v in (r.get("relevant") or {}).items()},
            groups={k: [i[:SHORT] for i in v] for k, v in (r.get("groups") or {}).items()},
            unjudged={i[:SHORT] for i in r.get("possibly_relevant_unjudged") or []},
            note=r.get("note"),
        )
        for r in rows
    ]


def flatten(resp: SearchResponse) -> list[str]:
    """Xen kẽ các nhóm: phần tử 1 của mỗi nhóm, rồi phần tử 2... Bỏ ảnh lặp giữa các nhóm."""
    lists = [[h.image_id[:SHORT] for h in g.hits] for g in resp.groups]
    out, seen = [], set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if rank < len(x) and x[rank] not in seen:
                seen.add(x[rank])
                out.append(x[rank])
    return out


def _has_msg(resp: SearchResponse, *fragments: str) -> bool:
    return any(f.lower() in m.lower() for m in resp.messages for f in fragments)


def judge_behavior(q: GoldQuery, resp: SearchResponse, ranked: list[str]) -> tuple[bool, str]:
    """Đúng/sai theo expected_behavior. Trả (đạt, lý do ngắn)."""
    hits = bool(ranked)
    e = q.expected
    if e == "results":
        return hits, "có kết quả" if hits else "không có kết quả"
    if e == "empty":
        return not hits, "rỗng" if not hits else "lẽ ra phải rỗng"
    if e == "empty_or_notice":
        return not hits, "rỗng" if not hits else "trả ảnh khi lẽ ra rỗng/báo"
    if e == "no_evidence":
        ok = not hits and not resp.reasoning and _has_msg(resp, "bằng chứng")
        return ok, "báo không có bằng chứng" if ok else "không báo đúng 'không có bằng chứng'"
    if e == "disambiguate":
        persons = [g for g in resp.groups if g.kind == "person"]
        ok = len(persons) >= 2 or _has_msg(resp, "khớp nhiều người")
        return ok, f"{len(persons)} nhóm theo người" if ok else "không tách / hỏi lại"
    if e == "disambiguate_or_no_evidence":
        ok = not hits and _has_msg(resp, "khớp nhiều người", "bằng chứng")
        return ok, "hỏi lại / báo không bằng chứng" if ok else "không xử lý đúng"
    if e == "partial_with_notice":
        ok = hits and _has_msg(resp, "không tìm thấy")
        return ok, "có kết quả + báo vế thiếu" if ok else "thiếu kết quả hoặc thiếu thông báo"
    return False, f"chưa hỗ trợ chấm '{e}'"


def score_query(q: GoldQuery, resp: SearchResponse) -> dict:
    ranked = flatten(resp)
    ok, why = judge_behavior(q, resp, ranked)
    out = {
        "id": q.id,
        "type": q.type,
        "query": q.query,
        "expected": q.expected,
        "scenarios": q.scenarios,
        "behavior_ok": ok,
        "behavior": why,
        "n_hits": len(ranked),
        "groups": [f"[{g.kind}] {g.label} ({g.total})" for g in resp.groups],
        "messages": resp.messages,
        "reasoning": [f"{s.obj} là {s.predicate_label} của {s.subj}" for s in resp.reasoning],
    }
    if q.relevant:
        judged = [i for i in ranked if i not in q.unjudged]
        rel = set(q.relevant)
        # P@10 bị chặn trên bởi |rel|/10 khi có ít hơn 10 ảnh đúng (Khả Như: 2 ảnh -> tối đa
        # 0,2), nên báo thêm P@R với R = min(|rel|, 10): 1,0 nghĩa là R ảnh đầu đều đúng.
        r_k = min(len(rel), 10)
        out.update(
            p10=precision_at_k(judged, rel, 10),
            p_r=precision_at_k(judged, rel, r_k),
            r50=recall_at_k(judged, rel, 50),
            ndcg10=ndcg_at_k(judged, {k: float(v) for k, v in q.relevant.items()}, 10),
        )
        if q.groups:
            top = set(judged[:10])
            out["group_cov10"] = sum(bool(top & set(v)) for v in q.groups.values()) / len(q.groups)
    return out


def evaluate(engine: SearchEngine, queries: list[GoldQuery]) -> list[dict]:
    results = []
    for q in queries:
        t = time.perf_counter()
        resp, _ = engine.search(SearchRequest(query=q.query, top_k=50))
        r = score_query(q, resp)
        r["ms"] = (time.perf_counter() - t) * 1000
        results.append(r)
    return results


def summarize(results: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        scored = [r for r in rs if "p10" in r]
        mean = lambda k, src=scored: (  # noqa: E731
            round(statistics.mean(r[k] for r in src if k in r), 3)
            if any(k in r for r in src)
            else None
        )
        return {
            "n": len(rs),
            "behavior_ok": sum(r["behavior_ok"] for r in rs),
            "n_scored": len(scored),
            "p10": mean("p10"),
            "p_r": mean("p_r"),
            "r50": mean("r50"),
            "ndcg10": mean("ndcg10"),
            "group_cov10": mean("group_cov10"),
        }

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    return {"all": agg(results), **{t: agg(rs) for t, rs in sorted(by_type.items())}}


# --- gộp entity và quan hệ ---


def _image_persons(store: MetaStore) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for image_id, person_id in store.conn.execute("SELECT image_id, person_id FROM image_person"):
        out[image_id[:SHORT]].add(person_id)
    return out


def merge_report(store: MetaStore, gold_persons: list[dict]) -> dict:
    """Pairwise trên ảnh golden đã vào index: hai ảnh 'cùng người' nếu có chung ít nhất một
    person (ảnh đôi thuộc cả hai người, ở cả gold lẫn hệ thống)."""
    pred = _image_persons(store)
    gold: dict[str, set[str]] = defaultdict(set)
    for p in gold_persons:
        for i in p["image_ids"]:
            if i in pred:
                gold[i].add(p["person_id"])
    ids = sorted(gold)
    tp = fp = fn = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            same_gold = bool(gold[ids[a]] & gold[ids[b]])
            same_pred = bool(pred[ids[a]] & pred[ids[b]])
            tp += same_gold and same_pred
            fp += same_pred and not same_gold
            fn += same_gold and not same_pred
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n_pred = len({p for i in ids for p in pred[i]})
    return {
        "images": len(ids),
        "gold_persons": len({p for ps in gold.values() for p in ps}),
        "system_persons": n_pred,
        "pair_precision": round(precision, 3),
        "pair_recall": round(recall, 3),
    }


def relation_report(store: MetaStore, gold_dir: Path, titles: set[str]) -> dict:
    """Ánh xạ person gold -> person hệ thống theo đa số ảnh, rồi so cạnh."""
    gold_persons = json.loads((gold_dir / "persons.json").read_text(encoding="utf-8"))
    gold_rel = json.loads((gold_dir / "relations.json").read_text(encoding="utf-8"))
    pred = _image_persons(store)
    to_sys = {}
    for p in gold_persons:
        votes = Counter(s for i in p["image_ids"] for s in pred.get(i, ()))
        if votes:
            to_sys[p["person_id"]] = votes.most_common(1)[0][0]

    def norm(subj: str, pred_: str, obj: str) -> tuple:
        return (
            (pred_, *sorted((subj, obj))) if pred_ in ("spouse", "sibling") else (pred_, subj, obj)
        )

    sys_edges = {
        norm(r["subj_person"], r["predicate"], r["obj_person"])
        for r in store.conn.execute("SELECT * FROM relation")
    }
    rows = []
    for r in gold_rel["relations"]:
        pred_map = {"child_of": "child", "spouse_of": "spouse"}
        if r["predicate"] not in pred_map:
            rows.append({**_gold_row(r), "found": None, "why": "loại cạnh chưa hỗ trợ"})
            continue
        s, o = to_sys.get(r["subject"]), to_sys.get(r["object"])
        edge = norm(o, "child", s) if r["predicate"] == "child_of" else norm(s, "spouse", o)
        in_meta = any(e["text"] in titles for e in r["evidence"] if e["kind"] == "source_title")
        rows.append({**_gold_row(r), "found": edge in sys_edges, "evidence_in_metadata": in_meta})
    gold_edges = {
        norm(to_sys.get(r["object"]), "child", to_sys.get(r["subject"]))
        if r["predicate"] == "child_of"
        else norm(to_sys.get(r["subject"]), "spouse", to_sys.get(r["object"]))
        for r in gold_rel["relations"]
        if r["predicate"] in ("child_of", "spouse_of")
    }
    return {
        "system_edges": len(sys_edges),
        "system_edges_correct": len(sys_edges & gold_edges),
        "gold": rows,
    }


def _gold_row(r: dict) -> dict:
    return {"edge": f"{r['subject']} {r['predicate']} {r['object']}", "status": r["status"]}


# --- báo cáo ---


def render_markdown(meta: dict, summary: dict, results: list[dict], merge: dict, rel: dict) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [
        f"# Đánh giá trên bộ golden — index `{meta['index_version']}`",
        "",
        f"Chạy lúc {meta['when']}. {meta['n_images']} ảnh trong index, {meta['n_persons']} person. "
        "Bộ golden đang được dùng để phát triển (không có tách dev/test), nên số đo lạc quan.",
        "",
        "## Tổng hợp",
        "",
        "| Loại | Số truy vấn | Hành vi đúng | Có điểm | P@10 | P@R | R@50 | nDCG@10 "
        "| Phủ nhóm@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['n']} | {s['behavior_ok']}/{s['n']} | {s['n_scored']} | "
            f"{fmt(s['p10'])} | {fmt(s['p_r'])} | {fmt(s['r50'])} | {fmt(s['ndcg10'])} | "
            f"{fmt(s['group_cov10'])} |"
        )
    lat = [r["ms"] for r in results]
    lines += [
        "",
        f"Latency engine (không gồm HTTP): p50 {statistics.median(lat):.1f} ms, "
        f"max {max(lat):.1f} ms. Bộ 500 ảnh, không dùng để kết luận về latency.",
        "",
        "## Gộp entity (so với gold/persons.json)",
        "",
        f"Ảnh golden đã vào index: {merge['images']}; person theo gold: {merge['gold_persons']}, "
        f"theo hệ thống: {merge['system_persons']}. Pairwise precision {merge['pair_precision']}, "
        f"recall {merge['pair_recall']}.",
        "",
        "## Quan hệ (so với gold/relations.json)",
        "",
        f"Hệ thống có {rel['system_edges']} cạnh, đúng {rel['system_edges_correct']}.",
        "",
        "| Cạnh gold | Trạng thái | Hệ thống tìm được | Bằng chứng có trong metadata |",
        "|---|---|---|---|",
    ]
    for g in rel["gold"]:
        found = "—" if g["found"] is None else ("có" if g["found"] else "không")
        in_meta = g.get("evidence_in_metadata")
        lines.append(
            f"| {g['edge']} | {g['status']} | {found} | "
            f"{'—' if in_meta is None else ('có' if in_meta else 'không')} |"
        )
    lines += [
        "",
        "## Từng truy vấn",
        "",
        "| id | Loại | Truy vấn | Mong đợi | Đạt | P@10 | P@R | R@50 | nDCG@10 | Phủ nhóm "
        "| Hệ thống trả |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        what = "; ".join(r["groups"][:3] + r["messages"][:2]) or "(rỗng)"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['query']} | {r['expected']} | "
            f"{'✓' if r['behavior_ok'] else '✗'} | {fmt(r.get('p10'))} | {fmt(r.get('p_r'))} | "
            f"{fmt(r.get('r50'))} | "
            f"{fmt(r.get('ndcg10'))} | {fmt(r.get('group_cov10'))} | {what.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Chấm hệ thống trên bộ golden")
    ap.add_argument("--golden", type=Path, default=Path("../data/golden"))
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--index-version", required=True)
    ap.add_argument(
        "--out", type=Path, default=None, help="mặc định ../reports/eval_golden_<ver>.md"
    )
    args = ap.parse_args(argv)

    engine = SearchEngine(
        Settings(_env_file=None, index_dir=args.index_dir, index_version=args.index_version)
    )
    engine.load()
    if engine.store is None:
        raise SystemExit(f"Không thấy index {args.index_dir}/v{args.index_version}")
    queries = load_gold_queries(args.golden / "gold" / "queries.json")
    results = evaluate(engine, queries)
    summary = summarize(results)
    gold_persons = json.loads((args.golden / "gold" / "persons.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.golden / "metadata.json").read_text(encoding="utf-8"))
    titles = {r["source_title"] for r in metadata}
    merge = merge_report(engine.store, gold_persons)
    rel = relation_report(engine.store, args.golden / "gold", titles)
    meta = {
        "index_version": args.index_version,
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "n_images": engine.n_images,
        "n_persons": len(engine.parser.aliases),
    }

    out = args.out or Path("../reports") / f"eval_golden_{args.index_version}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(meta, summary, results, merge, rel), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summary,
                "merge": merge,
                "relations": rel,
                "queries": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"merge: {merge}")
    print(f"relations: {rel['system_edges_correct']}/{rel['system_edges']} cạnh đúng")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
