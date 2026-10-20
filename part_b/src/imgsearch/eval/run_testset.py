"""Chấm hệ thống trên test set JSONL (PLAN 5.1–5.3) -> ../reports/eval_<split>.md + .json.

python -m imgsearch.eval.run_testset --index-version people --split dev
python -m imgsearch.eval.run_testset --index-version people --split test   # chỉ khi báo cáo

Chỉ số (PLAN 5.2), tính trên danh sách xen kẽ các nhóm (run_eval.flatten), top 50 mỗi nhóm:
- P@10, P@10 nhãn 2 (truy vấn chấm tay: chỉ tính ảnh khớp rõ), P@R (R = min(số ảnh đúng, 10)),
  R@50 chuẩn hóa (ảnh đúng trong top 50 / min(số ảnh đúng, 50): person có 276 ảnh thì 50 ảnh
  đúng đầu tiên là trọn điểm), nDCG@10 (nhãn 0/1/2 nếu có), MRR.
- Multi-intent: độ phủ ý định@10, tỷ lệ nhỏ nhất của một ý định trong top 10 (M6).
- Multi-hop: liên kết người gốc đúng, người đích đúng, từ chối đúng, tỷ lệ bịa.
Ground truth: imgsearch/eval/testset.py. Ảnh bị lọc an toàn bị bỏ khỏi ground truth (hệ thống
ẩn đúng các ảnh đó). Ảnh chưa chấm trong truy vấn "pool" tính là không liên quan.
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.eval.metrics import (
    intent_coverage_at_k,
    min_intent_share_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from imgsearch.eval.pooling import CONFIGS, available, config, eval_settings
from imgsearch.eval.run_eval import flatten, judge_behavior
from imgsearch.eval.testset import TestQuery, load_labels, load_testset
from imgsearch.retrieval.engine import SearchEngine

TOP_K = 50
METRICS = ("p10", "p10_2", "p_r", "r50", "ndcg10", "mrr")


@dataclass
class Truth:
    gains: dict[str, float]  # ảnh đúng -> mức liên quan (> 0)
    by_intent: list[set[str]] = field(default_factory=list)


def build_truth(
    q: TestQuery, engine: SearchEngine, labels: dict[str, dict[str, int]]
) -> Truth | None:
    """None: truy vấn không có ground truth để tính điểm (phủ định, pool chưa chấm)."""

    def images(pids: list[str]) -> set[str]:
        return set(engine.store.images_of(pids)) - engine.unsafe_ids if pids else set()

    if q.gt == "rule":
        if q.intent_persons:
            by_intent = [images(p) for p in q.intent_persons]
            return Truth({i: 1.0 for s in by_intent for i in s}, by_intent)
        return Truth({i: 1.0 for i in images(q.expected_persons)}) if q.expected_persons else None
    if q.gt == "pool":
        refs = q.intent_refs or [q.id]
        if not all(labels.get(r) for r in refs):
            return None
        gains: dict[str, float] = {}
        for r in refs:
            for i, g in labels[r].items():
                if g > 0:
                    gains[i] = max(gains.get(i, 0.0), float(g))
        by_intent = [{i for i, g in labels[r].items() if g > 0} for r in q.intent_refs]
        return Truth(gains, by_intent)
    if isinstance(q.gt, list) and q.gt:
        return Truth({i: 1.0 for i in q.gt if i not in engine.unsafe_ids})
    return None


def parse_info(engine: SearchEngine, q: TestQuery) -> dict:
    p = engine.parser.parse(q.query)
    return {
        "type": str(p.type),
        "n_intents": len(p.intents),
        "persons": sorted({x for i in p.intents for x in i.person_ids}),
        "anchors": sorted(p.anchor_person_ids),
    }


def score(q: TestQuery, resp: SearchResponse, truth: Truth | None, parsed: dict) -> dict:
    ranked = flatten(resp, short=None)
    ok, why = judge_behavior(q.expected_behavior, resp, ranked)
    out = {
        "id": q.id,
        "type": q.type,
        "scenario": q.scenario,
        "split": q.split,
        "query": q.query,
        "expected": q.expected_behavior,
        "behavior_ok": ok,
        "behavior": why,
        "n_hits": len(ranked),
        "groups": [f"[{g.kind}] {g.label} ({g.total})" for g in resp.groups],
        "messages": resp.messages,
        "parsed": parsed,
    }
    if truth is not None and truth.gains:
        rel = set(truth.gains)
        top50 = sum(1 for i in ranked[:TOP_K] if i in rel)
        out.update(
            n_rel=len(rel),
            p10=precision_at_k(ranked, rel, 10),
            p_r=precision_at_k(ranked, rel, min(len(rel), 10)),
            r50=top50 / min(len(rel), TOP_K),
            ndcg10=ndcg_at_k(ranked, truth.gains, 10),
            mrr=reciprocal_rank(ranked, rel),
        )
        if max(truth.gains.values()) >= 2:  # chấm tay 0/1/2: P@10 chỉ tính ảnh nhãn 2
            out["p10_2"] = precision_at_k(ranked, {i for i, g in truth.gains.items() if g >= 2}, 10)
        intents = {k: s for k, s in enumerate(truth.by_intent) if s}
        if len(intents) >= 2:
            out["intent_cov10"] = intent_coverage_at_k(ranked, intents, 10)
            out["min_share10"] = min_intent_share_at_k(ranked, intents, 10)
    elif truth is not None:
        out["n_rel"] = 0  # đã chấm nhưng không ảnh nào liên quan: không tính điểm xếp hạng
    expected = set(q.expected_persons)
    if q.type in ("single", "multi_intent") and expected and parsed["persons"]:
        out["link_ok"] = set(parsed["persons"]) <= expected
    elif q.type in ("single", "multi_intent") and expected:
        out["link_ok"] = False
    if q.type == "multi_hop":
        targets = {p for g in resp.groups if g.kind == "relation_target" for p in g.person_ids}
        out["hallucinated"] = q.expected_behavior == "no_evidence" and bool(ranked)
        if q.anchor_persons:
            out["anchor_ok"] = set(parsed["anchors"]) == set(q.anchor_persons)
        if q.expected_behavior == "results":
            out["target_ok"] = targets == expected
            out["targets"] = sorted(targets)
    out["error"] = classify(q, out)
    return out


def classify(q: TestQuery, r: dict) -> str | None:
    """Loại lỗi (PLAN 5.3), lấy loại đầu tiên khớp."""
    if r.get("hallucinated"):
        return "bịa: trả ảnh khi không có bằng chứng"
    if r.get("anchor_ok") is False:
        return "liên kết người gốc sai"
    if r.get("link_ok") is False:
        return "liên kết người sai"
    if q.type == "multi_intent" and (
        r["parsed"]["type"] != "multi_intent" or r["parsed"]["n_intents"] != len(q.intents)
    ):
        return "tách ý định sai"
    if r.get("target_ok") is False:
        return "thiếu cạnh KG" if not r.get("targets") else "cạnh / người đích sai"
    if not r["behavior_ok"]:
        return "hành vi sai"
    if r.get("p_r") is not None and r["p_r"] < 0.5:
        return "xếp hạng kém (P@R < 0,5)"
    return None


def evaluate(
    engine: SearchEngine, queries: list[TestQuery], labels: dict[str, dict[str, int]]
) -> list[dict]:
    results = []
    for q in queries:
        truth = build_truth(q, engine, labels)
        parsed = parse_info(engine, q)
        t = time.perf_counter()
        resp, _ = engine.search(SearchRequest(query=q.query, top_k=TOP_K))
        r = score(q, resp, truth, parsed)
        r["ms"] = (time.perf_counter() - t) * 1000
        results.append(r)
    return results


def _mean(rs: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rs if r.get(key) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _rate(rs: list[dict], key: str) -> str:
    vals = [r[key] for r in rs if key in r]
    return f"{sum(vals)}/{len(vals)}" if vals else "—"


def aggregate(rs: list[dict]) -> dict:
    out = {
        "n": len(rs),
        "behavior_ok": sum(r["behavior_ok"] for r in rs),
        "n_scored": sum(1 for r in rs if "p10" in r),
    }
    for k in (*METRICS, "intent_cov10", "min_share10"):
        out[k] = _mean(rs, k)
    for k in ("link_ok", "anchor_ok", "target_ok", "hallucinated"):
        out[k] = _rate(rs, k)
    return out


def summarize(results: list[dict]) -> dict:
    by_type, by_scen = defaultdict(list), defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
        by_scen[(r["type"], r["scenario"])].append(r)
    return {
        "all": aggregate(results),
        "by_type": {t: aggregate(rs) for t, rs in sorted(by_type.items())},
        "by_scenario": {f"{t}/{s}": aggregate(rs) for (t, s), rs in sorted(by_scen.items())},
    }


# --- báo cáo ---


def _f(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _row(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['n']} | {s['behavior_ok']}/{s['n']} | {s['n_scored']} | "
        + " | ".join(_f(s[k]) for k in METRICS)
        + " |"
    )


HEAD = (
    "| Nhóm | Số truy vấn | Hành vi đúng | Có điểm | P@10 | P@10 (nhãn 2) | P@R | R@50 "
    "| nDCG@10 | MRR |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def render(meta: dict, runs: dict[str, list[dict]], main_cfg: str) -> str:
    results = runs[main_cfg]
    summ = summarize(results)
    lines = [
        f"# Đánh giá trên test set `{meta['testset']}` — split `{meta['split']}`",
        "",
        f"Index `{meta['index_version']}` ({meta['n_images']} ảnh, {meta['n_persons']} person), "
        f"cấu hình chính `{main_cfg}`, chạy lúc {meta['when']}. Số liệu trong file này là "
        "**[Đã đo]** trên đúng split ghi ở tiêu đề.",
        "",
        f"Truy vấn cần chấm tay: {meta['pool_judged']}/{meta['pool_total']} đã có nhãn; truy vấn "
        "chưa chấm chỉ được chấm hành vi.",
        "",
        "## Tổng hợp theo loại truy vấn",
        "",
        HEAD,
        _row("**tất cả**", summ["all"]),
    ]
    lines += [_row(t, s) for t, s in summ["by_type"].items()]
    lines += ["", "## Theo mã kịch bản", "", HEAD]
    lines += [_row(k, s) for k, s in summ["by_scenario"].items()]

    mi = summ["by_type"].get("multi_intent")
    if mi:
        lines += [
            "",
            "## Multi-intent",
            "",
            f"Độ phủ ý định@10: {_f(mi['intent_cov10'])}; tỷ lệ nhỏ nhất của một ý định trong "
            f"top 10: {_f(mi['min_share10'])}; tách ý định / liên kết người đúng: "
            f"{mi['link_ok']}.",
        ]
    mh = summ["by_type"].get("multi_hop")
    if mh:
        neg = [r for r in results if r["type"] == "multi_hop" and r["expected"] != "results"]
        refused = sum(r["behavior_ok"] for r in neg)
        lines += [
            "",
            "## Multi-hop",
            "",
            f"Liên kết người gốc đúng: {mh['anchor_ok']}; người đích đúng (truy vấn có đáp án): "
            f"{mh['target_ok']}; từ chối đúng ở truy vấn phủ định: {refused}/{len(neg)}; "
            f"bịa (trả ảnh khi không có bằng chứng): {mh['hallucinated']}.",
        ]

    if len(runs) > 1:
        lines += [
            "",
            "## So sánh cấu hình (cùng split)",
            "",
            "`bm25`: chỉ BM25; `w2`: BM25 + dense ảnh SigLIP2, trọng số RRF 2 (mặc định); "
            "`w8`: dense gần như át BM25; `w2+t0.5`: thêm dense văn bản qwen3 (mentor) "
            "trọng số 0,5. "
            "Chỉ khác ở truy vấn có phần mô tả.",
            "",
            "| Cấu hình | Loại | Có điểm | P@10 | P@10 (nhãn 2) | P@R | R@50 | nDCG@10 | MRR |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for cfg, rs in runs.items():
            ctx = [r for r in rs if r["id"].startswith(("S-C", "M-C-"))]
            for name, group in (("tất cả", rs), ("bối cảnh (chấm tay)", ctx)):
                s = aggregate(group)
                lines.append(
                    f"| {cfg} | {name} | {s['n_scored']} | "
                    + " | ".join(_f(s[k]) for k in METRICS)
                    + " |"
                )

    errors = Counter(r["error"] for r in results if r["error"])
    lines += ["", "## Phân loại lỗi (PLAN 5.3)", "", "| Loại lỗi | Số truy vấn | Ví dụ |"]
    lines += ["|---|---|---|"]
    for err, n in errors.most_common():
        ex = [f"{r['id']} “{r['query']}”" for r in results if r["error"] == err][:4]
        lines.append(f"| {err} | {n} | {'; '.join(ex)} |")

    lat = sorted(r["ms"] for r in results)
    lines += [
        "",
        f"Latency engine (không HTTP, một luồng): p50 {statistics.median(lat):.1f} ms, "
        f"p95 {lat[int(0.95 * (len(lat) - 1))]:.1f} ms, max {lat[-1]:.1f} ms. Đo tải: "
        "reports/perf.md.",
        "",
        "## Từng truy vấn",
        "",
        "| id | KB | Truy vấn | Mong đợi | Đạt | Số ảnh đúng | P@10 | P@R | R@50 | nDCG@10 "
        "| Lỗi | Hệ thống trả |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        what = "; ".join(r["groups"][:3] + r["messages"][:2]) or "(rỗng)"
        lines.append(
            f"| {r['id']} | {r['scenario']} | {r['query']} | {r['expected']} | "
            f"{'✓' if r['behavior_ok'] else '✗'} | {r.get('n_rel', '—')} | {_f(r.get('p10'))} | "
            f"{_f(r.get('p_r'))} | {_f(r.get('r50'))} | {_f(r.get('ndcg10'))} | "
            f"{r['error'] or ''} | {what.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--testset", type=Path, default=Path("testsets/queries.v1.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("testsets/labels.v1.jsonl"))
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument(
        "--configs",
        default=None,
        help=f"trong {list(CONFIGS)}, cách nhau dấu phẩy; cái đầu là cấu hình chính. Mặc định: "
        "mọi cấu hình engine chạy được, chính = w2+t0.5 nếu có kênh văn bản, không thì w2",
    )
    ap.add_argument("--out", type=Path, default=None, help="mặc định ../reports/eval_<split>.md")
    args = ap.parse_args(argv)

    engine = SearchEngine(eval_settings(index_dir=args.index_dir, index_version=args.index_version))
    engine.load()
    if engine.store is None:
        raise SystemExit(f"Không thấy index {args.index_dir}/v{args.index_version}")
    queries = [q for q in load_testset(args.testset) if args.split in ("all", q.split)]
    labels = load_labels(args.labels)
    pool = [q for q in queries if q.gt == "pool"]
    ok = available(engine)
    if args.configs:
        configs = [c for c in args.configs.split(",") if c in ok]
    else:
        main_cfg = next(c for c in ("w2+t0.5", "w2", "bm25") if c in ok)
        configs = [main_cfg, *(c for c in ok if c != main_cfg)]
    runs = {}
    for cfg in configs:
        with config(engine, cfg):
            runs[cfg] = evaluate(engine, queries, labels)
    meta = {
        "testset": args.testset.name,
        "split": args.split,
        "index_version": args.index_version,
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "n_images": engine.n_images,
        "n_persons": len(engine.parser.aliases),
        "pool_total": len(pool),
        "pool_judged": sum(1 for q in pool if build_truth(q, engine, labels) is not None),
        "configs": configs,
    }
    out = args.out or Path("../reports") / f"eval_{args.split}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(meta, runs, configs[0]), encoding="utf-8")
    payload = {
        "meta": meta,
        "summary": {c: summarize(rs) for c, rs in runs.items()},
        "queries": runs[configs[0]],
    }
    out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for c, rs in runs.items():
        s = summarize(rs)["all"]
        print(c, {k: s[k] for k in ("n", "behavior_ok", "n_scored", *METRICS)})
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
