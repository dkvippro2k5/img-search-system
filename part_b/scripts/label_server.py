"""Trang chấm nhãn 0/1/2 cho file pool (scripts/make_pool.py). Chỉ nghe 127.0.0.1.

python scripts/label_server.py          # mở http://127.0.0.1:8765

Mỗi lần bấm là ghi thêm một dòng vào labels.v1.jsonl (dòng sau ghi đè dòng trước), tắt mở lại
không mất gì. Ảnh hiển thị theo thứ tự xáo cố định, không cho biết cấu hình nào trả ảnh nào.
Ảnh đọc từ thư mục ảnh đã tải (data/fetch.py); không có thì chuyển hướng tới image_url.
"""

import argparse
import hashlib
import json
import mimetypes
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from imgsearch.eval.testset import append_labels, load_labels

SHA = re.compile(r"^[0-9a-f]{64}$")


def load_pool(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        key = lambda i, q=r["qid"]: hashlib.md5(f"{q}:{i['image_id']}".encode()).hexdigest()  # noqa: E731
        r["images"] = [
            {k: i.get(k) for k in ("image_id", "description", "persons")}
            for i in sorted(r["images"], key=key)
        ]
    return rows


def image_urls(path: Path) -> dict[str, str]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {i["image_id"]: i.get("image_url") for r in rows for i in r["images"]}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, pool: Path, labels: Path, images: Path, **kw):
        self.pool_path, self.labels_path, self.images_dir = pool, labels, images
        super().__init__(*a, **kw)

    def log_message(self, *_):  # yên lặng
        pass

    def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            self._json({"pool": load_pool(self.pool_path), "labels": load_labels(self.labels_path)})
        elif self.path.startswith("/img/"):
            self._image(self.path[5:])
        else:
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def _image(self, sha: str) -> None:
        if not SHA.match(sha):
            return self._send(b"bad id", "text/plain", HTTPStatus.BAD_REQUEST)
        found = next((self.images_dir / sha[:2]).glob(f"{sha}.*"), None)
        if found is None:
            url = image_urls(self.pool_path).get(sha)
            if not url or not url.startswith("http"):
                return self._send(b"missing", "text/plain", HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", url)
            return self.end_headers()
        ctype = mimetypes.guess_type(found.name)[0] or "image/jpeg"
        self._send(found.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/labels":
            return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        rows = [r for r in body["rows"] if SHA.match(r["image_id"])]
        append_labels(self.labels_path, rows)
        self._json({"saved": len(rows)})


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Chấm nhãn test set</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#ddd;
  --g0:#c94a3a; --g1:#d9a400; --g2:#2f8f4e; --focus:#2463eb; }
* { box-sizing:border-box }
body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--ink);
  display:grid; grid-template-columns:260px 1fr; height:100vh }
nav { overflow:auto; border-right:1px solid var(--line); background:var(--card) }
nav div { padding:7px 12px; cursor:pointer; border-bottom:1px solid #eee }
nav div.on { background:#e8eefc } nav small { color:var(--muted); float:right }
nav div.done small { color:var(--g2); font-weight:600 }
main { overflow:auto; padding:12px 16px 80px }
h1 { font-size:20px; margin:4px 0 } .guide { color:var(--muted); margin:4px 0 10px }
.bar { position:sticky; top:-12px; background:var(--bg); padding:8px 0; z-index:2;
  display:flex; gap:8px; align-items:center; flex-wrap:wrap }
button { font:inherit; padding:6px 12px; border:1px solid var(--line); background:var(--card);
  border-radius:6px; cursor:pointer } button.primary { background:var(--focus); color:#fff;
  border-color:var(--focus) }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:10px }
.card { background:var(--card); border:3px solid var(--line); border-radius:8px; overflow:hidden;
  cursor:pointer } .card.focus { outline:3px solid var(--focus); outline-offset:1px }
.card.g0 { border-color:var(--g0) } .card.g1 { border-color:var(--g1) }
.card.g2 { border-color:var(--g2) }
.card img { width:100%; height:220px; object-fit:contain; background:#222; display:block }
.card .t { padding:5px 7px; font-size:12px; color:var(--muted); height:4.6em; overflow:hidden }
.card .p { padding:0 7px 4px; font-size:12px; font-weight:600 }
.card .b { display:flex } .card .b span { flex:1; text-align:center; padding:4px 0;
  border-top:1px solid #eee; font-weight:600 }
.card.g0 .b span:nth-child(1) { background:var(--g0); color:#fff }
.card.g1 .b span:nth-child(2) { background:var(--g1); color:#fff }
.card.g2 .b span:nth-child(3) { background:var(--g2); color:#fff }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; background:#fff }
</style></head><body>
<nav id="list"></nav>
<main>
  <div class="bar">
    <h1 id="q"></h1>
    <span id="prog"></span>
    <button id="rest0" class="primary">Còn lại = 0, sang truy vấn sau</button>
  </div>
  <p class="guide"><b>2</b>: khớp rõ mọi điều kiện (đúng người nếu truy vấn có tên + đúng
  hoạt động/bối cảnh) · <b>1</b>: khớp một phần (đúng chủ đề nhưng không rõ, hoặc thiếu một
  điều kiện) · <b>0</b>: không khớp. Có tên người mà ảnh không phải người đó (xem dòng tên) = 0.
  Chấm theo nội dung ảnh; tiêu đề chỉ để tham khảo khi truy vấn nhắc sự kiện cụ thể.<br>
  Phím: <kbd>0</kbd><kbd>1</kbd><kbd>2</kbd> chấm ô đang chọn và sang ô sau ·
  <kbd>←</kbd><kbd>→</kbd> di chuyển · bấm vào ô để chọn · <kbd>Enter</kbd> = nút xanh.</p>
  <div class="grid" id="grid"></div>
</main>
<script>
let pool = [], labels = {}, cur = 0, focus = 0;
const $ = (id) => document.getElementById(id);
async function load() {
  const d = await (await fetch("/api/data")).json();
  pool = d.pool; labels = d.labels;
  const firstTodo = pool.findIndex((q) => done(q) < q.images.length);
  show(firstTodo < 0 ? 0 : firstTodo);
}
const done = (q) => q.images.filter((i) => (labels[q.qid] || {})[i.image_id] !== undefined).length;
function renderList() {
  $("list").innerHTML = pool.map((q, k) => {
    const n = done(q), all = q.images.length;
    return `<div class="${k === cur ? "on" : ""} ${n === all ? "done" : ""}" data-k="${k}">
      ${q.qid} <small>${n}/${all}</small><br>${esc(q.query)}</div>`;
  }).join("");
  $("list").querySelectorAll("div").forEach((el) => (el.onclick = () => show(+el.dataset.k)));
  const total = pool.reduce((s, q) => s + q.images.length, 0);
  const got = pool.reduce((s, q) => s + done(q), 0);
  $("prog").textContent = `tổng ${got}/${total} ảnh`;
}
function show(k) {
  cur = k; focus = 0; const q = pool[k];
  $("q").textContent = `${q.qid} · ${q.query}`;
  $("grid").innerHTML = q.images.map((im, j) => `
    <div class="card" data-j="${j}">
      <img loading="lazy" src="/img/${im.image_id}" alt="">
      <div class="p">${esc((im.persons || []).join(", "))}</div>
      <div class="t">${esc(im.description || "")}</div>
      <div class="b"><span>0</span><span>1</span><span>2</span></div>
    </div>`).join("");
  $("grid").querySelectorAll(".card").forEach((el) => {
    el.onclick = (e) => {
      focus = +el.dataset.j;
      const s = e.target.closest(".b span");
      if (s) grade([...s.parentNode.children].indexOf(s), false); else paint();
    };
  });
  paint(); renderList(); $("grid").parentNode.scrollTop = 0;
}
function paint() {
  const q = pool[cur], lab = labels[q.qid] || {};
  $("grid").querySelectorAll(".card").forEach((el, j) => {
    const g = lab[q.images[j].image_id];
    el.className = "card" + (g === undefined ? "" : " g" + g) + (j === focus ? " focus" : "");
  });
  const f = $("grid").children[focus];
  if (f) f.scrollIntoView({ block: "nearest" });
}
async function save(rows) {
  for (const r of rows) (labels[r.qid] = labels[r.qid] || {})[r.image_id] = r.grade;
  const res = await fetch("/api/labels", { method: "POST", body: JSON.stringify({ rows }) });
  if (!res.ok) alert("Lưu lỗi: " + res.status);
}
async function grade(g, advance = true) {
  const q = pool[cur], im = q.images[focus];
  await save([{ qid: q.qid, image_id: im.image_id, grade: g }]);
  if (advance && focus < q.images.length - 1) focus++;
  paint(); renderList();
}
$("rest0").onclick = async () => {
  const q = pool[cur], lab = labels[q.qid] || {};
  const rows = q.images.filter((i) => lab[i.image_id] === undefined)
    .map((i) => ({ qid: q.qid, image_id: i.image_id, grade: 0 }));
  if (rows.length) await save(rows);
  if (cur < pool.length - 1) show(cur + 1); else renderList();
};
document.addEventListener("keydown", (e) => {
  if (["0", "1", "2"].includes(e.key)) grade(+e.key);
  else if (e.key === "ArrowRight") {
    focus = Math.min(focus + 1, pool[cur].images.length - 1); paint();
  }
  else if (e.key === "ArrowLeft") { focus = Math.max(focus - 1, 0); paint(); }
  else if (e.key === "Enter") $("rest0").click();
  else return;
  e.preventDefault();
});
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);
load();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=Path, default=Path("testsets/pool.v1.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("testsets/labels.v1.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("../data/raw/fetched/images"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not args.pool.exists():
        raise SystemExit(f"Chưa có {args.pool}: chạy scripts/make_pool.py trước")
    handler = partial(Handler, pool=args.pool, labels=args.labels, images=args.images)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Mở http://127.0.0.1:{args.port}  (Ctrl+C để dừng; nhãn ghi vào {args.labels})")
    server.serve_forever()


if __name__ == "__main__":
    main()
