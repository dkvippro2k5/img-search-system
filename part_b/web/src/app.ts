// Giao diện tìm ảnh: truy vấn mẫu, kết quả theo nhóm, chuỗi bằng chứng multi-hop, thời gian từng
// bước (Server-Timing), xem ảnh lớn. Mọi chữ từ server đi qua textContent, không chèn HTML (L9).

interface ImageHit {
  image_id: string;
  thumb_url: string;
  score: number;
  person_ids: string[];
  description?: string | null;
  source_url?: string | null;
}

type GroupKind = "intent" | "person" | "co_occurrence" | "relation_target";

interface ResultGroup {
  label: string;
  kind: GroupKind;
  person_ids: string[];
  hits: ImageHit[];
  total: number;
}

interface EvidenceStep {
  subj: string;
  subj_id: string;
  predicate: string;
  predicate_label: string;
  obj: string;
  obj_id: string;
  source: string;
  evidence: string;
  evidence_url?: string | null;
}

interface SearchResponse {
  query: string;
  query_type: string;
  groups: ResultGroup[];
  reasoning: EvidenceStep[];
  messages: string[];
  suggestions: string[];
}

interface Health {
  status: string;
  n_images?: number;
  n_persons?: number;
  dense?: string | null;
  text_dense?: string | null;
}

const KIND_LABELS: Record<GroupKind, string> = {
  intent: "",
  person: "Người trùng tên",
  co_occurrence: "Cùng xuất hiện",
  relation_target: "Theo quan hệ",
};

const QUERY_TYPES: Record<string, string> = {
  single: "Một ý định",
  multi_intent: "Nhiều ý định",
  multi_hop: "Quan hệ (multi-hop)",
};

// Truy vấn mẫu cho demo, theo mã kịch bản (capstone/SCENARIOS.md).
const EXAMPLES: [string, string[]][] = [
  ["Một ý định", ["Đen Vâu hát trong concert", "Đặng Văn Lâm bắt bóng", "hoa hậu đội vương miện", "vận động viên nhận huy chương"]],
  ["Không dấu, trùng tên", ["hoa hau doi vuong mien", "Quang Hải", "Trần Sỹ Thanh"]],
  ["Nhiều ý định", ["Trấn Thành và Hari Won", "Đen Vâu và Hồ Ngọc Hà"]],
  ["Quan hệ", ["con trai Phạm Nhật Vượng", "chồng của Hari Won", "con của Trấn Thành"]],
];

const STEP_LABELS: Record<string, string> = {
  queue: "chờ",
  parse: "phân tích",
  retrieve: "truy xuất",
  hydrate: "lấy dữ liệu",
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("search-form");
const input = $<HTMLInputElement>("q");
const safe = $<HTMLInputElement>("safe");
const statusEl = $("status");
const summaryEl = $("summary");
const messagesEl = $("messages");
const reasoningEl = $("reasoning");
const resultsEl = $("results");
const viewer = $<HTMLDialogElement>("viewer");

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, cls?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function link(href: string | null | undefined, text: string): HTMLElement {
  if (!href || !/^https?:\/\//.test(href)) return el("span", text);
  const a = el("a", text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

function fmt(n: number): string {
  return n.toLocaleString("vi-VN");
}

// --- đầu trang ---

async function loadInfo(): Promise<void> {
  const info = $("index-info");
  try {
    const h = (await (await fetch("/healthz")).json()) as Health;
    if (h.status !== "ok") {
      info.textContent = "Máy chủ đang nạp index…";
      setTimeout(loadInfo, 2000);
      return;
    }
    const channels = ["BM25", h.dense && `ảnh ${h.dense}`, h.text_dense && `văn bản ${h.text_dense}`];
    info.textContent = `${fmt(h.n_images ?? 0)} ảnh · ${fmt(h.n_persons ?? 0)} người · kênh: ${channels.filter(Boolean).join(", ")}`;
  } catch {
    info.textContent = "Không kết nối được máy chủ";
  }
}

function renderExamples(): void {
  const nav = $("examples");
  for (const [title, queries] of EXAMPLES) {
    const g = el("div", undefined, "ex-group");
    g.append(el("span", title));
    for (const q of queries) {
      const b = el("button", q, "chip");
      b.type = "button";
      b.onclick = () => run(q);
      g.append(b);
    }
    nav.append(g);
  }
}

// --- kết quả ---

function parseTiming(header: string | null): [string, number][] {
  if (!header) return [];
  return header.split(",").flatMap((part) => {
    const m = part.trim().match(/^([\w-]+);dur=([\d.]+)/);
    return m ? [[m[1], Number(m[2])] as [string, number]] : [];
  });
}

function renderSummary(res: SearchResponse, timing: [string, number][]): void {
  summaryEl.hidden = false;
  $("qtype").textContent = QUERY_TYPES[res.query_type] ?? res.query_type;
  const total = timing.find(([k]) => k === "total")?.[1];
  const nImages = res.groups.reduce((s, g) => s + g.total, 0);
  $("total-ms").textContent = [
    `${res.groups.length} nhóm`,
    `${fmt(nImages)} ảnh khớp`,
    total !== undefined && `máy chủ ${total.toFixed(0)} ms`,
  ].filter(Boolean).join(" · ");

  const bar = $("timing");
  bar.replaceChildren();
  const steps = timing.filter(([k]) => k !== "total");
  const sum = steps.reduce((s, [, v]) => s + v, 0) || 1;
  const legend = el("div", undefined, "timing-legend");
  for (const [name, ms] of steps) {
    const color = `var(--t-${name in STEP_LABELS ? name : "other"})`;
    const seg = el("span");
    seg.style.width = `${(100 * ms) / sum}%`;
    seg.style.background = color;
    seg.title = `${STEP_LABELS[name] ?? name}: ${ms.toFixed(1)} ms`;
    bar.append(seg);
    const item = el("span", `${STEP_LABELS[name] ?? name} ${ms.toFixed(1)} ms`);
    const dot = el("i");
    dot.style.background = color;
    item.prepend(dot);
    legend.append(item);
  }
  bar.after(legend);

  const nav = $("group-nav");
  nav.replaceChildren();
  if (res.groups.length > 1) {
    res.groups.forEach((g, i) => {
      const b = el("button", `${g.label} (${fmt(g.total)})`, "chip");
      b.type = "button";
      b.onclick = () => document.getElementById(`group-${i}`)?.scrollIntoView({ behavior: "smooth" });
      nav.append(b);
    });
  }
}

function renderMessages(res: SearchResponse): void {
  for (const m of res.messages) messagesEl.append(el("p", m, "message"));
  if (res.suggestions.length) {
    const p = el("p", "Bạn có muốn tìm: ", "message");
    res.suggestions.forEach((s, i) => {
      const b = el("button", s, "suggestion");
      b.type = "button";
      b.onclick = () => run(s);
      if (i) p.append(", ");
      p.append(b);
    });
    messagesEl.append(p);
  }
}

function renderReasoning(res: SearchResponse): void {
  if (!res.reasoning.length) return;
  const box = el("div", undefined, "chain");
  box.append(el("h2", "Bằng chứng quan hệ"));
  // Một cạnh có thể có nhiều bằng chứng (nhiều tiêu đề): gộp theo cặp người + quan hệ.
  const edges = new Map<string, EvidenceStep[]>();
  for (const step of res.reasoning) {
    const key = `${step.subj_id}|${step.predicate}|${step.obj_id}`;
    edges.set(key, [...(edges.get(key) ?? []), step]);
  }
  for (const steps of edges.values()) {
    const [first] = steps;
    // cạnh lưu một chiều: obj là <predicate> của subj
    const edge = el("div", undefined, "edge");
    edge.append(el("span", first.subj, "node"), el("span", `— ${first.predicate_label} →`, "rel"), el("span", first.obj, "node"));
    if (steps.length > 1) edge.append(el("span", `${steps.length} bằng chứng`, "muted"));
    const list = el("ul", undefined, "evidence");
    for (const step of steps) {
      const li = el("li");
      li.append(el("q", step.evidence), " ", link(step.evidence_url, `(nguồn: ${step.source})`));
      list.append(li);
    }
    box.append(edge, list);
  }
  reasoningEl.append(box);
}

function brokenTile(frame: HTMLElement, hit: ImageHit): void {
  const b = el("div", undefined, "broken");
  b.append(el("div", "Không tải được ảnh gốc"));
  if (hit.source_url) b.append(link(hit.source_url, "Mở bài nguồn"));
  frame.replaceChildren(b);
}

function renderHit(hit: ImageHit, group: ResultGroup): HTMLElement {
  const fig = el("figure", undefined, "hit");
  const frame = el("div", undefined, "frame");
  const img = el("img");
  img.alt = hit.description ?? "";
  img.loading = "lazy";
  img.referrerPolicy = "no-referrer";
  let retried = false;
  // Ảnh hotlink có thể bị CDN chặn khi tải dồn: thử lại một lần, rồi mới báo hỏng (R4).
  img.onerror = () => {
    if (retried) return brokenTile(frame, hit);
    retried = true;
    setTimeout(() => { img.src = hit.thumb_url; }, 1500 + Math.random() * 1500);
  };
  img.src = hit.thumb_url;
  frame.append(img);
  fig.append(frame);
  // Tiêu đề bài báo nguồn để người dùng tự đánh giá ngữ cảnh (Q5).
  fig.append(el("figcaption", hit.description ?? "(không có tiêu đề)"));
  fig.onclick = (e) => {
    if ((e.target as HTMLElement).closest("a")) return;
    openViewer(hit, group);
  };
  return fig;
}

function openViewer(hit: ImageHit, group: ResultGroup): void {
  const img = $<HTMLImageElement>("viewer-img");
  img.src = hit.thumb_url;
  img.referrerPolicy = "no-referrer";
  img.alt = hit.description ?? "";
  const cap = $("viewer-caption");
  cap.replaceChildren(link(hit.source_url, hit.description ?? "Nguồn"));
  $("viewer-meta").textContent = [
    group.label,
    hit.person_ids.length ? `person: ${hit.person_ids.join(", ")}` : "",
    `điểm ${hit.score.toFixed(3)}`,
  ].filter(Boolean).join(" · ");
  viewer.showModal();
}

function render(res: SearchResponse, timing: [string, number][]): void {
  statusEl.textContent = "";
  messagesEl.replaceChildren();
  reasoningEl.replaceChildren();
  resultsEl.replaceChildren();
  document.querySelectorAll(".timing-legend").forEach((n) => n.remove());

  renderSummary(res, timing);
  renderMessages(res);
  renderReasoning(res);

  res.groups.forEach((group, i) => {
    const section = el("section", undefined, "group");
    section.id = `group-${i}`;
    const head = el("div", undefined, "group-head");
    const kind = KIND_LABELS[group.kind];
    if (kind) head.append(el("span", kind, `kind ${group.kind}`));
    head.append(el("h2", group.label));
    const shown = group.hits.length < group.total ? `${group.hits.length}/${fmt(group.total)} ảnh` : `${fmt(group.total)} ảnh`;
    head.append(el("span", shown, "muted"));
    section.append(head);
    const grid = el("div", undefined, "grid");
    for (const hit of group.hits) grid.append(renderHit(hit, group));
    section.append(grid);
    resultsEl.append(section);
  });
  if (!res.groups.length && !res.messages.length) messagesEl.append(el("p", "Không tìm thấy ảnh phù hợp.", "message"));
}

async function search(query: string): Promise<void> {
  statusEl.textContent = "Đang tìm…";
  summaryEl.hidden = true;
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, safe_search: safe.checked }),
    });
    if (!resp.ok) {
      statusEl.textContent = resp.status === 422 ? "Truy vấn không hợp lệ (rỗng hoặc quá dài)" : `Lỗi ${resp.status}`;
      return;
    }
    render((await resp.json()) as SearchResponse, parseTiming(resp.headers.get("Server-Timing")));
  } catch {
    statusEl.textContent = "Không kết nối được máy chủ";
  }
}

// Truy vấn nằm trên URL (#q=...) để tải lại trang hoặc gửi link vẫn ra cùng kết quả.
function run(query: string): void {
  input.value = query;
  const hash = `#q=${encodeURIComponent(query)}`;
  if (location.hash !== hash) history.pushState(null, "", hash);
  window.scrollTo({ top: 0 });
  void search(query);
}

function fromHash(): void {
  const q = new URLSearchParams(location.hash.slice(1)).get("q");
  if (q) {
    input.value = q;
    void search(q);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (query) run(query);
});
safe.addEventListener("change", () => {
  const query = input.value.trim();
  if (query) void search(query);
});
window.addEventListener("popstate", fromHash);
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== input && !viewer.open) {
    e.preventDefault();
    input.focus();
    input.select();
  }
});
viewer.addEventListener("click", (e) => { if (e.target === viewer) viewer.close(); });

renderExamples();
void loadInfo();
fromHash();
