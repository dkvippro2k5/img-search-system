// Giao diện tối thiểu: gửi truy vấn, hiển thị kết quả theo nhóm, chuỗi bằng chứng multi-hop,
// thông báo và gợi ý. Mọi chữ từ server đi qua textContent, không chèn HTML (L9).

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

const KIND_LABELS: Record<GroupKind, string> = {
  intent: "",
  person: "Người trùng tên",
  co_occurrence: "Cùng xuất hiện",
  relation_target: "Theo quan hệ",
};

const form = document.getElementById("search-form") as HTMLFormElement;
const input = document.getElementById("q") as HTMLInputElement;
const statusEl = document.getElementById("status") as HTMLElement;
const messagesEl = document.getElementById("messages") as HTMLElement;
const reasoningEl = document.getElementById("reasoning") as HTMLElement;
const resultsEl = document.getElementById("results") as HTMLElement;

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

function renderMessages(res: SearchResponse): void {
  for (const m of res.messages) messagesEl.append(el("p", m, "message"));
  if (res.suggestions.length) {
    const p = el("p", "Bạn có muốn tìm: ", "message");
    res.suggestions.forEach((s, i) => {
      const b = el("button", s, "suggestion");
      b.type = "button";
      b.onclick = () => { input.value = s; void search(s); };
      if (i) p.append(", ");
      p.append(b);
    });
    messagesEl.append(p);
  }
}

function renderReasoning(res: SearchResponse): void {
  if (!res.reasoning.length) return;
  reasoningEl.append(el("h2", "Bằng chứng quan hệ"));
  for (const step of res.reasoning) {
    const p = el("p", `${step.obj} là ${step.predicate_label} của ${step.subj} — “${step.evidence}” `);
    p.append(link(step.evidence_url, `(${step.source})`));
    reasoningEl.append(p);
  }
}

function renderHit(hit: ImageHit): HTMLElement {
  const fig = el("figure", undefined, "hit");
  const img = el("img");
  img.src = hit.thumb_url;
  img.alt = hit.description ?? "";
  img.loading = "lazy";
  img.onerror = () => { img.removeAttribute("src"); }; // ảnh lỗi -> ô trống, không vỡ trang (R4)
  fig.append(img);
  // Tiêu đề bài báo nguồn để người dùng tự đánh giá ngữ cảnh (Q5).
  const cap = el("figcaption");
  cap.append(link(hit.source_url, hit.description ?? "Nguồn"));
  fig.append(cap);
  return fig;
}

function render(res: SearchResponse, serverMs: string | null): void {
  messagesEl.replaceChildren();
  reasoningEl.replaceChildren();
  resultsEl.replaceChildren();

  statusEl.textContent = [`Loại: ${res.query_type}`, serverMs && `server ${serverMs} ms`]
    .filter(Boolean)
    .join(" · ");
  renderMessages(res);
  renderReasoning(res);

  for (const group of res.groups) {
    const section = el("section", undefined, "group");
    const kind = KIND_LABELS[group.kind];
    const heading = el("h2", `${group.label} (${group.total})`);
    if (kind) heading.prepend(el("span", kind, "kind"));
    section.append(heading);
    const grid = el("div", undefined, "grid");
    for (const hit of group.hits) grid.append(renderHit(hit));
    section.append(grid);
    resultsEl.append(section);
  }
}

async function search(query: string): Promise<void> {
  statusEl.textContent = "Đang tìm…";
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
      statusEl.textContent = resp.status === 422 ? "Truy vấn không hợp lệ (rỗng hoặc quá dài)" : `Lỗi ${resp.status}`;
      return;
    }
    const total = resp.headers.get("Server-Timing")?.match(/total;dur=([\d.]+)/)?.[1] ?? null;
    render((await resp.json()) as SearchResponse, total);
  } catch {
    statusEl.textContent = "Không kết nối được máy chủ";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (query) void search(query);
});
