// Giao diện tối thiểu: gửi truy vấn, hiển thị kết quả theo nhóm và chuỗi bằng chứng multi-hop.

interface ImageHit {
  image_id: string;
  thumb_url: string;
  score: number;
  description?: string;
  source_url?: string;
}

interface ResultGroup {
  label: string;
  hits: ImageHit[];
  total: number;
}

interface EvidenceStep {
  subj: string;
  predicate: string;
  obj: string;
  source: string;
  evidence: string;
  evidence_url?: string;
}

interface SearchResponse {
  query: string;
  query_type: string;
  groups: ResultGroup[];
  reasoning: EvidenceStep[];
  messages: string[];
  suggestions: string[];
}

const form = document.getElementById("search-form") as HTMLFormElement;
const input = document.getElementById("q") as HTMLInputElement;
const statusEl = document.getElementById("status") as HTMLElement;
const reasoningEl = document.getElementById("reasoning") as HTMLElement;
const resultsEl = document.getElementById("results") as HTMLElement;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text; // textContent: không chèn HTML (L9)
  return node;
}

function render(res: SearchResponse, serverMs: string | null): void {
  resultsEl.replaceChildren();
  reasoningEl.replaceChildren();

  const notes = [...res.messages, ...res.suggestions.map((s) => `Bạn có muốn tìm: ${s}?`)];
  statusEl.textContent = [`Loại: ${res.query_type}`, serverMs && `server ${serverMs} ms`, ...notes]
    .filter(Boolean)
    .join(" · ");

  for (const step of res.reasoning) {
    const p = el("p", `${step.subj} →(${step.predicate}, ${step.source})→ ${step.obj}: ${step.evidence}`);
    reasoningEl.append(p);
  }

  for (const group of res.groups) {
    const section = el("section");
    section.className = "group";
    section.append(el("h2", `${group.label} (${group.total})`));
    const grid = el("div");
    grid.className = "grid";
    for (const hit of group.hits) {
      const img = el("img");
      img.src = hit.thumb_url;
      img.alt = hit.description ?? "";
      img.title = hit.description ?? "";
      img.loading = "lazy";
      img.onerror = () => { img.removeAttribute("src"); }; // ảnh lỗi -> ô trống, không vỡ trang (R4)
      grid.append(img);
    }
    section.append(grid);
    resultsEl.append(section);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  statusEl.textContent = "Đang tìm…";
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
      statusEl.textContent = `Lỗi ${resp.status}`;
      return;
    }
    const total = resp.headers.get("Server-Timing")?.match(/total;dur=([\d.]+)/)?.[1] ?? null;
    render((await resp.json()) as SearchResponse, total);
  } catch {
    statusEl.textContent = "Không kết nối được máy chủ";
  }
});
