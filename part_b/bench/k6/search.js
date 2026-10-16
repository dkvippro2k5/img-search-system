// Load test /search: mức đồng thời 1, 4, 8, 16 VU, mỗi mức 60 giây (PLAN.md mục 6).
// Truy vấn trộn 50% single, 30% multi-intent, 20% multi-hop lấy từ testsets/bench_queries.json.
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

const queries = new SharedArray("queries", () => JSON.parse(open("/testsets/bench_queries.json")));

const stage = (vus, start) => ({
  executor: "constant-vus",
  vus,
  duration: "60s",
  startTime: start,
  tags: { vus: String(vus) },
});

export const options = {
  scenarios: {
    warmup: { executor: "shared-iterations", vus: 1, iterations: 50, tags: { phase: "warmup" } },
    vu1: stage(1, "20s"),
    vu4: stage(4, "85s"),
    vu8: stage(8, "150s"),
    vu16: stage(16, "215s"),
  },
  thresholds: {
    "http_req_duration{phase:!warmup}": ["p(95)<3000"],
  },
  summaryTrendStats: ["avg", "p(50)", "p(95)", "p(99)", "max"],
};

export default function () {
  const q = queries[Math.floor(Math.random() * queries.length)];
  const res = http.post(`${BASE_URL}/search`, JSON.stringify({ query: q.query }), {
    headers: { "Content-Type": "application/json" },
    tags: { qtype: q.type },
  });
  check(res, { "status 200": (r) => r.status === 200 });
}
