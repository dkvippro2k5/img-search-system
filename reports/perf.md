# Báo cáo hiệu năng

Sinh tự động bởi `part_b/bench/make_report.py` từ kết quả k6 (`bench/run_bench.py`). Mọi số trong file là **[Đã đo]** trên phần cứng ghi dưới đây.

## Phần cứng và cấu hình

- Máy: Intel(R) Core(TM) Ultra 7 155H, 22 luồng logic, RAM 31.4 GB, Windows 11 (10.0.26200)
- Docker 29.6.2 trên Docker Desktop (kernel 6.18.33.2-microsoft-standard-WSL2): VM có 22 CPU, 15.3 GB RAM
- Container api: giới hạn 4 CPU, 4 GB RAM (docker-compose.yml), ONNX Runtime 4 luồng, uvicorn 1 worker; k6 chạy trong container cùng máy (cạnh tranh CPU với api).

## Kịch bản

- `POST /search` qua HTTP, top_k mặc định (30 ảnh mỗi nhóm), lọc an toàn bật.
- 100 truy vấn lấy từ test set (`testsets/bench_queries.json`): 50% single, 30% multi-intent, 20% multi-hop; mỗi request chọn ngẫu nhiên một truy vấn.
- Làm nóng 50 request (không tính), rồi 1, 4, 8, 16 VU, mỗi mức 60 s, VU gửi liên tục không nghỉ. Phân vị tính trên mọi request của mức đó.
- `nocache`: tắt cache vector truy vấn dense (`QUERY_CACHE_SIZE=0`), mỗi request đều chạy text encoder ONNX: đây là số chính. `cache`: bật cache (truy vấn lặp lại). API không có cache kết quả.
- Hậu tố `-cN`: `SEARCH_CONCURRENCY=N` (số truy vấn được tính toán cùng lúc, còn lại xếp hàng; 0 = không giới hạn). Không hậu tố = mặc định của docker-compose (2).
- Không hậu tố: cấu hình mặc định, **có** kênh dense văn bản qwen3 qua API mentor (trọng số 0,5, mỗi truy vấn có phần mô tả gọi mạng một lần). `-notext`: tắt kênh này (`TEXT_DENSE_ENABLED=false`); các biến thể `-cN` cũng đo khi chưa có kênh văn bản.
- Kênh văn bản gọi API của mentor qua đường hầm: key bị giới hạn **100 request/phút** (header `x-litellm-key-rpm-limit`, đo 20/10), vượt thì HTTP 429 và engine bỏ kênh cho request đó (HTTP vẫn 200). Latency kênh này phụ thuộc tải máy mentor, nên hai lần đo liên tiếp có thể chênh nhau; dưới tải lớn, chất lượng thực tế gần với `-notext`.
- Server p50/p95: thời gian trong server theo header `Server-Timing` total (gồm thời gian chờ semaphore, không gồm mạng).

## Tóm tắt

- `nocache`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **20.1** (ở 8 VU) [Đã đo]. Thông lượng giữ ổn định khi tăng tải (19.5 QPS ở 16 VU): đã bão hòa CPU, thêm VU chỉ tăng thời gian xếp hàng.
- `nocache-c0`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **33.9** (ở 4 VU) [Đã đo]. Thông lượng giảm khi tăng tải (33.9 QPS ở 4 VU -> 16.6 ở 16 VU): các request tranh luồng CPU.
- `nocache-c1`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **25.3** (ở 1 VU) [Đã đo]. Thông lượng giữ ổn định khi tăng tải (25.2 QPS ở 16 VU): đã bão hòa CPU, thêm VU chỉ tăng thời gian xếp hàng.
- `nocache-notext`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **31.2** (ở 8 VU) [Đã đo]. Thông lượng giữ ổn định khi tăng tải (29.9 QPS ở 16 VU): đã bão hòa CPU, thêm VU chỉ tăng thời gian xếp hàng.
- `cache`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **86.4** (ở 16 VU) [Đã đo]. QPS còn tăng ở mức VU cao nhất đã thử: chưa chạm trần.
- `cache-c0`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **162.6** (ở 1 VU) [Đã đo]. Thông lượng giảm khi tăng tải (162.6 QPS ở 1 VU -> 28.6 ở 16 VU): các request tranh luồng CPU.
- `cache-notext`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: **165.1** (ở 1 VU) [Đã đo]. Thông lượng giữ ổn định khi tăng tải (156.6 QPS ở 16 VU): đã bão hòa CPU, thêm VU chỉ tăng thời gian xếp hàng.

## Kết quả

### Chế độ `nocache`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **10.8 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 704 | 11.7 | 0.00% | 4 | 332 | 435 | 613 | 2 | 327 |
| 4 | 1099 | 18.3 | 0.00% | 174 | 604 | 1572 | 2501 | 162 | 598 |
| 8 | 1208 | 20.1 | 0.00% | 319 | 991 | 1845 | 2756 | 314 | 987 |
| 16 | 1169 | 19.5 | 0.00% | 787 | 1572 | 2157 | 2856 | 768 | 1548 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 318 | 417 | 4 |
| 4 | 609 | 657 | 340 |
| 8 | 1075 | 1008 | 707 |
| 16 | 1604 | 1614 | 1398 |

Container api trong lúc chạy k6: CPU trung bình 344%, cao nhất 408% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 2202 MB [Đã đo].

### Chế độ `nocache-c0`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **8.8 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1563 | 26.1 | 0.00% | 3 | 149 | 258 | 281 | 2 | 144 |
| 4 | 2036 | 33.9 | 0.00% | 10 | 401 | 689 | 1001 | 5 | 398 |
| 8 | 1364 | 22.7 | 0.00% | 254 | 1007 | 1434 | 1916 | 232 | 1004 |
| 16 | 999 | 16.6 | 0.00% | 821 | 2295 | 3209 | 4303 | 800 | 2281 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 148 | 249 | 4 |
| 4 | 392 | 663 | 63 |
| 8 | 970 | 1341 | 402 |
| 16 | 2259 | 2740 | 1136 |

Container api trong lúc chạy k6: CPU trung bình 349%, cao nhất 413% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 1857 MB [Đã đo].

### Chế độ `nocache-c1`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **9.8 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1518 | 25.3 | 0.00% | 3 | 148 | 250 | 292 | 2 | 145 |
| 4 | 1490 | 24.8 | 0.00% | 124 | 381 | 497 | 699 | 120 | 378 |
| 8 | 1405 | 23.4 | 0.00% | 299 | 700 | 949 | 1216 | 297 | 695 |
| 16 | 1511 | 25.2 | 0.00% | 622 | 1137 | 1420 | 2059 | 616 | 1126 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 147 | 229 | 4 |
| 4 | 388 | 381 | 332 |
| 8 | 720 | 703 | 610 |
| 16 | 1140 | 1151 | 1071 |

Container api trong lúc chạy k6: CPU trung bình 345%, cao nhất 406% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 1834 MB [Đã đo].

### Chế độ `nocache-notext`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **9.3 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1403 | 23.4 | 0.00% | 4 | 159 | 268 | 373 | 2 | 156 |
| 4 | 1770 | 29.5 | 0.00% | 113 | 365 | 531 | 788 | 109 | 362 |
| 8 | 1873 | 31.2 | 0.00% | 214 | 568 | 702 | 928 | 209 | 560 |
| 16 | 1793 | 29.9 | 0.00% | 528 | 926 | 1191 | 1611 | 520 | 917 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 156 | 246 | 6 |
| 4 | 362 | 495 | 204 |
| 8 | 589 | 576 | 411 |
| 16 | 966 | 904 | 891 |

Container api trong lúc chạy k6: CPU trung bình 347%, cao nhất 410% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 1829 MB [Đã đo].

### Chế độ `cache`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **14.3 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4469 | 74.5 | 0.00% | 5 | 39 | 70 | 343 | 3 | 36 |
| 4 | 5169 | 86.2 | 0.00% | 39 | 103 | 150 | 198 | 31 | 98 |
| 8 | 5097 | 85.0 | 0.00% | 89 | 179 | 219 | 325 | 84 | 172 |
| 16 | 5185 | 86.4 | 0.00% | 182 | 294 | 350 | 489 | 176 | 289 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 40 | 47 | 6 |
| 4 | 104 | 117 | 72 |
| 8 | 182 | 183 | 143 |
| 16 | 299 | 294 | 276 |

Lời gọi mentor thất bại trong lần đo này: 20 (trong đó 20 lần HTTP 429 do giới hạn request); request đó chạy không có kênh văn bản [Đã đo].

Container api trong lúc chạy k6: CPU trung bình 342%, cao nhất 412% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 2212 MB [Đã đo].

### Chế độ `cache-c0`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **10.7 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9754 | 162.6 | 0.00% | 3 | 18 | 30 | 184 | 2 | 16 |
| 4 | 4499 | 75.0 | 0.00% | 43 | 131 | 191 | 280 | 38 | 126 |
| 8 | 1763 | 29.4 | 0.00% | 235 | 647 | 909 | 1030 | 228 | 642 |
| 16 | 1719 | 28.6 | 0.00% | 491 | 1260 | 1871 | 2158 | 482 | 1253 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 17 | 26 | 3 |
| 4 | 132 | 164 | 59 |
| 8 | 551 | 868 | 289 |
| 16 | 1128 | 1784 | 619 |

Container api trong lúc chạy k6: CPU trung bình 309%, cao nhất 407% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 1811 MB [Đã đo].

### Chế độ `cache-notext`

Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): **9.3 s** [Đã đo].

| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | Server p50 | Server p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9904 | 165.1 | 0.00% | 3 | 17 | 30 | 219 | 2 | 15 |
| 4 | 9242 | 154.0 | 0.00% | 21 | 58 | 80 | 135 | 17 | 54 |
| 8 | 9312 | 155.2 | 0.00% | 48 | 89 | 107 | 162 | 44 | 85 |
| 16 | 9395 | 156.6 | 0.00% | 99 | 153 | 179 | 274 | 96 | 149 |

p95 theo loại truy vấn (ms):

| VU | single | multi_intent | multi_hop |
|---|---|---|---|
| 1 | 17 | 25 | 3 |
| 4 | 58 | 68 | 40 |
| 8 | 89 | 95 | 75 |
| 16 | 155 | 158 | 140 |

Container api trong lúc chạy k6: CPU trung bình 339%, cao nhất 401% (100% = 1 lõi; giới hạn 400%); RAM cao nhất 1867 MB [Đã đo].

