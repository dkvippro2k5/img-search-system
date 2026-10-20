# reports/

Báo cáo sinh từ script; không sửa tay (trừ `eval.md`, bản tổng hợp dev + test). Chỉ commit `eval.md` và `perf.md` (số liệu nộp cuối); các file còn lại sinh lại khi chạy (`.gitignore`).

| File | Sinh bởi (chạy trong `part_b/`) |
|---|---|
| `perf.md` | `python bench/run_bench.py` (k6 qua Docker); `bench/make_report.py` dựng lại từ `bench/results/` |
| `eval_dev.md`, `eval_test.md` (+ `.json`) | `python -m imgsearch.eval.run_testset --index-version people --split dev|test` |
| `eval.md` | tổng hợp tay từ `eval_dev.md` + `eval_test.md` |
| `eval_golden_<ver>.md` | `python -m imgsearch.eval.run_eval --index-version <ver>` |
| `dense_bakeoff.md`, `mentor_bakeoff.md` | `bench/dense_bakeoff.py`, `bench/mentor_bakeoff.py` |
