# Bộ dữ liệu giả `mini`

Dùng cho test và chạy thử server khi chưa có dữ liệu thật. Định dạng: `imgsearch/data/dataset.py`.
Tên người là người thật trong dữ liệu capstone, nhưng **caption, tiêu đề, URL, sha256 đều là giả**.
Hai cạnh quan hệ lấy nguyên văn tiêu đề bài báo được trích trong `capstone/PLAN.md` mục 4.5.

`image_id` = `sha256("mini:<key>")`, ví dụ key `den1`, `hnh1` (xem `tests/conftest.py::img`).

Tình huống được cài sẵn:

| Tình huống | Dữ liệu |
|---|---|
| Ảnh bị loại / chờ duyệt không vào index | `den4` (rejected), `rev1` (review) |
| Ảnh gần trùng không vào index | `hnh3` (`dup_of` = `hnh1`) |
| E4: entity phân mảnh | Phạm Nhật Vượng có `e_pnv1`, `e_pnv2` |
| E7: một ảnh thuộc hai entity cùng người | `pnv3` |
| M2: ảnh có cả hai người | `tt_hw1` (Trấn Thành + Hari Won) |
| E2/E3: trùng tên | Sơn Tùng M-TP (ca sĩ) và Sơn Tùng (nhà văn) |
| H1: quan hệ có bằng chứng, hai đầu có ảnh | Trấn Thành spouse Hari Won |
| H4: quan hệ có bằng chứng, người đích không có ảnh | Phạm Nhật Vượng child Phạm Nhật Minh Hoàng |
| Không suy quan hệ từ tên | Phạm Nhật Quân Anh có ảnh nhưng không có cạnh |
| Cạnh lỗi bị loại | `my_tam sibling nguoi_khong_ton_tai` |
| Entity chưa gộp được vào person | `e_unmerged` / ảnh `grp1` |
| So dấu từng token | `mt4` có "chùa", `hw2` có "chưa" |

Sinh lại index để chạy thử server (trong `part_b/`):

```bash
python scripts/build_index.py --input tests/fixtures/mini --version dev
```
