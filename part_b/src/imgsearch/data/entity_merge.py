"""Gộp entity (tên mô tả, phân mảnh) thành person.

Ứng viên từ tên chuẩn hóa (bỏ cụm mô tả, bỏ dấu, so khớp mờ) -> xác nhận bằng cosine
tâm ArcFace + category -> person_id; vùng không chắc cho người xem.
Cẩn thận trùng tên khác người (ví dụ "Sơn Tùng" nhà văn vs "Sơn Tùng M-TP" ca sĩ).
"""
