"""Trích quan hệ từ tiêu đề bài báo (`description`) bằng mẫu câu.

Ví dụ: "X - con trai (thứ n) của (tỷ phú|ông|bà) Y", "Y cùng con trai X", "ông xã/bà xã của Y".
Hai đầu phải liên kết được tới person đã gộp; lưu tiêu đề + source_url làm bằng chứng.
Loại nhiễu như "Anh trai say hi" (tên chương trình). Cạnh dùng trong test case phải kiểm tra tay.
"""
