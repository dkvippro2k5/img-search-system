"""Dense retrieval: text encoder ONNX (multilingual-e5 / BGE-M3) + FAISS flat (tìm chính xác).

Lọc theo person bằng mảng chỉ số vec_row. Embedding ảnh (CLIP/SigLIP) là tùy chọn, chỉ bật
khi bake-off cho thấy "caption + ảnh" tốt hơn rõ rệt "chỉ caption".
"""
