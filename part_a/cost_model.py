"""Mô hình chi phí (PLAN.md 3.6).

Chi_phí(N) = N * Σ_i f_i * c_i
  f_i: tỷ lệ ảnh đi tới tầng i (đo trên dữ liệu thật)
  c_i: giá_máy_mỗi_giờ / (thông_lượng_ảnh_mỗi_giây * 3600)
"""


def cost_per_image(price_per_hour: float, images_per_second: float) -> float:
    return price_per_hour / (images_per_second * 3600)


def pipeline_cost(n_images: int, stages: list[tuple[float, float]]) -> float:
    """stages: danh sách (f_i, c_i)."""
    return n_images * sum(f * c for f, c in stages)
