from imgsearch.retrieval.multi_hop import find_edges
from imgsearch.retrieval.multi_intent import diversify, intersect, restrict


def test_intersect():
    assert intersect([{"a", "b"}, {"b", "c"}]) == {"b"}
    assert intersect([]) == set()


def test_restrict_keeps_order_then_appends_rest():
    ranked = [("x", 3.0), ("a", 2.0), ("b", 1.0)]
    assert restrict(ranked, {"a", "c"}, ["c", "a"]) == [("a", 2.0), ("c", 0.0)]


def test_diversify_caps_each_person():
    ranked = [("1", 5.0), ("2", 4.0), ("3", 3.0), ("4", 2.0)]
    persons = {"1": ["p"], "2": ["p"], "3": ["p"], "4": ["q"]}
    assert [i for i, _ in diversify(ranked, persons, cap=2)] == ["1", "2", "4", "3"]
    assert diversify(ranked, persons, cap=0) == ranked


def test_find_edges_both_directions(mini_store):
    forward = find_edges(mini_store, "tran_thanh", "spouse")
    backward = find_edges(mini_store, "hari_won", "spouse")
    assert [e.target for e in forward] == ["hari_won"]
    assert [e.target for e in backward] == ["tran_thanh"]


def test_find_edges_inverse_predicate(mini_store):
    # Cạnh (Phạm Nhật Vượng, child, Minh Hoàng): hỏi "bố/mẹ của Minh Hoàng" ra Phạm Nhật Vượng.
    assert [e.target for e in find_edges(mini_store, "pham_nhat_minh_hoang", "parent")] == [
        "pham_nhat_vuong"
    ]
    # Nhưng không có nghĩa Minh Hoàng có con.
    assert find_edges(mini_store, "pham_nhat_minh_hoang", "child") == []
