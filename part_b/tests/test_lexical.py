import pytest

from conftest import img
from imgsearch.retrieval.bm25 import BM25Index, doc_terms, query_terms
from imgsearch.retrieval.lexical import LexicalIndex


@pytest.fixture(scope="module")
def lex(mini_index, mini_store) -> LexicalIndex:
    out, _ = mini_index
    return LexicalIndex.load(out / "vtest" / "lexical.npz", mini_store)


def ids(results) -> list[str]:
    return [image_id for image_id, _ in results]


def test_query_terms_accent_per_token():
    assert query_terms("áo dai") == ["a:áo", "f:dai"]


def test_query_terms_drop_stopwords_and_duplicates():
    assert query_terms("ảnh của") == []
    assert query_terms("") == []
    assert query_terms("đen ĐEN") == ["a:đen"]


def test_doc_terms_store_both_forms_and_weight_names():
    terms = doc_terms("Hà", "", "chùa")
    assert terms.count("a:hà") == 3 and terms.count("f:ha") == 3
    assert terms.count("a:chùa") == 1 and terms.count("f:chua") == 1


def test_bm25_save_load_roundtrip(tmp_path):
    index = BM25Index.build([["f:a", "f:b"], ["f:b"], ["f:c"]])
    index.save(tmp_path / "x.npz")
    again = BM25Index.load(tmp_path / "x.npz")
    assert again.search(["f:b"], 10) == index.search(["f:b"], 10)
    assert [d for d, _ in again.search(["f:a", "f:b"], 10)] == [0, 1]


def test_bm25_rare_term_scores_higher():
    index = BM25Index.build([["f:common", "f:rare"], ["f:common"], ["f:common"], ["f:other"]])
    (top, s_top), (_, s_next) = index.search(["f:common", "f:rare"], 2)
    assert top == 0 and s_top > s_next


def test_bm25_empty_corpus():
    assert BM25Index.build([]).search(["f:a"], 10) == []


def test_caption_and_title_match(lex):
    assert ids(lex.search("áo dài"))[0] == img("hnh1")


def test_L1_unaccented_query_finds_same_images(lex):
    assert ids(lex.search("ao dai"))[0] == img("hnh1")
    accented = ids(lex.search("Đen Vâu"))
    unaccented = ids(lex.search("den vau"))
    # Không dấu rộng hơn ("den" còn khớp "đen", "đèn"), nhưng phải chứa mọi kết quả có dấu
    # và ảnh của Đen Vâu (khớp cả hai từ) vẫn đứng đầu.
    assert set(accented) <= set(unaccented)
    assert set(unaccented[:3]) == {img("den1"), img("den2"), img("den3")}


def test_accented_token_does_not_match_other_tone(lex):
    assert ids(lex.search("chùa")) == [img("mt4")]
    assert ids(lex.search("chưa")) == [img("hw2")]
    assert set(ids(lex.search("chua"))) == {img("mt4"), img("hw2")}


def test_names_column_finds_person_images(lex):
    found = set(ids(lex.search("Đen Vâu")))
    assert {img("den1"), img("den2"), img("den3")} <= found
    assert img("den4") not in found  # rejected


def test_alias_is_searchable(lex):
    assert set(ids(lex.search("Hà Hồ"))) >= {img("hnh1"), img("hnh2")}


def test_E7_image_in_two_entities_returned_once(lex):
    results = ids(lex.search("Phạm Nhật Vượng"))
    assert results.count(img("pnv3")) == 1


def test_person_filter(lex):
    everyone = set(ids(lex.search("sân khấu")))
    assert {img("den1"), img("hnh2"), img("mt1"), img("hw2")} <= everyone
    assert ids(lex.search("sân khấu", person_ids=["my_tam"])) == [img("mt1")]


def test_more_matching_terms_rank_higher(lex):
    assert ids(lex.search("cầu lông huy chương"))[0] == img("ntli2")


def test_limit(lex):
    assert len(lex.search("sân khấu", limit=2)) == 2


@pytest.mark.parametrize(
    "query", ['"', "OR", "NEAR(a b)", "a* -b ^c", "{names}: x", "' OR 1=1 --", "x" * 2000]
)
def test_L9_fts_syntax_in_query_is_harmless(lex, query):
    assert isinstance(lex.search(query), list)
