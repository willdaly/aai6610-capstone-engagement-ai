"""Chunker properties and BM25 retrieval behavior.

The size-bound and packing properties are checked on synthetic pages so the bounds are
exercised precisely (a long paragraph that must be split, several short paragraphs that
must be packed). URL/title propagation and basic retrieval are checked on the fake
fixture corpus.
"""

from index import (
    MAX_CHUNK_WORDS,
    MIN_CHUNK_WORDS,
    BM25Index,
    Page,
    chunk_page,
    tokenize,
)


def _page(blocks, url="https://example.org/p.html", title="Test Page"):
    return Page(url=url, title=title, blocks=tuple(blocks))


def _text(word: str, n: int) -> dict:
    return {"kind": "text", "text": " ".join([word] * n)}


def _heading(text: str) -> dict:
    return {"kind": "heading", "text": text}


class TestChunkSizeBounds:
    def test_no_chunk_exceeds_max_words(self):
        # One 1200-word paragraph must split into chunks each within the max bound.
        page = _page([_text("alpha", 1200)])
        chunks = chunk_page(page)
        assert len(chunks) >= 3
        assert all(c.word_count <= MAX_CHUNK_WORDS for c in chunks)

    def test_short_paragraphs_pack_toward_the_minimum(self):
        # Six 60-word paragraphs under one heading should pack into >=200 word chunks,
        # not stay as six tiny chunks.
        blocks = [_heading("Section")] + [_text(f"w{i}", 60) for i in range(6)]
        chunks = chunk_page(page := _page(blocks))
        assert page  # silence linters about walrus
        # 360 words total -> at least one full chunk over the minimum, plus a tail.
        assert any(c.word_count >= MIN_CHUNK_WORDS for c in chunks)
        assert all(c.word_count <= MAX_CHUNK_WORDS for c in chunks)

    def test_below_minimum_chunk_is_the_last_in_its_section(self):
        # A section with 360 words -> a ~>=200 chunk then a short tail. The short tail is
        # the last chunk of that (url, section) group; a short chunk never appears mid
        # section. Also covers the whole-section-shorter-than-min case.
        blocks = (
            [_heading("Long")] + [_text(f"a{i}", 60) for i in range(6)]
            + [_heading("Short")] + [_text("b", 30)]
        )
        chunks = chunk_page(_page(blocks))
        by_section: dict = {}
        for i, c in enumerate(chunks):
            by_section.setdefault(c.section, []).append((i, c))
        for section, items in by_section.items():
            last_idx = items[-1][0]
            for idx, chunk in items:
                if chunk.word_count < MIN_CHUNK_WORDS:
                    assert idx == last_idx, (
                        f"short chunk mid-section in {section!r}: idx {idx} of {items}"
                    )


class TestChunkMetadata:
    def test_every_chunk_carries_source_url(self, fixture_chunks):
        assert fixture_chunks
        for c in fixture_chunks:
            assert c.url.startswith("https://")
            assert c.chunk_id.startswith(c.url)

    def test_chunk_text_starts_with_page_title(self, fixture_chunks):
        for c in fixture_chunks:
            assert c.text.startswith(c.title)

    def test_section_heading_is_prepended_when_present(self, fixture_chunks):
        sectioned = [c for c in fixture_chunks if c.section]
        assert sectioned, "fixture should produce sectioned chunks"
        for c in sectioned:
            assert c.section in c.text.split("\n\n", 1)[0]

    def test_body_excludes_the_prepended_header(self, fixture_chunks):
        for c in fixture_chunks:
            assert c.body in c.text
            assert not c.body.startswith(c.title)


class TestTokenize:
    def test_lowercases_and_drops_stopwords(self):
        assert "the" not in tokenize("The Foundation")
        assert "foundation" in tokenize("The Foundation")

    def test_drops_single_characters_and_punctuation(self):
        toks = tokenize("a b clinic, network!")
        assert "clinic" in toks and "network" in toks
        assert "a" not in toks and "b" not in toks


class TestRetrieval:
    def test_in_scope_query_retrieves_the_expected_page(self, fixture_index):
        results = fixture_index.search("how do I contact the foundation", k=3)
        assert results
        top_chunk, top_score = results[0]
        assert top_chunk.url.endswith("contact.html")
        assert top_score > 0

    def test_results_are_sorted_by_descending_score(self, fixture_index):
        results = fixture_index.search("clinic care network", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_out_of_scope_query_has_zero_coverage_and_score(self, fixture_index):
        assert fixture_index.query_coverage("quarterly earnings stock market") == 0.0
        results = fixture_index.search("quarterly earnings stock market", k=3)
        assert results[0][1] == 0.0

    def test_empty_or_stopword_only_query_returns_no_results(self, fixture_index):
        assert fixture_index.search("the and of", k=3) == []
        assert fixture_index.query_coverage("the and of") == 0.0

    def test_index_over_zero_chunks_is_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            BM25Index([])
