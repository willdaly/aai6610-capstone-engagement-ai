"""Chunk the scraped corpus and build a BM25 retrieval index.

Deliberately simple, per ``docs/assistant_plan.md``: BM25 over chunked page text, no
embeddings, no network, no model dependency. Everything here is deterministic and runs
offline against either the real corpus (``assistant/corpus/pages/``) or the committed
mini-corpus fixture, so the retrieval path is fully testable without a scrape.

Chunking follows the plan: group by heading where the page has headings, else by
paragraph, targeting 200-500 words per chunk, with the page title and section heading
prepended to each chunk's text and the source URL carried on every chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PAGES_DIR = REPO_ROOT / "assistant" / "corpus" / "pages"

MIN_CHUNK_WORDS = 200
MAX_CHUNK_WORDS = 500

# A small, fixed English stopword list. Kept inline (not a dependency) so tokenization is
# reproducible and reviewable. Retrieval quality on the eval set is reported in findings;
# if this list ever needs tuning, tune it there with the numbers, not silently.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have he her his i in into is it its
    my of on or our she that the their them they this to was we were what when where
    which who will with you your
    how do does did done can could would should i'm i've there here about into any
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    blocks: tuple[dict, ...]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. ``text`` is what BM25 sees and what the answer quotes.

    ``text`` is ``"<title> — <section>\\n\\n<body>"`` (section omitted when the chunk has
    none), so it always starts with the page title. ``body`` is the raw content without
    the prepended header. ``section`` is the heading this chunk sits under, or None.
    """

    chunk_id: str
    url: str
    title: str
    section: str | None
    text: str
    body: str
    word_count: int


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and single characters."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def load_corpus(pages_dir: Path | str = CORPUS_PAGES_DIR) -> list[Page]:
    """Load page JSON files written by the scraper (or the fixture) into Page objects.

    Pages are returned sorted by URL so chunk ordering and IDs are deterministic across
    runs and machines.
    """
    pages_dir = Path(pages_dir)
    pages: list[Page] = []
    for path in sorted(pages_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages.append(
            Page(
                url=data["url"],
                title=data["title"],
                blocks=tuple(data["blocks"]),
            )
        )
    return pages


def _split_oversized(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_page(
    page: Page, min_words: int = MIN_CHUNK_WORDS, max_words: int = MAX_CHUNK_WORDS
) -> list[Chunk]:
    """Split one page into 200-500 word chunks grouped under headings.

    Guarantees, which the property tests check:

    - Every chunk's body is at most ``max_words`` words (an oversized paragraph is split).
    - A chunk shorter than ``min_words`` is always the last chunk within its
      ``(url, section)`` group; short tails and short sections are not padded.
    - Every chunk carries the page's URL and its text starts with the page title.
    """
    chunks: list[Chunk] = []
    section: str | None = None
    buffer: list[str] = []
    buf_words = 0

    def make(body: str) -> Chunk:
        header = page.title if not section else f"{page.title} — {section}"
        return Chunk(
            chunk_id=f"{page.url}#{len(chunks)}",
            url=page.url,
            title=page.title,
            section=section,
            text=f"{header}\n\n{body}",
            body=body,
            word_count=len(body.split()),
        )

    def flush() -> None:
        nonlocal buffer, buf_words
        if buffer:
            chunks.append(make("\n".join(buffer)))
            buffer = []
            buf_words = 0

    for block in page.blocks:
        if block["kind"] == "heading":
            flush()
            section = block["text"]
            continue

        text = block["text"]
        words = len(text.split())
        if words > max_words:
            flush()
            for piece in _split_oversized(text, max_words):
                buffer = [piece]
                buf_words = len(piece.split())
                flush()
            continue

        if buf_words + words > max_words:
            flush()
        buffer.append(text)
        buf_words += words
        if buf_words >= min_words:
            flush()

    flush()
    return chunks


def chunk_corpus(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page))
    return chunks


class BM25Index:
    """A BM25 index over chunks, with the chunks kept alongside for result assembly."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build an index over zero chunks.")
        self.chunks = chunks
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self.vocabulary = {tok for doc in self._tokenized for tok in doc}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top-k ``(chunk, score)`` by BM25, highest score first.

        Scores are raw BM25 scores; the answer layer applies the confidence threshold.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(chunk, float(score)) for chunk, score in ranked[:k]]

    def top_overlap(self, query: str, top_chunk: "Chunk") -> float:
        """Fraction of the query's content tokens that appear in the top-ranked chunk.

        This is the signal that separates in-scope from out-of-scope on a large, varied
        corpus: an unrelated question may share a common word with *some* page (so
        corpus-wide coverage is nonzero), but its terms rarely co-occur in the single best
        chunk. Near 1.0 means the best passage really is about the question.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        chunk_tokens = set(tokenize(top_chunk.text))
        present = sum(1 for tok in q_tokens if tok in chunk_tokens)
        return present / len(q_tokens)

    def query_coverage(self, query: str) -> float:
        """Fraction of the query's content tokens that appear anywhere in the corpus.

        A question about a topic the corpus never mentions has near-zero coverage; the
        answer layer uses this as a fast out-of-scope signal before scoring.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        present = sum(1 for tok in q_tokens if tok in self.vocabulary)
        return present / len(q_tokens)


def build_index(pages_dir: Path | str = CORPUS_PAGES_DIR) -> BM25Index:
    """Convenience: load the corpus, chunk it, and build the index."""
    return BM25Index(chunk_corpus(load_corpus(pages_dir)))
