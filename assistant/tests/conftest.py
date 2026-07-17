"""Shared fixtures for the assistant tests.

Everything here runs offline against the committed fake mini-corpus. The real scraped
corpus is gitignored, so tests that need it (retrieval quality on the real eval set,
manifest integrity against the real pages) skip when it is absent rather than fail.
"""

from pathlib import Path

import pytest

from index import BM25Index, chunk_corpus, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PAGES_DIR = REPO_ROOT / "assistant" / "tests" / "fixtures" / "mini_corpus" / "pages"
REAL_CORPUS_PAGES_DIR = REPO_ROOT / "assistant" / "corpus" / "pages"
MANIFEST_PATH = REPO_ROOT / "assistant" / "corpus_manifest.json"


@pytest.fixture(scope="session")
def fixture_pages():
    """Pages loaded from the fake mini-corpus fixture."""
    return load_corpus(FIXTURE_PAGES_DIR)


@pytest.fixture(scope="session")
def fixture_chunks(fixture_pages):
    return chunk_corpus(fixture_pages)


@pytest.fixture(scope="session")
def fixture_index(fixture_chunks):
    return BM25Index(fixture_chunks)
