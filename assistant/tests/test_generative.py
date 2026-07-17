"""Generative mode's citation post-check, exercised without a network call.

The Anthropic client is stubbed, so these run keyless in CI. They lock the rule from the
plan: a generative answer is trusted only if every URL it cites was among the retrieved
passages and it cited at least one; otherwise it is demoted to the escalation response.
"""

import sys
import types

import pytest

from answer import Assistant
from index import BM25Index, chunk_corpus, load_corpus


@pytest.fixture(scope="module")
def assistant():
    from conftest import FIXTURE_PAGES_DIR

    return Assistant(BM25Index(chunk_corpus(load_corpus(FIXTURE_PAGES_DIR))))


def _install_fake_anthropic(monkeypatch, answer_text: str):
    """Make `import anthropic` inside _generative_answer return a stub with a fixed reply."""
    block = types.SimpleNamespace(type="text", text=answer_text)
    response = types.SimpleNamespace(content=[block])

    class FakeMessages:
        def create(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


def _results(assistant, query):
    return assistant.index.search(query, k=3)


def test_grounded_answer_passes(assistant, monkeypatch):
    query = "What is the care network?"
    results = _results(assistant, query)
    cited = results[0][0].url
    _install_fake_anthropic(
        monkeypatch, f"The care network is a list of clinics. Sources: {cited}"
    )
    r = assistant._generative_answer(query, results, coverage=1.0)
    assert r.kind == "answer"
    assert r.mode == "generative"
    assert any(c.url == cited for c in r.citations)


def test_answer_citing_unretrieved_url_is_demoted_to_escalation(assistant, monkeypatch):
    query = "What is the care network?"
    results = _results(assistant, query)
    _install_fake_anthropic(
        monkeypatch,
        "The care network is described here. Sources: https://evil.example.com/made-up",
    )
    r = assistant._generative_answer(query, results, coverage=1.0)
    assert r.kind == "escalation"


def test_answer_with_no_citation_is_demoted_to_escalation(assistant, monkeypatch):
    query = "What is the care network?"
    results = _results(assistant, query)
    _install_fake_anthropic(monkeypatch, "The care network is a list of clinics.")
    r = assistant._generative_answer(query, results, coverage=1.0)
    assert r.kind == "escalation"


def test_generative_mode_requires_a_key(assistant, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    query = "What is the care network?"
    r = assistant._generative_answer(query, _results(assistant, query), coverage=1.0)
    # No key: never touches the network; escalates instead.
    assert r.kind == "escalation"
