"""CLI smoke test on the fake fixture corpus, so CI needs no scrape.

Monkeypatches the CLI's corpus location to the fixture and exercises the single-question
path for an answer, a refusal, and an escalation, plus the banner's required disclosures.
"""

import chat
from answer import AnswerResult, Citation
from index import BM25Index, chunk_corpus, load_corpus


def _fixture_assistant():
    from answer import Assistant
    from conftest import FIXTURE_PAGES_DIR

    return Assistant(BM25Index(chunk_corpus(load_corpus(FIXTURE_PAGES_DIR))))


def test_banner_states_identity_and_no_data_collection():
    text = chat.banner(generative=False)
    low = text.lower()
    assert "not affiliated" in low or "unofficial" in low
    assert "not store" in low or "not stored" in low or "no data collection" in low
    assert "medical advice" in low


def test_cli_answers_refuses_and_escalates(monkeypatch, capsys):
    assistant = _fixture_assistant()
    monkeypatch.setattr(chat, "build_assistant", lambda: assistant)

    chat.main(["--ask", "How do I contact the foundation?"])
    out = capsys.readouterr().out
    assert "[ANSWER]" in out
    assert "http" in out  # a cited source URL

    chat.main(["--ask", "Should I change my daughter's seizure medication?"])
    out = capsys.readouterr().out
    assert "can't answer" in out.lower()

    chat.main(["--ask", "What is the best pizza recipe?"])
    out = capsys.readouterr().out
    assert "contact the foundation" in out.lower()


def test_format_result_deduplicates_source_urls():
    result = AnswerResult(
        kind="answer",
        text="body",
        citations=[
            Citation("T", "https://x.org/a"),
            Citation("T", "https://x.org/a"),
        ],
    )
    rendered = chat.format_result(result)
    assert rendered.count("https://x.org/a") == 1
