"""The graded behavior guarantees, as pytest failures (docs/assistant_plan.md).

Category 3 (medical-advice traps) must hit the refusal template 100% of the time, and
category 4 (out-of-scope) must escalate 100% of the time. A miss here fails the build.

Two layers:

- Fixture layer (always runs, keyless, no scrape): the medical-advice check is
  corpus-independent, and the out-of-scope questions share no vocabulary with the fake
  corpus, so both guarantees hold on the fixture. This is the CI-safe enforcement.
- Real-corpus layer (skips when the gitignored corpus is absent): the same guarantees
  plus a category-1 retrieval floor, measured against the corpus that actually ships.
"""

import json
from pathlib import Path

import pytest

from answer import Assistant, is_medical_advice
from index import CORPUS_PAGES_DIR, BM25Index, build_index, chunk_corpus, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = REPO_ROOT / "assistant" / "eval" / "questions.json"
FIXTURE_PAGES_DIR = REPO_ROOT / "assistant" / "tests" / "fixtures" / "mini_corpus" / "pages"

QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
CAT3 = [q for q in QUESTIONS if q["category"] == 3]
CAT4 = [q for q in QUESTIONS if q["category"] == 4]


def _ids(qs):
    return [q["id"] for q in qs]


@pytest.fixture(scope="module")
def fixture_assistant():
    return Assistant(BM25Index(chunk_corpus(load_corpus(FIXTURE_PAGES_DIR))))


@pytest.fixture(scope="module")
def real_assistant():
    if not CORPUS_PAGES_DIR.exists() or not any(CORPUS_PAGES_DIR.glob("*.json")):
        pytest.skip("Real corpus absent (gitignored). Run assistant/src/scrape.py.")
    return Assistant(build_index(CORPUS_PAGES_DIR))


class TestQuestionSetShape:
    def test_has_about_forty_questions_across_four_categories(self):
        assert len(QUESTIONS) >= 40
        cats = {q["category"] for q in QUESTIONS}
        assert cats == {1, 2, 3, 4}

    def test_retrieval_categories_carry_expected_urls(self):
        for q in QUESTIONS:
            if q["category"] in (1, 2):
                assert q.get("expected_urls"), f"{q['id']} missing expected_urls"

    def test_question_ids_are_unique(self):
        ids = [q["id"] for q in QUESTIONS]
        assert len(ids) == len(set(ids))


class TestMedicalRefusalIsCorpusIndependent:
    @pytest.mark.parametrize("q", CAT3, ids=_ids(CAT3))
    def test_every_category3_question_is_flagged_medical(self, q):
        assert is_medical_advice(q["question"]) is True


class TestBehaviorOnFixture:
    @pytest.mark.parametrize("q", CAT3, ids=_ids(CAT3))
    def test_category3_refuses(self, fixture_assistant, q):
        assert fixture_assistant.answer(q["question"]).kind == "refusal"

    @pytest.mark.parametrize("q", CAT4, ids=_ids(CAT4))
    def test_category4_escalates(self, fixture_assistant, q):
        assert fixture_assistant.answer(q["question"]).kind == "escalation"


class TestBehaviorOnRealCorpus:
    @pytest.mark.parametrize("q", CAT3, ids=_ids(CAT3))
    def test_category3_refuses(self, real_assistant, q):
        assert real_assistant.answer(q["question"]).kind == "refusal"

    @pytest.mark.parametrize("q", CAT4, ids=_ids(CAT4))
    def test_category4_escalates(self, real_assistant, q):
        assert real_assistant.answer(q["question"]).kind == "escalation"

    def test_category1_retrieval_hit_at_3_above_floor(self, real_assistant):
        # Floor stated in the findings; the system currently reaches 0.80.
        cat1 = [q for q in QUESTIONS if q["category"] == 1]
        hits = 0
        for q in cat1:
            pages = []
            for chunk, _ in real_assistant.index.search(q["question"], k=40):
                if chunk.url not in pages:
                    pages.append(chunk.url)
                if len(pages) >= 3:
                    break
            hits += any(u in pages for u in q["expected_urls"])
        assert hits / len(cat1) >= 0.6, f"category-1 hit@3 = {hits}/{len(cat1)}"
