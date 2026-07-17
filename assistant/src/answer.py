"""Answer, refuse, or escalate: the behavior core of the assistant.

The rules here are the graded ethics content of this build (``docs/assistant_plan.md``,
behavior rules 1-4), not stylistic choices:

- **Grounded or escalated.** A substantive answer must come from retrieved corpus
  passages, cited by title and URL. Below the retrieval-confidence threshold, or with no
  corpus coverage, the assistant escalates ("contact the Foundation") instead of guessing.
  It never answers from general knowledge.
- **No medical advice, ever.** A question seeking diagnosis, treatment or medication
  choices, dosing, prognosis for a specific person, or interpretation of a specific
  person's symptoms or scans gets a fixed refusal that redirects to the family's clinical
  team and the Foundation's clinical resources. This fires *before* retrieval and even
  when the corpus holds related educational content.
- **Honest identity, no data collection.** Enforced in the CLI banner (see chat.py) and
  the system prompt; this module never stores the query.

Two modes share this decision logic. Extractive mode (default, offline, deterministic)
quotes the retrieved passages. Generative mode (only when ``ANTHROPIC_API_KEY`` is set)
synthesizes from the same passages under the committed system prompt, then a citation
post-check demotes any answer citing a URL that was not retrieved to an escalation. The
medical refusal and the escalation path are identical in both modes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from index import BM25Index, Chunk, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "assistant" / "templates"
SYSTEM_PROMPT_PATH = REPO_ROOT / "assistant" / "system_prompt.md"

# Retrieval-confidence gate. Three signals, all tuned on the eval set's in-scope vs
# out-of-scope split (see docs/assistant_findings.md for the distributions):
#
# - best_overlap: the best fraction of the question's content words covered by any one of
#   the top retrieved chunks. This is the discriminating signal on a large, varied corpus,
#   where an out-of-scope question can share a common word with some page but its terms do
#   not co-occur in any good passage. On the eval set it separates cleanly: in-scope min
#   0.40, out-of-scope max 0.33, so 0.35 sits in the gap.
# - coverage: fraction of the question's content words present anywhere in the corpus. A
#   cheap first filter for questions about topics the corpus never mentions.
# - top_score: a raw BM25 floor, so a question that matches nothing at all escalates even
#   if the overlap arithmetic is generous on a very short query.
OVERLAP_THRESHOLD = 0.35
COVERAGE_THRESHOLD = 0.5
SCORE_THRESHOLD = 1.0

TOP_K = 3
# Extractive answers quote at most this many words of body per cited passage, trimmed at
# a sentence boundary, so the reply stays short and readable in a terminal.
QUOTE_WORD_BUDGET = 90


@dataclass(frozen=True)
class Citation:
    title: str
    url: str


@dataclass
class AnswerResult:
    kind: str  # "answer" | "refusal" | "escalation"
    text: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "extractive"  # "extractive" | "generative"
    # Diagnostics, useful in eval and tests; not shown to the user.
    top_score: float = 0.0
    coverage: float = 0.0
    retrieved_urls: list[str] = field(default_factory=list)


# --- Medical-advice detection -------------------------------------------------------

# A question is treated as seeking medical advice if it either matches a direct
# advice/dosing/diagnosis pattern, or it refers to a specific person AND raises a clinical
# concern (symptom, treatment, safety). The second arm is what catches gently phrased
# traps ("my daughter's eye looks cloudy, is that normal?"). The cost of a false positive
# is a needless escalation, which is safe; the cost of a false negative is giving medical
# advice, which is not. When unsure, this errs toward refusing.

_PERSONAL_RE = re.compile(
    r"\b(?:my|our|his|her|their|the)\s+"
    r"(?:child|kid|children|son|daughter|baby|babies|newborn|infant|toddler|"
    r"husband|wife|mom|mother|dad|father|partner|patient)\b",
    re.I,
)
_FIRST_PERSON_MEDICAL_RE = re.compile(r"\b(?:i|we)\s+(?:have|has|got|am|are|was|were)\b", re.I)

_SYMPTOM_RE = re.compile(
    r"\b(?:seizure|seizures|rash|swelling|swollen|cloudy|bulging|glaucoma|pressure|"
    r"pain|fever|vomit|vomiting|bleeding|bruis|lump|spasm|twitch|stroke|weakness|"
    r"numb|headache|migraine|symptom|symptoms|birthmark|port[-\s]?wine|stain|"
    r"mri|ct|eeg|scan|scans|x[-\s]?ray|ultrasound|biopsy|leptomeningeal|"
    r"calcification|side effect|side[-\s]?effects)\b",
    re.I,
)
_TREATMENT_RE = re.compile(
    r"\b(?:medication|medications|medicine|medicines|drug|drugs|dose|doses|dosage|"
    r"dosing|mg|milligram|steroid|steroids|aspirin|anticonvulsant|anti[-\s]?seizure|"
    r"surgery|operation|operate|laser|treatment|treatments|therapy|chemo|radiation|"
    r"prescri\w*|inject\w*|vaccine|supplement)\b",
    re.I,
)
_CONCERN_RE = re.compile(
    r"\b(?:should|safe|okay|ok to|normal|worried|worry|concern\w*|dangerous|serious|"
    r"risk|emergency|need to|have to|is it|are they|will (?:he|she|they|my))\b",
    re.I,
)
_DIRECT_ADVICE_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bdos(?:e|es|age|ing)\b",
        r"\bhow (?:much|many)\b.*\b(?:give|take|dose|mg|medication|medicine)\b",
        r"\bwhat (?:medication|medicine|drug|dose|dosage|treatment|surgery)\b",
        r"\bwhich\b.{0,40}\b(?:medication|medicine|drug|treatment|surgery)\b",
        r"\bwhich (?:doctor|specialist|clinic)\b.*\b(?:should|best|right)\b",
        r"\bshould (?:i|we|he|she|my|our|they)\b.*\b(?:take|give|start|stop|switch|"
        r"increase|decrease|wean|try|see|test|operate|do)\b",
        r"\b(?:stop|start|change|increase|decrease|wean)\b.*\b(?:medication|medicine|"
        r"dose|drug|treatment)\b",
        r"\bis (?:it|this) (?:safe|okay|ok|normal|dangerous|serious)\b",
        r"\bdiagnos\w*\b",
        r"\bprognos\w*\b",
        r"\blife expectancy\b",
        r"\bhow long\b.*\b(?:live|survive)\b",
        r"\bwhat (?:does|do)\b.*\b(?:scan|mri|ct|eeg|result|results|symptom|symptoms)\b.*\bmean\b",
        r"\binterpret\b.*\b(?:scan|mri|result|results|symptom)\b",
        r"\bwill (?:my|our|he|she|they)\b.*\b(?:be (?:ok|okay|fine|alright)|get worse|"
        r"be cured|recover|die)\b",
        r"\bhome remedy\b|\bnatural (?:cure|remedy|treatment)\b",
    )
]


def is_medical_advice(query: str) -> bool:
    """True if the question seeks medical advice about a specific person.

    Rule-based and English-only by design (see the plan: BM25/deterministic, no model in
    the default path). Limitations are documented in the findings.
    """
    for pattern in _DIRECT_ADVICE_PATTERNS:
        if pattern.search(query):
            return True

    personal = bool(_PERSONAL_RE.search(query) or _FIRST_PERSON_MEDICAL_RE.search(query))
    clinical = bool(
        _SYMPTOM_RE.search(query)
        or _TREATMENT_RE.search(query)
        or _CONCERN_RE.search(query)
    )
    return personal and clinical


# --- Template loading ---------------------------------------------------------------


def _load_template(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


# --- The assistant ------------------------------------------------------------------


class Assistant:
    """Wraps a BM25 index with the answer/refuse/escalate decision logic."""

    def __init__(
        self,
        index: BM25Index,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        coverage_threshold: float = COVERAGE_THRESHOLD,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        self.index = index
        self.overlap_threshold = overlap_threshold
        self.coverage_threshold = coverage_threshold
        self.score_threshold = score_threshold
        self.contact = self._find_page(("contact",))
        # Prefer the clinical *care* network page over other pages that merely contain
        # "clinical" (e.g. clinical-research-participation), so a medical refusal points a
        # family to care resources rather than a research sign-up.
        self.clinical = (
            self._find_page(("clinical-care", "care-network", "care_network"))
            or self._find_page(("clinical",))
            or self.contact
        )

    def _find_page(self, url_substrings: tuple[str, ...]) -> Citation | None:
        for chunk in self.index.chunks:
            low = chunk.url.lower()
            if any(sub in low for sub in url_substrings):
                return Citation(title=chunk.title, url=chunk.url)
        return None

    # -- rendering -------------------------------------------------------------------

    def _contact_fields(self) -> dict[str, str]:
        contact = self.contact or Citation(
            "Contact the Sturge-Weber Foundation",
            "https://sturge-weber.org/who-we-are/contact.html",
        )
        clinical = self.clinical or contact
        return {
            "contact_title": contact.title,
            "contact_url": contact.url,
            "clinical_title": clinical.title,
            "clinical_url": clinical.url,
        }

    def refusal(self, mode: str = "extractive") -> AnswerResult:
        text = _load_template("refusal_medical").format(**self._contact_fields())
        cites = [c for c in (self.clinical, self.contact) if c]
        return AnswerResult(kind="refusal", text=text, citations=cites, mode=mode)

    def escalation(self, mode: str = "extractive", **diag) -> AnswerResult:
        text = _load_template("escalation").format(**self._contact_fields())
        cites = [self.contact] if self.contact else []
        return AnswerResult(kind="escalation", text=text, citations=cites, mode=mode, **diag)

    def _trim_quote(self, body: str) -> str:
        words = body.split()
        if len(words) <= QUOTE_WORD_BUDGET:
            return body.strip()
        snippet = " ".join(words[:QUOTE_WORD_BUDGET])
        # Trim back to the last sentence end so quotes don't stop mid-sentence.
        m = list(re.finditer(r"[.!?](?:\s|$)", snippet))
        if m:
            snippet = snippet[: m[-1].end()].strip()
        return snippet.strip()

    def _extractive_answer(
        self, results: list[tuple[Chunk, float]], coverage: float
    ) -> AnswerResult:
        # Use up to two distinct source pages, top-ranked first.
        used: list[Chunk] = []
        seen_urls: set[str] = set()
        for chunk, _score in results:
            if chunk.url in seen_urls:
                continue
            seen_urls.add(chunk.url)
            used.append(chunk)
            if len(used) == 2:
                break

        parts = []
        citations = []
        for chunk in used:
            label = chunk.section or chunk.title
            parts.append(f"From \"{label}\":\n{self._trim_quote(chunk.body)}")
            citations.append(Citation(title=chunk.title, url=chunk.url))
        pointer = _load_template("answer_pointer")
        body = "\n\n".join(parts) + "\n\n" + pointer
        return AnswerResult(
            kind="answer",
            text=body,
            citations=citations,
            mode="extractive",
            top_score=results[0][1],
            coverage=coverage,
            retrieved_urls=[c.url for c, _ in results],
        )

    # -- the decision ----------------------------------------------------------------

    def answer(self, query: str, generative: bool = False) -> AnswerResult:
        """Route a question to an answer, a medical refusal, or an escalation.

        Order matters: the medical-advice check runs first and unconditionally, before any
        retrieval, so related educational content in the corpus can never turn into advice.
        """
        if is_medical_advice(query):
            return self.refusal(mode="generative" if generative else "extractive")

        coverage = self.index.query_coverage(query)
        results = self.index.search(query, k=TOP_K)
        top_score = results[0][1] if results else 0.0
        top_overlap = self.index.best_overlap(query, [c for c, _ in results])
        diag = dict(top_score=top_score, coverage=coverage,
                    retrieved_urls=[c.url for c, _ in results])

        below_confidence = (
            not results
            or top_overlap < self.overlap_threshold
            or coverage < self.coverage_threshold
            or top_score < self.score_threshold
        )
        if below_confidence:
            return self.escalation(mode="generative" if generative else "extractive", **diag)

        if generative:
            return self._generative_answer(query, results, coverage)
        return self._extractive_answer(results, coverage)

    # -- generative mode -------------------------------------------------------------

    def _generative_answer(
        self, query: str, results: list[tuple[Chunk, float]], coverage: float
    ) -> AnswerResult:
        """Synthesize an answer from the retrieved passages using the Claude API.

        Requires ANTHROPIC_API_KEY. The system prompt (committed) carries the behavior
        rules. After generation, a citation post-check demotes the answer to an escalation
        if it cites any URL that was not among the retrieved passages, so the model cannot
        smuggle in an ungrounded source.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Should not happen (chat.py gates on the key), but never fall back to a
            # network call without one. Escalate rather than error.
            return self.escalation(top_score=results[0][1], coverage=coverage,
                                    retrieved_urls=[c.url for c, _ in results])

        import anthropic

        retrieved_urls = {c.url for c, _ in results}
        passages = "\n\n".join(
            f"[Passage {i+1}] {c.title} ({c.url})\n{c.body}"
            for i, (c, _) in enumerate(results)
        )
        system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        user = f"PASSAGES:\n{passages}\n\nQUESTION: {query}"

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        answer_text = "".join(block.text for block in resp.content if block.type == "text")

        # Citation post-check: pull every URL the model wrote and require that all of them
        # are among the retrieved passages, and that it cited at least one. An answer that
        # cites nothing, or cites a URL it was not given, is not trustworthy as grounded,
        # so it is demoted to the escalation response.
        cited_urls = {u.rstrip(".,);:") for u in re.findall(r"https?://[^\s()]+", answer_text)}
        ungrounded = [u for u in cited_urls if u not in retrieved_urls]
        if not cited_urls or ungrounded:
            return self.escalation(top_score=results[0][1], coverage=coverage,
                                   retrieved_urls=[c.url for c, _ in results])

        citations = [Citation(title=c.title, url=c.url) for c, _ in results if c.url in cited_urls]
        return AnswerResult(
            kind="answer",
            text=answer_text.strip(),
            citations=citations,
            mode="generative",
            top_score=results[0][1],
            coverage=coverage,
            retrieved_urls=[c.url for c, _ in results],
        )
