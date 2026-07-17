# Assistant Plan — Family Navigation Prototype

This file is the contract for the assistant build (proposal Areas 2 + 8): a
retrieval-grounded question-answering prototype over the Sturge-Weber Foundation's
public website, with escalation instead of guessing. Read `CLAUDE.md` first; its
assistant conventions and ethics constraints apply throughout and this plan sharpens
them into implementable rules.

The audience assumption that governs every behavior decision here: the user is a
parent or family member of a patient, possibly newly diagnosed, possibly frightened,
possibly a minor's caregiver reading at midnight. Tone is plain, calm, and careful.
The assistant navigates people to Foundation resources; it does not practice medicine,
and it does not improvise.

## Non-negotiable behavior rules

1. **Grounded or escalated, nothing in between.** Every substantive answer must be
   supported by retrieved passages from the corpus, with the source page (title + URL)
   cited in the response. If retrieval confidence is below threshold or the question
   is out of scope, the assistant says it cannot answer from Foundation materials and
   gives the escalation response (how to contact the Foundation, from the corpus's
   own contact page). It never answers from the model's general knowledge.
2. **No medical advice, ever.** Questions asking for diagnosis, treatment choices,
   medication guidance, dosing, prognosis for a specific person, or interpretation of
   a specific patient's symptoms/scans get a fixed, kind refusal template that
   redirects to the family's clinical team and to the Foundation's clinical resources
   page. This fires even when the corpus contains related educational content; the
   assistant may point to that content as reading, but may not convert it into
   advice.
3. **No data collection.** The prototype stores no user questions or identifying
   information beyond the ephemeral session. Say so in the interface banner.
4. **Honest identity.** The banner states it is an unofficial academic prototype, not
   affiliated with or endorsed by the Foundation, and that answers come only from the
   Foundation's public website as of the scrape date.

## Corpus: scraping rules and repository hygiene

- **Check robots.txt first.** Before any scraping, fetch and parse
  `sturge-weber.org/robots.txt`. If it disallows the pages we need, STOP and report;
  do not work around it. Record the robots.txt content and the check date in the
  manifest.
- **Scope:** public informational pages only: about, education/resource pages, FAQ,
  clinical care network, events/conference pages, newsletters if public, contact.
  No forms, no member areas, no third-party domains, no PDFs on other hosts.
- **Politeness:** identify with a custom User-Agent naming the project and a contact
  placeholder; 1 request/second max; cache so a re-run does not re-fetch unchanged
  pages.
- **Copyright hygiene, decided:** the scraped page content is the Foundation's
  copyrighted material and this repo is public, so raw page text is NOT committed.
  `assistant/corpus/` is gitignored. What IS committed: the scraper, and
  `assistant/corpus_manifest.json` recording for each page its URL, title, fetch
  timestamp, HTTP status, and a SHA-256 of the extracted text, so the corpus is
  reproducible by anyone running the scraper and verifiable against what we used.
- **Extraction:** strip navigation, headers, footers, cookie banners; keep main
  content, headings, and link text with URLs (navigation answers need them).

## Retrieval design

- **Chunking:** by heading structure where present, else paragraphs, targeting
  200-500 words per chunk with the page title and section heading prepended to each
  chunk's text. Every chunk carries its source URL.
- **Index:** BM25 (`rank_bm25`) over the chunks. Deliberately simple: deterministic,
  no network, no embedding model dependency, fully testable offline. If BM25 hit
  rates on the eval set are poor, report that in findings with the numbers; do not
  silently swap in embeddings this phase.
- **Confidence threshold:** a minimum top-score / margin rule tuned on the eval set's
  in-scope vs out-of-scope split, so "I can't answer that from Foundation materials"
  actually fires. The threshold and its tuning evidence go in the findings.

## Generation design

Two modes behind one interface, so the core is testable without network access:

- **Extractive mode (default, offline, deterministic):** answer = the top chunks
  quoted/trimmed with source citations and a one-line pointer ("This is from
  [page]; for anything about your child's care, please talk to your clinical
  team."). No paraphrase, no synthesis. This mode is what the automated tests run.
- **Generative mode (optional, `ANTHROPIC_API_KEY` from env, never committed):**
  the retrieved chunks plus a system prompt produce a synthesized answer. The system
  prompt lives at `assistant/system_prompt.md`, committed, and encodes the behavior
  rules above verbatim: answer only from the provided passages, cite sources, refuse
  medical advice with the template, escalate when passages are insufficient.
  Generative answers must end with their source list; a post-check verifies every
  cited URL is one of the retrieved chunks' URLs and fails the response to
  escalation if not.

Interface for this phase: a CLI chat loop (`python assistant/src/chat.py`) with the
banner from behavior rule 4. No web UI this phase.

## Evaluation

`assistant/eval/questions.json`, hand-written, committed, ~40 questions in four
labeled categories:

1. **In-scope factual** (~15): answerable from specific corpus pages; each carries
   the expected source URL(s).
2. **Navigation** (~10): "where do I find / how do I contact / when is" questions
   with expected URLs.
3. **Medical-advice traps** (~10): questions that MUST trigger the refusal template
   (treatment choices, dosing, "should my daughter...", symptom interpretation),
   including ones phrased gently enough that a naive system would answer.
4. **Out-of-scope** (~5): unrelated topics that must trigger escalation, not an
   answer.

Metrics, all deterministic on extractive mode:

- Retrieval: hit@3 and MRR against expected URLs for categories 1-2, reported per
  category.
- Behavior: 100% of category 3 must hit the refusal template and 100% of category 4
  must hit escalation. These two are TESTS, not just metrics: any miss fails pytest.
- If a generative-mode run is available (key present), report Ragas-style
  faithfulness and answer relevance on categories 1-2 in the findings as a
  supplementary table, clearly marked as run-dependent, with the grading model
  named. The automated suite must pass without any API key.

## Deliverables

1. `assistant/src/scrape.py`, `assistant/src/index.py`, `assistant/src/answer.py`,
   `assistant/src/chat.py`: scraper, chunker+BM25 index, answer/refusal/escalation
   logic, CLI.
2. `assistant/corpus_manifest.json` (committed), `assistant/corpus/` (gitignored).
3. `assistant/system_prompt.md` and the refusal/escalation templates as data files,
   not string literals buried in code.
4. `assistant/eval/questions.json` and `assistant/eval/run_eval.py` writing
   `docs/assistant/eval_results.csv`.
5. `docs/assistant_findings.md`: corpus stats (pages, chunks, scrape date), retrieval
   metrics per category, threshold tuning evidence, behavior test results, known
   failure modes with verbatim examples, and limitations. Repo prose style.
6. Tests: chunker properties (size bounds, URL propagation), manifest integrity
   (hashes match a freshly built corpus), retrieval hit@3 above a stated floor on
   category 1, categories 3 and 4 at 100%, no corpus text committed (test walks the
   git index), CLI answer path smoke test on a canned mini-corpus fixture committed
   under `model/tests`-style fixtures so CI needs no scrape.

## Out of scope

Web UI, embeddings/vector databases, conversation memory across turns, multilingual
support, any collection of user data, fine-tuning, and any change to the model/
pipeline side of the repo. If the Foundation's robots.txt or site structure blocks
the corpus, stop and report rather than substituting another organization's content.
