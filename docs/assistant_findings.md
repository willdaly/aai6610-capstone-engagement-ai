# Assistant Findings — Family Navigation Prototype

What this build is, what it does, and where it falls short. It follows
`docs/assistant_plan.md`, which is the contract. Every number here comes from a run
produced this session against the corpus described below; nothing is estimated. The
corpus itself is gitignored (it is the Foundation's copyrighted text); it is reproducible
from the scraper and verifiable against `assistant/corpus_manifest.json`.

This is an unofficial academic prototype. It is not affiliated with or endorsed by the
Sturge-Weber Foundation, and it does not use any Foundation-provided data.

## Corpus

- Source: `sturge-weber.org`, public pages discovered from the sitemap.
- Scrape date: 2026-07-17.
- robots.txt (checked 2026-07-17, recorded in the manifest): `User-agent: *`, `Allow: /`,
  `Disallow:` (empty). Nothing we wanted is disallowed, so scraping proceeded. A registry
  sign-up and a contact-update form were excluded by our own denylist (they are data
  collection, not information), not by robots.
- Politeness: a custom User-Agent naming the project and a contact address, one request
  per second, conditional GETs (ETag / Last-Modified) so a re-run does not re-download
  unchanged pages, and retries on transient 5xx. The site returned blanket HTTP 500s for
  part of this session (a server-side outage, not a block); the scrape ran once it
  recovered.
- Result: 108 pages, 46,879 words. One sitemap link 404s (`/events/past-events/`) and is
  recorded as a failure. Words per page range 6 to 3,604 (median 295).
- Chunking: 490 chunks, 1 to 367 words each (median 53), none over the 500-word bound. On
  this site most pages have short sections, so most chunks are a single section under the
  200-word target rather than packed to it; the size test covers the packing and splitting
  paths on synthetic pages where they actually occur.

### Corpus quality notes

- Five pages hold fewer than 40 words (video/image galleries and a near-empty scholarship
  stub). They stay in the corpus but carry almost no retrievable text.
- One page, `/events/mylas-mission-5k-and-jeeputv-ride.html`, currently serves injected
  casino-spam content (title begins "Mostbet Azərbaycan 2026: Kazino..."). This is what
  the live site returns; the page appears compromised. It is left in the corpus as scraped
  and flagged here. Its vocabulary does not overlap family questions, so it does not
  surface in the eval, but it is a real data-quality signal worth reporting to the
  Foundation.

## Retrieval

BM25 over the chunks (`rank_bm25`), deterministic and offline. Metrics are over distinct
source pages (chunks from one page are collapsed, so a single page cannot fill the top
three with its own chunks).

| Category | n | hit@3 | MRR |
|---|---|---|---|
| 1 — in-scope factual | 15 | 0.80 | 0.68 |
| 2 — navigation | 11 | 0.91 | 0.81 |

These are moderate, not strong, and the plan anticipated that: BM25 stays, and the gaps
are reported rather than papered over with embeddings. The misses have a common cause,
described under failure modes.

## Confidence threshold and its tuning

A question is answered only if a retrieved passage is actually about it; otherwise it
escalates. Three signals gate this, all measured on the eval set:

- **best_overlap**: the best fraction of the question's content words covered by any one of
  the top-3 chunks. This is the signal that separates in-scope from out-of-scope.
- **coverage**: fraction of content words present anywhere in the corpus (a cheap first
  filter for topics the corpus never mentions).
- **top_score**: a raw BM25 floor, so a query that matches almost nothing escalates.

Distributions on the eval set (in-scope = categories 1-2, out-of-scope = category 4):

| signal | in-scope min / median | out-of-scope max |
|---|---|---|
| best_overlap | 0.40 / 1.00 | 0.33 |
| coverage | 0.83 / 1.00 | 1.00 |
| top_score | 6.27 / 11.90 | 7.97 |

best_overlap separates cleanly: in-scope never drops below 0.40, out-of-scope never rises
above 0.33. The threshold is set at **0.35**, in that gap. coverage and top_score overlap
between the two groups and cannot separate them alone (an out-of-scope question can share
common words with some page, and BM25 will still score it), which is why best_overlap
carries the decision; coverage (0.5) and top_score (1.0) act only as floors. An earlier
version used the top-1 chunk's overlap and wrongly escalated two in-scope questions whose
best passage was not rank 1 (scholarships, caregiver chat); taking the best over the top 3
fixed both with no out-of-scope leakage.

## Behavior (the graded rules)

These are enforced as tests (`assistant/tests/test_behavior.py`); a miss fails the build.

| Category | n | requirement | result |
|---|---|---|---|
| 3 — medical-advice traps | 11 | 100% hit the refusal template | 100% |
| 4 — out-of-scope | 6 | 100% escalate | 100% |

The medical-advice check runs before retrieval and fires even when the corpus holds
related educational content, so an educational passage about seizures or glaucoma can
never be turned into advice for a specific child. The three category-3 questions on the
hardest education/advice line are flagged in `questions.json` for hand review
(`review_flag`): laser-treatment timing, an ER judgment during a seizure, a personal
prognosis question, and specialist triage. All currently refuse, which is the intended
conservative behavior; they are flagged because a reasonable reviewer might weigh the
wording differently.

## Generative mode

Implemented behind `ANTHROPIC_API_KEY` and off by default. Retrieved passages plus the
committed system prompt (`assistant/system_prompt.md`) produce a synthesized answer, then
a citation post-check pulls every URL from the answer and demotes it to the escalation
response if it cites a URL that was not retrieved, or cites nothing at all. The check is
unit-tested with a stubbed client (`assistant/tests/test_generative.py`), so it runs with
no key and no network.

No API key was present this session, so generative mode was not run against the corpus and
no reference-free faithfulness or answer-relevance (Ragas-style) numbers are reported here;
per the plan and the no-fabrication rule, those are produced only from an actual run. When
run, the grading model is `claude-sonnet-5`, and the table would be marked run-dependent.
The full automated suite passes with no key present.

## Known failure modes (verbatim)

**1. Navigation queries lose to pages that merely repeat the query word.** "How do I
contact the Sturge-Weber Foundation?" retrieves the mobile-app privacy policy and the
volunteer page, not the contact page, because "contact" appears many times in the privacy
policy while the site-name words ("sturge", "weber", "foundation") appear on nearly every
page and so carry little discriminating weight. The answer it gives is not wrong (the
privacy policy does list the real contact email) but the citation is the wrong page. The
refusal and escalation paths are unaffected: they find the contact page directly by URL,
not by retrieval.

**2. Short factual questions miss their page when a near-synonym page scores higher.**

- "What are the different types of Sturge-Weber syndrome?" returns
  `neurological-conditions`, `opthalmalogical-information`, `understanding-sturge-weber`;
  the intended `types-of-sturge-weber-syndrome` is just outside the top 3.
- "What research does the Foundation support?" returns `grants-for-professionals` and two
  `who-we-are` pages instead of the patient research pages.
- "Are there books about Sturge-Weber syndrome and port-wine birthmarks?" returns research
  and education pages; the books page does not make the top 3.

All three are BM25 vocabulary-overlap effects, not bugs, and are the reason category-1
hit@3 is 0.80 rather than higher.

**3. The medical classifier is rule-based and English-only.** It keys on a specific person
plus a clinical cue (symptom, treatment, dosing, imaging), or on direct advice patterns.
It will over-trigger on some phrasings (asking generally about a medication by name near a
family word) and could be evaded by unusual phrasing or another language. Over-triggering
is safe here (it escalates); under-triggering is the risk, so the rules err toward
refusing, and the category-3 test guards the known traps.

## Limitations

- BM25 has no semantics: it matches words, not meaning, so synonyms and paraphrases hurt
  retrieval, and the ubiquitous site-name words dilute ranking. Embeddings were out of
  scope this phase by design.
- Answers are extractive quotes, so they can read as terse or start mid-list; they are
  faithful to the page but not polished. Generative mode addresses this but needs a key.
- The corpus is a snapshot dated 2026-07-17. The site changes, and one page is currently
  serving spam. Answers can go stale; the banner says so.
- Scores inform outreach and navigation only. This tool never gates any family's access to
  services, and it never gives medical advice.

## Reproducing this

1. `python assistant/src/scrape.py` rebuilds the corpus under `assistant/corpus/`
   (gitignored). The committed `assistant/corpus_manifest.json` records each page's URL,
   title, fetch time, HTTP status, word count, and a SHA-256 of the extracted text; the
   manifest-integrity test checks a freshly built corpus against those hashes.
2. `python assistant/eval/run_eval.py` regenerates `docs/assistant/eval_results.csv` and
   prints the tables above.
3. `pytest` runs the full suite (model plus assistant) with no API key.
