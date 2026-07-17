# assistant/

Family navigation assistant prototype (brief Areas 2 + 8): a retrieval-grounded Q&A
assistant over the Sturge-Weber Foundation's public website content, escalating to staff
for anything out of scope. `docs/assistant_plan.md` is the contract; `docs/assistant_findings.md`
reports what was built and how it performs.

This is an unofficial academic prototype. It is not affiliated with or endorsed by the
Foundation, and it uses no Foundation-provided data.

## Layout

- `src/scrape.py`: checks robots.txt, scrapes public pages from the sitemap politely
  (custom User-Agent, 1 request/second, conditional-GET caching), and extracts main
  content into heading/paragraph blocks.
- `src/index.py`: chunks the corpus (200-500 words, grouped by heading, source URL on
  every chunk) and builds a BM25 index.
- `src/answer.py`: routes each question to a grounded answer, the medical-advice refusal,
  or an escalation. Extractive by default; generative mode is optional and gated on
  `ANTHROPIC_API_KEY`, with a citation post-check.
- `src/chat.py`: the CLI, with the honest-identity and no-data-collection banner.
- `templates/`, `system_prompt.md`: the refusal/escalation templates and the generative
  system prompt, as committed data files.
- `eval/`: the hand-written question set and `run_eval.py`.
- `corpus/`: the scraped pages. Gitignored (the Foundation's copyrighted text); rebuild
  with the scraper. Only `corpus_manifest.json` (URLs, titles, hashes) is committed.
- `tests/`: chunker, retrieval, manifest integrity, behavior, CLI, and generative tests,
  with a fake mini-corpus fixture so the suite needs no scrape and no API key.

## Run

```bash
python assistant/src/scrape.py           # build the corpus (network)
python assistant/src/chat.py             # extractive CLI
python assistant/src/chat.py --generative  # needs ANTHROPIC_API_KEY
python assistant/eval/run_eval.py        # metrics -> docs/assistant/eval_results.csv
pytest                                   # full suite, no key needed
```

## Constraints, from CLAUDE.md and the plan

- Answers ground in retrieved passages. Out-of-scope questions get a "contact the
  Foundation" response rather than a guess.
- This is a support tool for families of patients, including children. Tone is plain,
  calm, and careful.
- The assistant never gives medical advice. It points to Foundation resources.
- No collection of personal data.
