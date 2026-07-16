# assistant/

Family navigation assistant prototype (brief Areas 2 + 8): a retrieval-grounded Q&A
assistant over the Sturge-Weber Foundation's public website content, escalating to staff
for anything out of scope.

Nothing is built here yet. This directory is a placeholder for:

- `corpus/`: chunked public pages scraped from sturge-weber.org, with a recorded scrape
  date. Public pages only, and respect robots.txt.
- A scraper script that produces that corpus.
- The RAG prototype itself.
- A small hand-written evaluation question set, committed to the repo, scored on
  faithfulness and answer relevance (Ragas-style reference-free metrics).

Constraints, from CLAUDE.md:

- Answers ground in retrieved passages. Out-of-scope questions get a "contact the
  Foundation" response rather than a guess.
- This is a support tool for families of patients, including children. Tone is plain,
  calm, and careful.
- The assistant never gives medical advice. It points to Foundation resources.
- No collection of personal data.
