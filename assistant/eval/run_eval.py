"""Evaluate the assistant on the committed question set and write eval_results.csv.

Deterministic, extractive-mode metrics (no API key needed), per docs/assistant_plan.md:

- Retrieval (categories 1-2): hit@3 and MRR against expected URLs, measured over distinct
  source *pages* (chunks from one page are collapsed, so a page cannot fill the top-3 with
  its own chunks). Reported per category.
- Behavior (categories 3-4): the fraction hitting the medical-advice refusal (cat 3) and
  the escalation (cat 4). The plan requires 100% on both; test_behavior.py enforces it.
- Threshold tuning evidence: the confidence signals (top-chunk overlap, corpus coverage,
  BM25 top score) for in-scope (cat 1-2) vs out-of-scope (cat 4), so the escalation
  thresholds in answer.py are justified by the split rather than asserted.

Writes docs/assistant/eval_results.csv (per-question) and prints a summary.

With ANTHROPIC_API_KEY set and --generative, additionally computes reference-free
faithfulness and answer-relevance on categories 1-2 (see run_generative_eval in step 5's
findings; the default run here never needs a key).

Run:  python assistant/eval/run_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "assistant" / "src"))

from answer import Assistant  # noqa: E402
from index import CORPUS_PAGES_DIR, build_index  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "assistant" / "eval" / "questions.json"
RESULTS_PATH = REPO_ROOT / "docs" / "assistant" / "eval_results.csv"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def ranked_pages(index, query: str, k: int = 10) -> list[str]:
    """Distinct source-page URLs in rank order (chunks collapsed to their page)."""
    pages: list[str] = []
    for chunk, _score in index.search(query, k=40):
        if chunk.url not in pages:
            pages.append(chunk.url)
        if len(pages) >= k:
            break
    return pages


def evaluate(index, assistant: Assistant, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        query = q["question"]
        result = assistant.answer(query)
        top_pages = ranked_pages(index, query)
        expected = q.get("expected_urls", [])

        hit3 = ""
        rr = ""
        if q["category"] in (1, 2):
            hit3 = int(any(u in top_pages[:3] for u in expected))
            rr = 0.0
            for i, url in enumerate(top_pages):
                if url in expected:
                    rr = round(1.0 / (i + 1), 4)
                    break

        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": query,
                "kind": result.kind,
                "expected_urls": " ".join(expected),
                "top3_pages": " ".join(top_pages[:3]),
                "hit@3": hit3,
                "reciprocal_rank": rr,
                "top_overlap": round(
                    index.best_overlap(query, [c for c, _ in index.search(query, k=3)]), 4
                ),
                "coverage": round(index.query_coverage(query), 4),
                "top_score": round(result.top_score if result.kind != "refusal" else 0.0, 4),
                "review_flag": q.get("review_flag", ""),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    def cat(n):
        return [r for r in rows if r["category"] == n]

    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else 0.0

    summary = {}
    for n in (1, 2):
        rs = cat(n)
        summary[f"cat{n}_hit@3"] = mean([r["hit@3"] for r in rs])
        summary[f"cat{n}_mrr"] = mean([r["reciprocal_rank"] for r in rs])
        summary[f"cat{n}_n"] = len(rs)
    summary["cat3_refusal_rate"] = mean([int(r["kind"] == "refusal") for r in cat(3)])
    summary["cat3_n"] = len(cat(3))
    summary["cat4_escalation_rate"] = mean([int(r["kind"] == "escalation") for r in cat(4)])
    summary["cat4_n"] = len(cat(4))
    return summary


def tuning_evidence(rows: list[dict]) -> dict:
    """Confidence-signal separation between in-scope (cat 1-2) and out-of-scope (cat 4)."""
    def stats(rs, key):
        xs = [r[key] for r in rs]
        return {"min": round(min(xs), 3), "median": round(statistics.median(xs), 3),
                "max": round(max(xs), 3)} if xs else {}

    in_scope = [r for r in rows if r["category"] in (1, 2)]
    oos = [r for r in rows if r["category"] == 4]
    return {
        "in_scope": {k: stats(in_scope, k) for k in ("top_overlap", "coverage", "top_score")},
        "out_of_scope": {k: stats(oos, k) for k in ("top_overlap", "coverage", "top_score")},
    }


def write_csv(rows: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the assistant.")
    parser.parse_args()

    if not CORPUS_PAGES_DIR.exists() or not any(CORPUS_PAGES_DIR.glob("*.json")):
        sys.exit("No corpus found. Run: python assistant/src/scrape.py")

    index = build_index(CORPUS_PAGES_DIR)
    assistant = Assistant(index)
    questions = load_questions()
    rows = evaluate(index, assistant, questions)
    write_csv(rows)

    summary = summarize(rows)
    tuning = tuning_evidence(rows)

    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}  ({len(rows)} questions)\n")
    print("Retrieval (over distinct pages):")
    print(f"  Category 1 (in-scope factual, n={summary['cat1_n']}): "
          f"hit@3={summary['cat1_hit@3']:.2f}  MRR={summary['cat1_mrr']:.2f}")
    print(f"  Category 2 (navigation,       n={summary['cat2_n']}): "
          f"hit@3={summary['cat2_hit@3']:.2f}  MRR={summary['cat2_mrr']:.2f}")
    print("\nBehavior (must be 1.00):")
    print(f"  Category 3 (medical refusal,  n={summary['cat3_n']}): "
          f"refusal rate={summary['cat3_refusal_rate']:.2f}")
    print(f"  Category 4 (escalation,       n={summary['cat4_n']}): "
          f"escalation rate={summary['cat4_escalation_rate']:.2f}")
    print("\nThreshold tuning evidence (confidence signals):")
    for scope, sig in tuning.items():
        print(f"  {scope}:")
        for name, s in sig.items():
            print(f"    {name:11} min={s['min']:.2f} median={s['median']:.2f} max={s['max']:.2f}")


if __name__ == "__main__":
    main()
