"""CLI chat loop for the family navigation assistant.

Run:  ``python assistant/src/chat.py``            (extractive, offline, default)
      ``python assistant/src/chat.py --generative``  (needs ANTHROPIC_API_KEY)

The banner states the honest-identity and no-data-collection rules (behavior rules 3-4).
The assistant answers only from the scraped corpus, refuses medical-advice questions, and
escalates anything it cannot ground. Nothing typed here is stored.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from answer import AnswerResult, Assistant
from index import CORPUS_PAGES_DIR, build_index

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest_scrape_date() -> str:
    import json

    manifest = REPO_ROOT / "assistant" / "corpus_manifest.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))["scrape"]["scrape_date"]
        except (KeyError, ValueError):
            pass
    return "unknown"


def banner(generative: bool) -> str:
    scrape_date = _manifest_scrape_date()
    mode = "generative (Claude API)" if generative else "extractive (offline)"
    return (
        "=" * 74 + "\n"
        "Family Navigation Assistant  (unofficial academic prototype)\n"
        + "-" * 74 + "\n"
        "This is a student capstone project. It is NOT affiliated with or endorsed by\n"
        "the Sturge-Weber Foundation. Answers come only from the Foundation's public\n"
        f"website as scraped on {scrape_date}, and may be out of date.\n"
        "\n"
        "It does not give medical advice. For anything about a specific person's care,\n"
        "please talk with your own clinical team.\n"
        "\n"
        "No data collection: your questions are not stored or logged anywhere.\n"
        f"\nMode: {mode}.  Type a question, or 'quit' to exit.\n"
        + "=" * 74
    )


def format_result(result: AnswerResult) -> str:
    label = {
        "answer": "ANSWER",
        "refusal": "I can't answer that",
        "escalation": "Contact the Foundation",
    }[result.kind]
    lines = [f"[{label}]", "", result.text]
    if result.citations:
        lines += ["", "Sources:"]
        seen = set()
        for c in result.citations:
            if c.url in seen:
                continue
            seen.add(c.url)
            lines.append(f"  - {c.title} ({c.url})")
    return "\n".join(lines)


def build_assistant() -> Assistant:
    if not CORPUS_PAGES_DIR.exists() or not any(CORPUS_PAGES_DIR.glob("*.json")):
        sys.exit(
            "No corpus found. Build it first:\n  python assistant/src/scrape.py\n"
            f"(expected page files under {CORPUS_PAGES_DIR})"
        )
    return Assistant(build_index(CORPUS_PAGES_DIR))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Family navigation assistant CLI.")
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Use Claude to synthesize answers from retrieved passages (needs ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Answer a single question and exit (for scripting/demos).",
    )
    args = parser.parse_args(argv)

    generative = args.generative
    if generative and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "--generative needs ANTHROPIC_API_KEY in the environment. "
            "Falling back to extractive mode.\n",
            file=sys.stderr,
        )
        generative = False

    assistant = build_assistant()

    if args.ask:
        print(format_result(assistant.answer(args.ask, generative=generative)))
        return

    print(banner(generative))
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTake care.")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Take care.")
            break
        print()
        print(format_result(assistant.answer(query, generative=generative)))


if __name__ == "__main__":
    main()
