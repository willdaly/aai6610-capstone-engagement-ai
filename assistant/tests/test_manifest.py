"""Manifest integrity and the no-corpus-committed guarantee.

Two integrity checks:

- Against the committed real manifest and the local real corpus (skips when the corpus
  is absent, e.g. on CI, since the corpus is gitignored).
- A round-trip on the fake fixture: build a manifest from it, verify it passes, then
  tamper with a page and confirm verification catches the change.

Plus the hard repository-hygiene rule: no scraped corpus text is tracked by git.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scrape import extracted_text_for_page, verify_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PAGES_DIR = REPO_ROOT / "assistant" / "tests" / "fixtures" / "mini_corpus" / "pages"
REAL_CORPUS_PAGES_DIR = REPO_ROOT / "assistant" / "corpus" / "pages"
MANIFEST_PATH = REPO_ROOT / "assistant" / "corpus_manifest.json"


def _build_manifest_from(pages_dir: Path) -> dict:
    pages = []
    for path in sorted(pages_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        text = extracted_text_for_page(data)
        pages.append(
            {
                "url": data["url"],
                "title": data["title"],
                "http_status": data.get("http_status", 200),
                "fetch_timestamp": "fixture",
                "word_count": len(text.split()),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return {"pages": pages}


class TestFixtureRoundTrip:
    def test_freshly_built_manifest_verifies(self):
        manifest = _build_manifest_from(FIXTURE_PAGES_DIR)
        assert verify_manifest(manifest, FIXTURE_PAGES_DIR) == []

    def test_tampered_page_is_detected(self, tmp_path):
        # Copy the fixture, build a manifest, then edit one page's text.
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        for path in FIXTURE_PAGES_DIR.glob("*.json"):
            (pages_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = _build_manifest_from(pages_dir)

        victim = next(pages_dir.glob("*.json"))
        data = json.loads(victim.read_text(encoding="utf-8"))
        data["blocks"].append({"kind": "text", "text": "an added sentence not in the hash"})
        victim.write_text(json.dumps(data), encoding="utf-8")

        mismatches = verify_manifest(manifest, pages_dir)
        assert any(m["reason"] == "sha256 mismatch" for m in mismatches)

    def test_missing_page_is_detected(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        for path in FIXTURE_PAGES_DIR.glob("*.json"):
            (pages_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = _build_manifest_from(pages_dir)
        next(pages_dir.glob("*.json")).unlink()
        mismatches = verify_manifest(manifest, pages_dir)
        assert any(m["reason"] == "missing from corpus on disk" for m in mismatches)


class TestRealManifest:
    def test_real_corpus_matches_committed_manifest(self):
        if not REAL_CORPUS_PAGES_DIR.exists() or not any(REAL_CORPUS_PAGES_DIR.glob("*.json")):
            pytest.skip("Real corpus absent (gitignored). Run assistant/src/scrape.py.")
        if not MANIFEST_PATH.exists():
            pytest.skip("corpus_manifest.json absent. Run assistant/src/scrape.py.")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mismatches = verify_manifest(manifest, REAL_CORPUS_PAGES_DIR)
        assert mismatches == [], f"corpus does not match manifest: {mismatches}"


class TestNoCorpusTextCommitted:
    """The scraped corpus is copyrighted; git must not track any of it."""

    def test_git_does_not_track_corpus_pages(self):
        tracked = subprocess.run(
            ["git", "ls-files", "assistant/corpus"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert tracked == "", f"corpus files are tracked by git:\n{tracked}"

    def test_corpus_directory_is_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "assistant/corpus/pages/anything.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "assistant/corpus/ is not gitignored"
