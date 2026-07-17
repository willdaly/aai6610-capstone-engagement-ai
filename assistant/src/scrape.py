"""Scrape the Sturge-Weber Foundation public website into a local corpus.

This is the corpus builder for the family navigation assistant (proposal Areas 2 + 8).
Read ``docs/assistant_plan.md`` first; the scraping rules there are the contract and are
enforced here, not paraphrased:

- robots.txt is fetched and parsed before anything else. If it disallows a page we want,
  that page is skipped and recorded; the run does not work around a Disallow.
- Scope is public informational pages discovered from the sitemap, minus an explicit
  denylist of forms and data-collection pages (a registry sign-up and a contact-update
  form). No third-party hosts, no member areas.
- Politeness: a custom User-Agent that names the project and a contact address, one
  request per second, and conditional GETs (ETag / Last-Modified) so a re-run does not
  re-download pages that have not changed.

What lands on disk:

- ``assistant/corpus/pages/<slug>.json``  (gitignored): per page, the extracted text as
  an ordered list of heading/paragraph blocks plus fetch metadata. This is the
  Foundation's copyrighted content, so it is never committed.
- ``assistant/corpus/http_cache.json``    (gitignored): ETag / Last-Modified per URL, so
  a re-run can issue conditional requests.
- ``assistant/corpus_manifest.json``      (committed): for each page its URL, title,
  fetch timestamp, HTTP status, word count, and a SHA-256 of the extracted text, plus
  the robots.txt content and the scrape date. The manifest lets anyone rebuild the
  corpus with this script and verify they got the same text we did, without us
  redistributing that text.

Run:  ``python assistant/src/scrape.py``  (add ``--limit N`` for a quick smoke run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "assistant" / "corpus"
PAGES_DIR = CORPUS_DIR / "pages"
HTTP_CACHE_PATH = CORPUS_DIR / "http_cache.json"
MANIFEST_PATH = REPO_ROOT / "assistant" / "corpus_manifest.json"

HOST = "sturge-weber.org"
BASE_URL = f"https://{HOST}"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

# Names the project and gives a contact address, per the plan's politeness rule. The
# contact is the repo owner's academic address; this is an unofficial student project.
USER_AGENT = (
    "SWF-Capstone-Assistant/0.1 "
    "(AAI6610 academic prototype, not affiliated with the Foundation; "
    "contact: rwilliamdaly@gmail.com)"
)
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

# Transient server errors (the site returned blanket HTTP 500s during one scrape session)
# and connection errors get a few backed-off retries before a URL is recorded as failed.
# This is not politeness relief: the per-request delay still applies to every attempt.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5.0

# Redirects are followed manually so we can stay on the Foundation's own domain. The plan
# scopes the corpus to sturge-weber.org: a redirect that leaves the host (for example an
# event page pointing at a former event's own domain, which may since have lapsed) is
# recorded and skipped, never fetched. Same-host redirects (trailing slash, http->https)
# are followed up to a small limit.
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5

# Forms and data-collection pages. The plan's scope is informational pages only and rule
# 3 is "no data collection", so a registry sign-up and a contact-update form are out even
# though the sitemap lists them. Matched as URL-path substrings.
DENYLIST_SUBSTRINGS = (
    "/sws-registry",
    "/update-contact-information",
)

# Chrome inside the main content region: translate widget, site search boxes, social
# buttons, footer link blocks, the skip-to-content link. Removed before text extraction
# so navigation labels do not pollute the corpus. Matched against element class names.
CHROME_CLASS_RE = re.compile(
    r"(google-translate|search-\d|social-media|footer-|quicklinks|skip-to-main)"
)

_HEADING_SENTINEL = "\x00HEADING\x00"


@dataclass
class PageRecord:
    """One scraped page. ``extracted_text`` is the newline-joined block text."""

    url: str
    title: str
    http_status: int
    fetch_timestamp: str
    from_cache: bool
    blocks: list[dict]
    extracted_text: str
    word_count: int
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass
class ScrapeResult:
    pages: list[PageRecord] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def load_robots(session: requests.Session) -> tuple[str, urllib.robotparser.RobotFileParser]:
    """Fetch robots.txt and return its raw text and a parser.

    Raises on a non-200 so the run stops rather than guessing that scraping is allowed
    when we could not even read the policy.
    """
    resp = session.get(ROBOTS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise RuntimeError(
            f"robots.txt returned HTTP {resp.status_code}. Stopping: the plan requires a "
            "readable robots.txt before any scraping."
        )
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(resp.text.splitlines())
    return resp.text, parser


def discover_urls(session: requests.Session) -> list[str]:
    """Return in-scope page URLs from the sitemap, in sitemap order, deduplicated.

    Same host only, denylist removed. Callers still robots-check each URL before fetching.
    """
    resp = session.get(SITEMAP_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    # Sitemap namespace; loc elements are namespaced.
    locs = [el.text.strip() for el in root.iter() if el.tag.endswith("}loc") and el.text]

    seen: set[str] = set()
    urls: list[str] = []
    for url in locs:
        if HOST not in url:
            continue
        if any(sub in url for sub in DENYLIST_SUBSTRINGS):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _slug_for(url: str) -> str:
    """A filesystem-safe, stable slug for a URL's path."""
    path = re.sub(rf"^https?://{re.escape(HOST)}/?", "", url)
    path = path.rstrip("/") or "index"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path)


def extract(html: str, url: str) -> tuple[str, list[dict]]:
    """Extract the page title and an ordered list of content blocks.

    Returns ``(title, blocks)`` where each block is ``{"kind": "heading"|"text",
    "text": str}``. Navigation, header, footer, forms, scripts, and known chrome blocks
    are removed first. Headings are detected structurally (real ``h1``-``h6`` tags and
    short standalone ``strong``/``b`` runs, which is how this site marks its sections),
    so the chunker can group paragraphs under their heading.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else url

    main = (
        soup.find("main", id="main-content")
        or soup.find("main")
        or soup.find("div", class_="internal-content")
        or soup.find("article")
        or soup.body
        or soup
    )

    for tag in main.find_all(["nav", "header", "footer", "form", "script", "style", "noscript"]):
        tag.decompose()
    for tag in main.find_all(class_=CHROME_CLASS_RE):
        tag.decompose()

    # Mark headings with a sentinel so they survive the flatten-to-text step. Real
    # heading tags always count; a strong/bold run counts only when it is the whole of
    # its parent block and is short, which is how section labels appear on this site.
    for h in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = h.get_text(" ", strip=True)
        if text:
            h.string = f"{_HEADING_SENTINEL}{text}"
    for s in main.find_all(["strong", "b"]):
        text = s.get_text(" ", strip=True)
        if not text or len(text.split()) > 12:
            continue
        if s.parent is not None and s.parent.get_text(" ", strip=True) == text:
            s.string = f"{_HEADING_SENTINEL}{text}"

    blocks: list[dict] = []
    seen_heading: set[str] = set()
    for raw_line in main.get_text("\n", strip=True).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_HEADING_SENTINEL):
            text = line[len(_HEADING_SENTINEL):].strip()
            if not text:
                continue
            # A heading repeated verbatim (the page title echoed inside content) is noise.
            key = text.lower()
            if key in seen_heading:
                blocks.append({"kind": "text", "text": text})
            else:
                seen_heading.add(key)
                blocks.append({"kind": "heading", "text": text})
        else:
            # Drop sentinel remnants if a heading run was concatenated mid-line.
            line = line.replace(_HEADING_SENTINEL, "").strip()
            if line:
                blocks.append({"kind": "text", "text": line})

    return title, blocks


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _same_host(url: str) -> bool:
    """True if ``url`` is on sturge-weber.org (a leading www. counts as the same host)."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc == HOST


def _get_with_retries(
    session: requests.Session, url: str, headers: dict
) -> requests.Response:
    """A single polite GET (no auto-redirect), retried on transient 5xx and connection
    errors. Sleeps before every attempt, so the politeness delay applies to re-runs too.
    Raises the last exception if every attempt fails to connect.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            resp = session.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=False
            )
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        if resp.status_code >= 500 and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        return resp
    raise last_exc  # type: ignore[misc]


def fetch_page(
    session: requests.Session, url: str, cache: dict
) -> tuple[str, object]:
    """Fetch a page, following only same-host redirects.

    Returns a tagged result:

    - ``("cache", response)``   the page is unchanged since last run (HTTP 304); reuse the
      cached body.
    - ``("offhost", target)``   the URL redirects off sturge-weber.org; ``target`` is the
      off-host URL. The page is not fetched, per the plan's same-domain scope.
    - ``("response", response)`` a final, non-redirect response (200 or an error status).
    """
    entry = cache.get(url, {})
    headers = {}
    if entry.get("etag"):
        headers["If-None-Match"] = entry["etag"]
    if entry.get("last_modified"):
        headers["If-Modified-Since"] = entry["last_modified"]

    current = url
    resp = None
    for _hop in range(MAX_REDIRECTS + 1):
        resp = _get_with_retries(session, current, headers)

        if resp.status_code == 304 and current == url and entry.get("body") is not None:
            return "cache", resp

        if resp.status_code in REDIRECT_CODES:
            location = resp.headers.get("Location")
            if not location:
                return "response", resp
            target = urljoin(current, location)
            if not _same_host(target):
                return "offhost", target
            # Same-host redirect: follow it as a plain GET (conditional headers only
            # applied to the original URL).
            current = target
            headers = {}
            continue

        return "response", resp

    return "response", resp


def scrape(
    session: requests.Session,
    urls: list[str],
    robots: urllib.robotparser.RobotFileParser,
    cache: dict,
) -> ScrapeResult:
    result = ScrapeResult()
    for url in urls:
        if not robots.can_fetch(USER_AGENT, url):
            result.excluded.append({"url": url, "reason": "robots.txt disallow"})
            continue

        try:
            kind, payload = fetch_page(session, url, cache)
        except requests.RequestException as exc:
            result.failures.append({"url": url, "error": str(exc)})
            continue

        if kind == "offhost":
            result.excluded.append({"url": url, "reason": f"off-host redirect to {payload}"})
            print(f"  [skip] off-host redirect -> {payload}  ({url})", file=sys.stderr)
            continue

        from_cache = kind == "cache"
        if from_cache:
            html = cache[url]["body"]
            status = 200
            etag = cache[url].get("etag")
            last_modified = cache[url].get("last_modified")
        else:
            resp = payload  # type: ignore[assignment]
            if resp.status_code == 200:
                html = resp.text
                status = 200
                etag = resp.headers.get("ETag")
                last_modified = resp.headers.get("Last-Modified")
                cache[url] = {"etag": etag, "last_modified": last_modified, "body": html}
            else:
                result.failures.append({"url": url, "http_status": resp.status_code})
                continue

        title, blocks = extract(html, url)
        extracted_text = "\n".join(b["text"] for b in blocks)
        record = PageRecord(
            url=url,
            title=title,
            http_status=status,
            fetch_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            from_cache=from_cache,
            blocks=blocks,
            extracted_text=extracted_text,
            word_count=len(extracted_text.split()),
            sha256=_sha256(extracted_text),
            etag=etag,
            last_modified=last_modified,
        )
        result.pages.append(record)
        marker = "cache" if from_cache else "get"
        print(f"  [{marker}] {record.word_count:>5} words  {url}", file=sys.stderr)
    return result


def write_corpus(result: ScrapeResult, cache: dict) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for record in result.pages:
        path = PAGES_DIR / f"{_slug_for(record.url)}.json"
        payload = {
            "url": record.url,
            "title": record.title,
            "http_status": record.http_status,
            "fetch_timestamp": record.fetch_timestamp,
            "word_count": record.word_count,
            "sha256": record.sha256,
            "blocks": record.blocks,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    HTTP_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def write_manifest(result: ScrapeResult, robots_text: str) -> dict:
    scrape_date = datetime.now(timezone.utc).date().isoformat()
    pages = [
        {
            "url": r.url,
            "title": r.title,
            "http_status": r.http_status,
            "fetch_timestamp": r.fetch_timestamp,
            "word_count": r.word_count,
            "sha256": r.sha256,
        }
        for r in sorted(result.pages, key=lambda r: r.url)
    ]
    manifest = {
        "scrape": {
            "host": HOST,
            "scrape_date": scrape_date,
            "user_agent": USER_AGENT,
            "request_delay_seconds": REQUEST_DELAY_SECONDS,
            "sitemap_url": SITEMAP_URL,
            "robots_txt_url": ROBOTS_URL,
            "robots_txt_checked": scrape_date,
            "robots_txt_content": robots_text.strip(),
            "denylist_substrings": list(DENYLIST_SUBSTRINGS),
        },
        "pages": pages,
        "failures": sorted(result.failures, key=lambda f: f["url"]),
        "excluded": sorted(result.excluded, key=lambda e: e["url"]),
        "summary": {
            "page_count": len(pages),
            "total_words": sum(p["word_count"] for p in pages),
            "failure_count": len(result.failures),
            "excluded_count": len(result.excluded),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def extracted_text_for_page(page_json: dict) -> str:
    """The canonical text a manifest hash is taken over: block texts joined by newline.

    Kept in one place so the scraper and the integrity check can never disagree about
    what the SHA-256 covers.
    """
    return "\n".join(b["text"] for b in page_json["blocks"])


def verify_manifest(manifest: dict, pages_dir: Path | str) -> list[dict]:
    """Recompute each page's text hash from disk and compare to the manifest.

    Returns a list of mismatch records (empty when the corpus on disk matches the
    committed manifest exactly). A mismatch means the local corpus differs from what the
    manifest claims: a stale manifest, an edited page, or a changed source page.
    """
    pages_dir = Path(pages_dir)
    by_url = {p["url"]: p for p in manifest["pages"]}
    on_disk: dict[str, dict] = {}
    for path in pages_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        on_disk[data["url"]] = data

    mismatches: list[dict] = []
    for url, entry in by_url.items():
        page = on_disk.get(url)
        if page is None:
            mismatches.append({"url": url, "reason": "missing from corpus on disk"})
            continue
        actual = _sha256(extracted_text_for_page(page))
        if actual != entry["sha256"]:
            mismatches.append(
                {"url": url, "reason": "sha256 mismatch",
                 "manifest": entry["sha256"], "actual": actual}
            )
    for url in on_disk:
        if url not in by_url:
            mismatches.append({"url": url, "reason": "on disk but not in manifest"})
    return mismatches


def load_cache() -> dict:
    if HTTP_CACHE_PATH.exists():
        return json.loads(HTTP_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def run(limit: int | None = None) -> dict:
    session = build_session()

    print("Fetching robots.txt ...", file=sys.stderr)
    robots_text, robots = load_robots(session)

    if not robots.can_fetch(USER_AGENT, BASE_URL + "/"):
        raise RuntimeError(
            "robots.txt disallows the site root for our User-Agent. Stopping per the plan.\n"
            f"--- robots.txt ---\n{robots_text}"
        )

    urls = discover_urls(session)
    if limit is not None:
        urls = urls[:limit]
    print(f"Discovered {len(urls)} in-scope URLs from the sitemap.", file=sys.stderr)

    cache = load_cache()
    result = scrape(session, urls, robots, cache)
    write_corpus(result, cache)
    manifest = write_manifest(result, robots_text)

    summary = manifest["summary"]
    print(
        f"\nScrape complete: {summary['page_count']} pages, "
        f"{summary['total_words']} words, "
        f"{summary['failure_count']} failures, "
        f"{summary['excluded_count']} excluded (robots or off-host redirect).",
        file=sys.stderr,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the SWF public site into a corpus.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Scrape only the first N URLs (smoke run)."
    )
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
