# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1.3: EBA Regulatory Publications Scraper
# MAGIC
# MAGIC Builds a local knowledge base of EBA publications for a regulatory reporting AI agent.
# MAGIC
# MAGIC **Sections scraped:**
# MAGIC - Press releases filtered to supervisory-reporting topics (2025–2026)
# MAGIC - `regulation-and-policy/supervisory-reporting`
# MAGIC - `risk-and-data-analysis/reporting/reporting-frameworks`
# MAGIC
# MAGIC **CLI usage (standalone):**
# MAGIC ```bash
# MAGIC python 1.3_eba_data_scraper.py --section all --output eba_knowledge_base
# MAGIC python 1.3_eba_data_scraper.py --section press --dry-run
# MAGIC ```

# COMMAND ----------

import argparse
import json
import logging
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# COMMAND ----------

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

BASE_URL = "https://www.eba.europa.eu"
PRESS_RELEASES_URL = f"{BASE_URL}/publications-and-media/press-releases"
SUPERVISORY_REPORTING_URL = f"{BASE_URL}/regulation-and-policy/supervisory-reporting"
REPORTING_FRAMEWORKS_URL = (
    f"{BASE_URL}/risk-and-data-analysis/reporting/reporting-frameworks"
)

RELEVANCE_KEYWORDS: list[str] = [
    "COREP",
    "FINREP",
    "DPM",
    "XBRL",
    "ITS on reporting",
    "CRR3",
    "Basel III",
    "Pillar 3",
    "MREL",
    "supervisory reporting",
    "reporting framework",
    "validation rules",
    "benchmarking",
    "operational risk reporting",
    "credit risk reporting",
    "market risk reporting",
]

# Default target years — override via --years CLI arg for historic backfills.
TARGET_YEARS: frozenset[int] = frozenset({2026})

# All 24 EU official language codes (ISO 639-1).
_EU_LANG_CODES: frozenset[str] = frozenset({
    "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR",
    "GA", "HR", "HU", "IT", "LT", "LV", "MT", "NL", "PL", "PT",
    "RO", "SK", "SL", "SV",
})
# Only download files in these languages (plus generic files with no language code).
_ACCEPT_LANG_CODES: frozenset[str] = frozenset({"EN", "NL"})
# Detect a 2-letter language code that appears as a word-boundary token in a filename stem.
_LANG_IN_STEM: re.Pattern = re.compile(r"(?<=[_\-\(])([A-Z]{2})(?=[)_\-.]|$)")

# Polite-scraping delays
PAGE_DELAY: float = 1.5   # seconds between HTML page requests
PDF_DELAY: float = 2.0    # seconds between file downloads

MAX_RETRIES: int = 3
BACKOFF_BASE: int = 2     # exponential back-off multiplier
MAX_RELATED_ZIP_MB: int = 8
MAX_RELATED_ZIP_BYTES: int = MAX_RELATED_ZIP_MB * 1024 * 1024

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EBA-KB-Scraper/1.0; regulatory research bot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## HTTP Utilities

# COMMAND ----------

def fetch(
    url: str,
    stream: bool = False,
    dry_run: bool = False,
) -> Optional[requests.Response]:
    """GET with retry + exponential back-off. Returns None on permanent failure."""
    if dry_run:
        logger.info(f"[DRY-RUN] Would fetch: {url}")
        return None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, stream=stream, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {wait}s …")
                time.sleep(wait)
    logger.error(f"Giving up on {url} after {MAX_RETRIES} attempts")
    return None


def fetch_content_length(url: str) -> Optional[int]:
    """Best-effort HEAD request for content length. Returns None when unknown."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.head(
                url,
                headers=HEADERS,
                allow_redirects=True,
                timeout=20,
            )
            resp.raise_for_status()
            raw = resp.headers.get("Content-Length")
            if raw and raw.isdigit():
                return int(raw)
            return None
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            logger.debug(f"HEAD attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    return None


def slugify(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe slug (max 24 chars)."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "-", text).strip("-")[:24]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp without depending on notebook global imports."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def safe_filename(url: str) -> str:
    """Derive a safe, Windows-compatible filename from the last URL path segment."""
    from urllib.parse import unquote
    name = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    if not name:
        return f"file_{abs(hash(url)) % 0xFFFFFF:06x}"
    name = re.sub(r"\s+", "_", name.strip())
    # Strip characters illegal on Windows
    name = re.sub(r'[<>:"\\|?*]', "_", name)
    # Truncate long stems to keep total path under Windows 260-char limit
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        name = stem[:80] + "." + ext
    else:
        name = name[:80]
    return name

# COMMAND ----------

# MAGIC %md
# MAGIC ## File Downloader

# COMMAND ----------

def build_download_path(dest_dir: Path, filename: str, max_path_length: int = 220) -> Path:
    """Return a path guaranteed to stay under a conservative Windows path limit."""
    dest = dest_dir / filename
    if len(str(dest.resolve(strict=False))) <= max_path_length:
        return dest

    suffix = Path(filename).suffix
    stem = Path(filename).stem
    digest = f"{abs(hash(filename)) % 0xFFFFFF:06x}"

    # Trim progressively until path fits.
    for keep in (48, 32, 24, 16, 12, 8):
        trimmed_stem = stem[:keep].rstrip("._-")
        candidate = dest_dir / f"{trimmed_stem}_{digest}{suffix}"
        if len(str(candidate.resolve(strict=False))) <= max_path_length:
            return candidate

    # Final fallback for extremely deep paths.
    return dest_dir / f"f_{digest}{suffix}"


def download_file(
    url: str,
    dest_dir: Path,
    dry_run: bool = False,
    max_bytes: Optional[int] = None,
) -> Optional[str]:
    """
    Download a document (PDF, ZIP, XLSX, XML …) to dest_dir.
    Skips files that already exist (idempotent).
    Returns the saved filename, or None on failure.
    """
    filename = safe_filename(url)

    if not is_language_acceptable(url):
        lang = _detect_file_language(url)
        logger.info(f"  [LANG-SKIP] {lang} file skipped: {filename}")
        return None

    dest = build_download_path(dest_dir, filename)
    filename = dest.name

    if dest.exists():
        logger.info(f"  [SKIP] Already present: {filename}")
        return filename

    if dry_run:
        logger.info(f"  [DRY-RUN] Would download: {url} → {filename}")
        return filename

    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = fetch(url, stream=True)
    if resp is None:
        return None

    overflow = False
    written = 0
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65_536):
            if not chunk:
                continue
            if max_bytes is not None and (written + len(chunk)) > max_bytes:
                overflow = True
                break
            fh.write(chunk)
            written += len(chunk)

    if overflow:
        dest.unlink(missing_ok=True)
        max_kb = max_bytes // 1024 if max_bytes else 0
        logger.info(f"  [ZIP-SKIP] Exceeded size cap ({max_kb} KB): {filename}")
        return None

    size_kb = dest.stat().st_size // 1024
    logger.info(f"  [FILE] Saved: {filename} ({size_kb} KB)")
    time.sleep(PDF_DELAY)
    return filename


def extract_zip_file(zip_path: Path, target_dir: Path, delete_zip: bool = False) -> list[str]:
    """
    Extract a ZIP into target_dir and return extracted file paths relative to target_dir.
    Optionally delete the original ZIP after successful extraction.
    """
    if not zip_path.exists():
        return []

    extracted_files: list[str] = []
    extraction_succeeded = False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Block path traversal attempts inside ZIP archives.
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                zf.extract(member, path=target_dir)
                if not member.is_dir():
                    extracted_rel_path = member_path.as_posix()
                    extracted_abs_path = target_dir / member_path
                    if not is_filename_language_acceptable(extracted_abs_path.name):
                        lang = _detect_language_from_name(extracted_abs_path.name)
                        extracted_abs_path.unlink(missing_ok=True)
                        logger.info(
                            f"  [LANG-SKIP] Extracted {lang} file removed: {extracted_rel_path}"
                        )
                        continue
                    extracted_files.append(extracted_rel_path)
        extraction_succeeded = True
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning(f"  [ZIP-EXTRACT] Failed for {zip_path.name}: {exc}")
        return []

    if delete_zip and extraction_succeeded:
        try:
            zip_path.unlink(missing_ok=True)
            logger.info(f"  [ZIP-CLEANUP] Deleted archive after extract: {zip_path.name}")
        except OSError as exc:
            logger.warning(f"  [ZIP-CLEANUP] Could not delete {zip_path.name}: {exc}")

    logger.info(f"  [ZIP-EXTRACT] Extracted {len(extracted_files)} files from {zip_path.name}")
    return extracted_files

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relevance Filter

# COMMAND ----------

_KW_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(kw) for kw in RELEVANCE_KEYWORDS),
    flags=re.IGNORECASE,
)


def is_relevant(text: str) -> bool:
    """Return True if text contains at least one relevance keyword."""
    return bool(_KW_PATTERN.search(text))


def _detect_language_from_name(name: str) -> Optional[str]:
    """Return ISO 639-1 language code from a filename-like token, or None."""
    stem = name.upper()
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]

    # Common EBA pattern: filename is exactly the language code (e.g., bg.zip).
    if len(stem) == 2 and stem in _EU_LANG_CODES:
        return stem

    # Pattern-based detection for delimiters like _EN, -NL, (FR), etc.
    for m in _LANG_IN_STEM.finditer(stem):
        code = m.group(1)
        if code in _EU_LANG_CODES:
            return code

    # Fallback: scan alpha tokens split by non-letters.
    # Catches names like file-en-v2 or package_nl_final.
    for token in re.split(r"[^A-Z]+", stem):
        if len(token) == 2 and token in _EU_LANG_CODES:
            return token

    return None


def _detect_file_language(url: str) -> Optional[str]:
    """Return the ISO 639-1 language code found in a URL filename, or None."""
    from urllib.parse import unquote
    name = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    return _detect_language_from_name(name)


def is_language_acceptable(url: str) -> bool:
    """
    Return True if a file URL should be downloaded.
    - Generic files (no EU language code in the name) → always True.
    - Language-tagged files → only EN or NL.
    """
    lang = _detect_file_language(url)
    return lang is None or lang in _ACCEPT_LANG_CODES


def is_filename_language_acceptable(name: str) -> bool:
    """Apply the EN/NL-or-generic language policy to a local filename."""
    lang = _detect_language_from_name(Path(name).name)
    return lang is None or lang in _ACCEPT_LANG_CODES

# COMMAND ----------

# MAGIC %md
# MAGIC ## Press Releases Scraper

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deduplication & Related Content Handler

# COMMAND ----------

class DeduplicationIndex:
    """
    Global tracker for downloaded files across all press releases.
    Prevents the same document from being downloaded multiple times.
    """
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index: dict = self._load_index()
    
    def _load_index(self) -> dict:
        """Load existing deduplication index or create empty one."""
        if self.index_path.exists():
            try:
                return json.load(self.index_path.open(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Could not load index from {self.index_path}, starting fresh")
        return {
            "files": {},  # url → {filename, path, hash, size_kb}
            "references": {},  # filename → [press_release_ids]
        }
    
    def save_index(self) -> None:
        """Persist index to disk."""
        self.index_path.write_text(json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[DEDUP] Index saved: {len(self.index['files'])} unique files tracked")
    
    def file_already_downloaded(self, url: str) -> Optional[dict]:
        """Check if a file URL was already downloaded. Returns file info or None."""
        return self.index["files"].get(url)
    
    def record_file(self, url: str, filename: str, file_path: Path, press_release_id: str) -> None:
        """Record that a file was downloaded and which press releases reference it."""
        if url not in self.index["files"]:
            size_kb = file_path.stat().st_size // 1024 if file_path.exists() else 0
            self.index["files"][url] = {
                "filename": filename,
                "path": str(file_path),
                "size_kb": size_kb,
                "first_downloaded_at": utc_now_iso(),
            }
        
        # Track which press releases reference this file
        if filename not in self.index["references"]:
            self.index["references"][filename] = []
        if press_release_id not in self.index["references"][filename]:
            self.index["references"][filename].append(press_release_id)


def _extract_related_content_links(soup: BeautifulSoup) -> list[dict]:
    """
    Extract 'Related content' links from a press release page.
    Returns list of {label, url} dicts.
    """
    related_links: list[dict] = []
    
    # Look for a dedicated related-content container.
    # EBA often uses: <aside class="page__section page__section__related_content">.
    related_section = soup.find(
        ("aside", "section", "div"),
        class_=re.compile(r"related|related_content|section-link|link-block", re.I),
    )
    
    if not related_section:
        # Fallback: locate a heading that says "Related content" and traverse nearby blocks.
        for heading in soup.find_all(("h2", "h3", "h4", "span", "strong")):
            if "related content" in heading.get_text(" ", strip=True).lower():
                container = heading.find_parent(("aside", "section", "div"))
                if container:
                    related_section = container
                    break
    
    if related_section:
        for a in related_section.find_all("a", href=True):
            href = urljoin(BASE_URL, a["href"])
            label = a.get_text(" ", strip=True)
            if not label:
                continue
            if href == BASE_URL:
                continue
            related_links.append({"label": label, "url": href})

    if related_links:
        return related_links

    # Final fallback: pick links found inside compact teaser cards under related areas.
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        parent = a.find_parent(("aside", "section", "div"))
        parent_class = " ".join(parent.get("class", [])) if parent else ""
        if not re.search(r"related|teaser", parent_class, re.I):
            continue
        label = a.get_text(" ", strip=True)
        if label and href != BASE_URL:
            related_links.append({"label": label, "url": href})
    
    return related_links


def scrape_related_page(
    page_url: str,
    release_dir: Path,
    dedup_index: DeduplicationIndex,
    press_release_id: str,
    dry_run: bool = False,
) -> dict:
    """
    Scrape a related content page linked from a press release.
    Downloads related documents, using deduplication to avoid re-downloading.
    Related ZIP files are size-capped to avoid very large multilingual
    taxonomy/IT bundles while still allowing small useful ZIPs.
    
    Returns {title, url, files_downloaded, new_files_count}.
    """
    logger.info(f"  [RELATED] Fetching: {page_url}")
    time.sleep(PAGE_DELAY)
    resp = fetch(page_url, dry_run=dry_run)
    if resp is None:
        return {"title": page_url, "url": page_url, "files_downloaded": [], "new_files_count": 0}
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Extract page title
    title_el = soup.find("h1")
    page_title = title_el.get_text(strip=True) if title_el else page_url
    
    # Find all downloadable documents
    doc_urls: dict[str, str] = {}
    content_el = soup.find("main") or soup.body or soup
    for a in content_el.find_all("a", href=True):
        full_url = urljoin(page_url, a["href"])
        if re.search(r"\.(pdf|zip|xlsx|xls|xml)(\?|$)", full_url, re.I) and not re.search(
            r"/rss\.xml(\?|$)", full_url, re.I
        ):
            if re.search(r"\.zip(\?|$)", full_url, re.I):
                size = fetch_content_length(full_url)
                if size is not None and size > MAX_RELATED_ZIP_BYTES:
                    logger.info(
                        f"    [ZIP-SKIP] Related ZIP too large ({size // 1024} KB): {safe_filename(full_url)}"
                    )
                    continue
            label = a.get_text(strip=True) or safe_filename(full_url)
            doc_urls[full_url] = label
    
    # Download with deduplication
    files_downloaded: list[dict] = []
    archive_source_map: dict[str, dict] = {}
    new_files_count = 0
    
    for doc_url, label in doc_urls.items():
        # Check if already downloaded
        existing = dedup_index.file_already_downloaded(doc_url)
        if existing:
            logger.info(f"    [DEDUP] Already have: {existing['filename']} (from earlier press release)")
            files_downloaded.append({
                "filename": existing["filename"],
                "label": label,
                "source_url": doc_url,
                "is_reference": True,  # Mark as reference, not newly downloaded
                "original_path": existing["path"],
            })
        else:
            # Download and track
            max_bytes = MAX_RELATED_ZIP_BYTES if re.search(r"\.zip(\?|$)", doc_url, re.I) else None
            fname = download_file(
                doc_url,
                release_dir,
                dry_run=dry_run,
                max_bytes=max_bytes,
            )
            if fname:
                file_path = release_dir / fname
                dedup_index.record_file(doc_url, fname, file_path, press_release_id)
                file_entry = {
                    "filename": fname,
                    "label": label,
                    "source_url": doc_url,
                    "is_reference": False,
                }
                if file_path.suffix.lower() == ".zip":
                    extracted_files = extract_zip_file(file_path, release_dir, delete_zip=True)
                    file_entry["is_archive"] = True
                    file_entry["archive_deleted_after_extract"] = bool(extracted_files)
                    file_entry["extracted_files"] = extracted_files
                    for extracted_rel_path in extracted_files:
                        archive_source_map[extracted_rel_path] = {
                            "archive_filename": fname,
                            "archive_url": doc_url,
                        }
                files_downloaded.append(file_entry)
                new_files_count += 1
    
    return {
        "title": page_title,
        "url": page_url,
        "files_downloaded": files_downloaded,
        "archive_source_map": archive_source_map,
        "new_files_count": new_files_count,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Press Releases Scraper
# COMMAND ----------

def _parse_listing_items(soup: BeautifulSoup) -> list[dict]:
    """
    Extract {title, date, url} items from a press-release listing page.

    EBA's press-release listing currently renders each result as an <article>
    containing a permalink under /publications-and-media/press-releases/ plus a
    visible publication date such as '19 MARCH 2026'.
    """
    items: list[dict] = []
    seen_urls: set[str] = set()
    press_release_path = re.compile(
        r"/publications-and-media/press-releases/[^/?#]+$", re.I
    )
    date_pattern = re.compile(r"\b\d{1,2}\s+[A-Z]+\s+20\d{2}\b")

    for node in soup.find_all("article"):
        link_tag = node.find("a", href=press_release_path)
        if not link_tag:
            continue

        title = link_tag.get_text(strip=True)
        if not title:
            continue

        href = urljoin(BASE_URL, link_tag["href"])
        if href in seen_urls:
            continue

        article_text = " ".join(node.get_text(" ", strip=True).split())
        date_match = date_pattern.search(article_text)
        if not date_match:
            continue

        date_str = date_match.group(0)
        seen_urls.add(href)

        items.append({"title": title, "date": date_str, "url": href})

    if items:
        return items

    # Fallback: parse direct press-release links even if the article markup changes.
    for a in soup.find_all("a", href=press_release_path):
        title = a.get_text(strip=True)
        href = urljoin(BASE_URL, a["href"])
        if not title or href in seen_urls:
            continue
        items.append({"title": title, "date": "", "url": href})
        seen_urls.add(href)

    return items


def _parse_date(raw: str) -> Optional[datetime]:
    """Try common EBA date formats; fall back to bare year extraction."""
    for fmt in (
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            pass
    m = re.search(r"\b(20\d{2})\b", raw)
    return datetime(int(m.group(1)), 1, 1) if m else None


def _extract_press_release(soup: BeautifulSoup, page_url: str) -> dict:
    """
    Return {text, doc_urls} from an individual press release page.
    doc_urls covers PDFs, ZIPs, XLSX, and XML (taxonomy/DPM packages).
    """
    content_el = (
        soup.find("div", class_=re.compile(
            r"field--body|article-body|content-body|field-items|main-content",
            re.I,
        ))
        or soup.find("main")
        or soup.body
    )
    text = content_el.get_text(separator="\n", strip=True) if content_el else ""

    doc_urls: list[str] = []
    source_root = content_el if content_el is not None else soup
    for a in source_root.find_all("a", href=True):
        full_url = urljoin(page_url, a["href"])
        if re.search(r"\.(pdf|zip|xlsx|xls|xml)(\?|$)", full_url, re.I) and not re.search(
            r"/rss\.xml(\?|$)", full_url, re.I
        ):
            doc_urls.append(full_url)

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique_docs: list[str] = []
    for u in doc_urls:
        if u not in seen:
            seen.add(u)
            unique_docs.append(u)

    return {"text": text, "doc_urls": unique_docs}


def scrape_press_releases(
    output_dir: Path,
    dry_run: bool = False,
    years: Optional[frozenset] = None,
) -> list[dict]:
    """
    Paginate EBA press releases, filter by year and relevance keywords,
    and download content + attached documents.
    Stops pagination once a date earlier than the earliest target year.
    """
    if years is None:
        years = TARGET_YEARS
    section_dir = output_dir / "press_releases"
    section_dir.mkdir(parents=True, exist_ok=True)

    collected: list[dict] = []
    page = 0
    stop_pagination = False
    dedup_index = DeduplicationIndex(output_dir / "deduplication_manifest.json")

    while not stop_pagination:
        # EBA Drupal site: first page is the bare URL, subsequent pages use ?page=N (1-based)
        listing_url = PRESS_RELEASES_URL if page == 0 else f"{PRESS_RELEASES_URL}?page={page}"
        logger.info(f"[press_releases] Page {page}: {listing_url}")

        resp = fetch(listing_url, dry_run=dry_run)
        if dry_run:
            logger.info("[DRY-RUN] Exiting after first page in dry-run mode")
            break
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = _parse_listing_items(soup)

        if not items:
            logger.info("[press_releases] Empty listing page — stopping")
            break

        for item in items:
            dt = _parse_date(item["date"])
            if dt is None:
                logger.debug(
                    f"  Cannot parse date '{item['date']}' for: {item['title']}"
                )
                # Don't stop — date may be missing from listing; try to continue
                continue

            if dt.year < min(years):
                logger.info(
                    f"  Year {dt.year} is before {min(years)} — stopping pagination"
                )
                stop_pagination = True
                break

            if dt.year not in years:
                continue  # future year or gap — keep paginating

            if not is_relevant(item["title"]):
                logger.debug(f"  [SKIP] Not relevant: {item['title']}")
                continue

            logger.info(f"  [MATCH] {item['date']} | {item['title']}")

            date_str = dt.strftime("%Y-%m-%d")
            slug = slugify(item["title"])
            release_dir = section_dir / f"{date_str}_{slug}"
            release_dir.mkdir(parents=True, exist_ok=True)

            meta_path = release_dir / "metadata.json"
            existing_meta: Optional[dict] = None
            if meta_path.exists():
                with meta_path.open(encoding="utf-8") as fh:
                    existing_meta = json.load(fh)

                if existing_meta.get("related_content_pages"):
                    logger.info(f"  [SKIP] Already scraped (with related content): {release_dir.name}")
                    collected.append(existing_meta)
                    continue

                logger.info(f"  [REFRESH] Backfilling related content: {release_dir.name}")

            time.sleep(PAGE_DELAY)
            pr_resp = fetch(item["url"])
            if pr_resp is None:
                continue

            pr_soup = BeautifulSoup(pr_resp.text, "html.parser")
            content = _extract_press_release(pr_soup, item["url"])

            # Save press release text
            txt_path = release_dir / "press_release.txt"
            if not txt_path.exists() and not dry_run:
                txt_path.write_text(content["text"], encoding="utf-8")

            # Download attached documents
            downloaded: list[str] = (
                list(existing_meta.get("downloaded_files", []))
                if existing_meta
                else ["press_release.txt"]
            )
            if "press_release.txt" not in downloaded:
                downloaded.insert(0, "press_release.txt")

            for doc_url in content["doc_urls"]:
                fname = download_file(doc_url, release_dir, dry_run=dry_run)
                if fname:
                    if fname not in downloaded:
                        downloaded.append(fname)
                    file_path = release_dir / fname
                    if file_path.suffix.lower() == ".zip":
                        extract_zip_file(file_path, release_dir, delete_zip=True)
                    dedup_index.record_file(doc_url, fname, file_path, release_dir.name)

            # NEW: Extract and follow "Related content" links
            related_content_pages: list[dict] = []
            related_links = _extract_related_content_links(pr_soup)
            
            if related_links:
                logger.info(f"  [RELATED] Found {len(related_links)} related content pages")
                related_dir = release_dir / "related_content"
                related_dir.mkdir(parents=True, exist_ok=True)

                for link_info in related_links:
                    related_page_data = scrape_related_page(
                        link_info["url"],
                        related_dir,
                        dedup_index,
                        release_dir.name,
                        dry_run=dry_run,
                    )
                    related_content_pages.append(related_page_data)

            meta = {
                "title": item["title"],
                "date": date_str,
                "source_url": item["url"],
                "downloaded_files": downloaded,
                "scraped_at": utc_now_iso(),
            }
            
            if related_content_pages:
                meta["related_content_pages"] = related_content_pages
                meta["total_new_files_from_related"] = sum(
                    p.get("new_files_count", 0) for p in related_content_pages
                )
            
            if not dry_run:
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            collected.append(meta)

        page += 1  # page 0 = bare URL, page 1+ = ?page=N
        time.sleep(PAGE_DELAY)

    logger.info(
        f"[press_releases] Finished — {len(collected)} relevant releases collected"

    )
    return collected

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section Page Scraper (Supervisory Reporting & Reporting Frameworks)

# COMMAND ----------

def scrape_section_page(
    section_key: str,
    section_url: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Scrape a single EBA section page and download all linked documents
    (PDFs, ZIPs, XLSX, XML). Skips files already on disk.
    """
    section_dir = output_dir / section_key
    section_dir.mkdir(parents=True, exist_ok=True)
    meta_path = section_dir / "metadata.json"

    logger.info(f"[{section_key}] Fetching: {section_url}")
    resp = fetch(section_url, dry_run=dry_run)
    if dry_run:
        return {"section": section_key, "source_url": section_url, "dry_run": True}
    if resp is None:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Collect all document links with associated anchor labels
    content_el = soup.find("main") or soup.body or soup
    doc_entries: dict[str, str] = {}
    for a in content_el.find_all("a", href=True):
        full_url = urljoin(section_url, a["href"])
        if re.search(r"\.(pdf|zip|xlsx|xls|xml)(\?|$)", full_url, re.I) and not re.search(
            r"/rss\.xml(\?|$)", full_url, re.I
        ):
            label = a.get_text(strip=True) or safe_filename(full_url)
            doc_entries[full_url] = label

    logger.info(f"[{section_key}] Found {len(doc_entries)} document links")

    downloaded: list[dict] = []
    for doc_url, label in doc_entries.items():
        fname = download_file(doc_url, section_dir, dry_run=dry_run)
        if fname:
            file_path = section_dir / fname
            extracted_files: list[str] = []
            if file_path.suffix.lower() == ".zip":
                extracted_files = extract_zip_file(file_path, section_dir, delete_zip=True)
            downloaded.append(
                {
                    "filename": fname,
                    "label": label,
                    "source_url": doc_url,
                    "is_archive": file_path.suffix.lower() == ".zip",
                    "archive_deleted_after_extract": bool(extracted_files),
                    "extracted_files": extracted_files,
                }
            )

    title_el = soup.find("h1")
    page_title = title_el.get_text(strip=True) if title_el else section_key

    meta = {
        "title": page_title,
        "section": section_key,
        "source_url": section_url,
        "downloaded_files": downloaded,
        "scraped_at": utc_now_iso(),
    }
    if not dry_run:
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    logger.info(
        f"[{section_key}] Done — {len(downloaded)} files downloaded"
    )
    return meta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Orchestrator

# COMMAND ----------

def run(
    section: str,
    output_dir: Path,
    dry_run: bool = False,
    years: Optional[frozenset] = None,
) -> None:
    """Run the scraper for the requested section(s) and write scrape_log.json."""
    if years is None:
        years = TARGET_YEARS
    output_dir.mkdir(parents=True, exist_ok=True)

    scrape_log: dict = {
        "started_at": utc_now_iso(),
        "section": section,
        "dry_run": dry_run,
        "target_years": sorted(years),
        "results": {},
    }

    if section in ("all", "press"):
        scrape_log["results"]["press_releases"] = scrape_press_releases(
            output_dir, dry_run=dry_run, years=years
        )

    if section in ("all", "supervisory"):
        scrape_log["results"]["supervisory_reporting"] = scrape_section_page(
            "supervisory_reporting",
            SUPERVISORY_REPORTING_URL,
            output_dir,
            dry_run=dry_run,
        )

    if section in ("all", "frameworks"):
        scrape_log["results"]["reporting_frameworks"] = scrape_section_page(
            "reporting_frameworks",
            REPORTING_FRAMEWORKS_URL,
            output_dir,
            dry_run=dry_run,
        )

    scrape_log["finished_at"] = utc_now_iso()

    log_path = output_dir / "scrape_log.json"
    if not dry_run:
        log_path.write_text(
            json.dumps(scrape_log, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Scrape log saved → {log_path}")
    else:
        logger.info("[DRY-RUN] Would write scrape_log.json")

    logger.info("All done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry Point — Notebook Widgets or CLI

# COMMAND ----------

def _get_args() -> argparse.Namespace:
    """
    Resolve run parameters from:
    1. Databricks widgets (when running as a notebook job)
    2. argparse (when invoked from the command line)
    3. Sensible defaults (when executed interactively in a notebook REPL)
    """
    # ── Databricks notebook context ──────────────────────────────────────────
    dbutils_client = globals().get("dbutils")
    if dbutils_client is not None:
        dbutils_client.widgets.dropdown(
            "section", "all", ["all", "press", "supervisory", "frameworks"]
        )
        dbutils_client.widgets.text("output", "eba_knowledge_base")
        dbutils_client.widgets.dropdown("dry_run", "false", ["true", "false"])
        return argparse.Namespace(
            section=dbutils_client.widgets.get("section"),
            output=dbutils_client.widgets.get("output"),
            dry_run=dbutils_client.widgets.get("dry_run").lower() == "true",
            years=[2026],
        )

    # ── Interactive notebook / IPython REPL ──────────────────────────────────
    if "ipykernel" in sys.modules:
        return argparse.Namespace(
            section="all",
            output="eba_knowledge_base",
            dry_run=False,
            years=[2026],
        )

    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="EBA regulatory publications scraper"
    )
    parser.add_argument(
        "--section",
        choices=["all", "press", "supervisory", "frameworks"],
        default="all",
        help="Which section(s) to scrape (default: all)",
    )
    parser.add_argument(
        "--output",
        default="eba_knowledge_base",
        help="Root output directory (default: eba_knowledge_base)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned actions without downloading anything",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2026],
        metavar="YEAR",
        help="Target year(s) for press releases (default: 2026). E.g. --years 2025 2026",
    )
    return parser.parse_args()

# COMMAND ----------

if __name__ == "__main__":
    args = _get_args()
    target_years = frozenset(getattr(args, "years", [2026]))
    logger.info(
        f"Starting EBA scraper | section={args.section} "
        f"| output={args.output} | dry_run={args.dry_run} "
        f"| years={sorted(target_years)}"
    )
    run(
        section=args.section,
        output_dir=Path(args.output),
        dry_run=args.dry_run,
        years=target_years,
    )

# COMMAND ----------
