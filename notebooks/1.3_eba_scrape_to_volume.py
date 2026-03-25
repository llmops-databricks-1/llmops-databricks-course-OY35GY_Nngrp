# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1.3c: EBA Scrape Directly to Unity Catalog Volume (Test Mode)
# MAGIC
# MAGIC Combines scraping + upload in one flow:
# MAGIC - Scrapes EBA press-release pages
# MAGIC - Downloads file bytes in memory
# MAGIC - Uploads directly to Unity Catalog Volume (no local KB folder)
# MAGIC - Writes metadata to Delta table
# MAGIC
# MAGIC **Current test mode:** uploads only 1 file to validate end-to-end.

# COMMAND ----------

import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import VolumeType
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from your_custom_package.config import get_config, get_env

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Session, Config, and Target Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = get_config(env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
VOLUME = cfg.volume
VOLUME_PATH = cfg.volume_path
TABLE_NAME = f"{CATALOG}.{SCHEMA}.eba_documents"

MAX_FILES = 1  # test mode: keep to one downloaded file
TARGET_YEARS = {2026}

logger.info(f"Env={env} | catalog={CATALOG} | schema={SCHEMA} | volume={VOLUME}")

w = WorkspaceClient()

# Discover what is accessible
catalogs = [c.name for c in w.catalogs.list()]
logger.info(f"Accessible catalogs: {catalogs}")

schemas_in_catalog = [s.name for s in w.schemas.list(catalog_name=CATALOG)]
logger.info(f"Accessible schemas in {CATALOG}: {schemas_in_catalog}")

# Ensure schema exists (fallback when CREATE SCHEMA is missing)
if SCHEMA not in schemas_in_catalog:
    try:
        w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        logger.info(f"Schema created: {CATALOG}.{SCHEMA}")
    except PermissionDenied:
        if schemas_in_catalog:
            SCHEMA = schemas_in_catalog[0]
            TABLE_NAME = f"{CATALOG}.{SCHEMA}.eba_documents"
            VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
            logger.warning(f"Falling back to schema: {CATALOG}.{SCHEMA}")
        else:
            raise

# Ensure volume exists (fallback when CREATE VOLUME is missing)
volumes_in_schema = [
    v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)
]
logger.info(f"Accessible volumes in {CATALOG}.{SCHEMA}: {volumes_in_schema}")

if VOLUME not in volumes_in_schema:
    try:
        w.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            name=VOLUME,
            volume_type=VolumeType.MANAGED,
        )
        logger.info(f"Volume created: {VOLUME_PATH}")
    except PermissionDenied as exc:
        if volumes_in_schema:
            VOLUME = volumes_in_schema[0]
            VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
            logger.warning(f"Falling back to volume: {VOLUME_PATH}")
        else:
            raise RuntimeError(
                f"No CREATE VOLUME permission and no existing volume in {CATALOG}.{SCHEMA}."
            ) from exc

logger.info(f"Using target path: {VOLUME_PATH}")
logger.info(f"Using metadata table: {TABLE_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Scrape + Direct Upload Helpers

# COMMAND ----------

BASE_URL = "https://www.eba.europa.eu"
PRESS_RELEASES_URL = f"{BASE_URL}/publications-and-media/press-releases"

RELEVANCE_KEYWORDS = [
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
]
_KW_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in RELEVANCE_KEYWORDS), re.IGNORECASE
)

_EU_LANG_CODES = {
    "BG",
    "CS",
    "DA",
    "DE",
    "EL",
    "EN",
    "ES",
    "ET",
    "FI",
    "FR",
    "GA",
    "HR",
    "HU",
    "IT",
    "LT",
    "LV",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SL",
    "SV",
}
_ACCEPT_LANG_CODES = {"EN", "NL"}
_LANG_IN_STEM = re.compile(r"(?<=[_\-\(])([A-Z]{2})(?=[)_\-.]|$)")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EBA-Direct-Volume-Scraper/1.0)",
    "Accept-Language": "en-GB,en;q=0.8",
}


def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_bytes(url: str) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.content


def safe_filename(url: str) -> str:
    name = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    if not name:
        name = f"file_{abs(hash(url)) % 0xFFFFFF:06x}"
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r'[<>:"\\|?*]', "_", name)
    return name[:120]


def _detect_file_language(url: str) -> Optional[str]:
    stem = unquote(urlparse(url).path.rstrip("/").split("/")[-1]).upper()
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    if len(stem) == 2 and stem in _EU_LANG_CODES:
        return stem
    for m in _LANG_IN_STEM.finditer(stem):
        code = m.group(1)
        if code in _EU_LANG_CODES:
            return code
    for token in re.split(r"[^A-Z]+", stem):
        if len(token) == 2 and token in _EU_LANG_CODES:
            return token
    return None


def is_language_acceptable(url: str) -> bool:
    lang = _detect_file_language(url)
    return lang is None or lang in _ACCEPT_LANG_CODES


def is_relevant(title: str) -> bool:
    return bool(_KW_PATTERN.search(title))


def parse_date(raw: str) -> Optional[datetime]:
    for fmt in ("%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            pass
    m = re.search(r"\b(20\d{2})\b", raw)
    return datetime(int(m.group(1)), 1, 1) if m else None


def parse_listing_items(soup: BeautifulSoup) -> list[dict]:
    items: list[dict] = []
    seen = set()
    path_re = re.compile(r"/publications-and-media/press-releases/[^/?#]+$", re.I)
    date_re = re.compile(r"\b\d{1,2}\s+[A-Z]+\s+20\d{2}\b")

    for article in soup.find_all("article"):
        a = article.find("a", href=path_re)
        if not a:
            continue
        title = a.get_text(strip=True)
        href = urljoin(BASE_URL, a["href"])
        if not title or href in seen:
            continue
        text = " ".join(article.get_text(" ", strip=True).split())
        m = date_re.search(text)
        if not m:
            continue
        items.append({"title": title, "url": href, "date": m.group(0)})
        seen.add(href)
    return items


def extract_doc_urls(page_url: str) -> list[str]:
    html = fetch_text(page_url)
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("main") or soup.body or soup
    out: list[str] = []
    seen = set()
    for a in content.find_all("a", href=True):
        full = urljoin(page_url, a["href"])
        if not re.search(r"\.(pdf|zip|xlsx|xls|xml)(\?|$)", full, re.I):
            continue
        if re.search(r"/rss\.xml(\?|$)", full, re.I):
            continue
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
    return out


def upload_bytes_to_volume(path_in_volume: str, content: bytes) -> None:
    with io.BytesIO(content) as fh:
        w.files.upload(path_in_volume, fh, overwrite=True)


# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Test Run (One File)

# COMMAND ----------

uploaded_records = []
files_uploaded = 0
page = 0
stop = False

while not stop and files_uploaded < MAX_FILES:
    listing_url = PRESS_RELEASES_URL if page == 0 else f"{PRESS_RELEASES_URL}?page={page}"
    logger.info(f"Listing page {page}: {listing_url}")

    soup = BeautifulSoup(fetch_text(listing_url), "html.parser")
    items = parse_listing_items(soup)
    if not items:
        break

    for item in items:
        if files_uploaded >= MAX_FILES:
            stop = True
            break

        dt = parse_date(item["date"])
        if dt is None:
            continue
        if dt.year < min(TARGET_YEARS):
            stop = True
            break
        if dt.year not in TARGET_YEARS:
            continue
        if not is_relevant(item["title"]):
            continue

        logger.info(f"MATCH: {item['date']} | {item['title']}")
        doc_urls = extract_doc_urls(item["url"])

        for doc_url in doc_urls:
            if files_uploaded >= MAX_FILES:
                stop = True
                break
            if not is_language_acceptable(doc_url):
                continue

            fname = safe_filename(doc_url)
            pr_slug = re.sub(r"[^a-z0-9]+", "-", item["title"].lower()).strip("-")[:40]
            rel_path = f"press_releases/{dt.strftime('%Y-%m-%d')}_{pr_slug}/{fname}"
            vol_path = f"{VOLUME_PATH}/{rel_path}"

            content = fetch_bytes(doc_url)
            upload_bytes_to_volume(vol_path, content)

            uploaded_records.append(
                {
                    "volume_path": vol_path,
                    "relative_path": rel_path,
                    "section": "press_releases",
                    "press_release_id": f"{dt.strftime('%Y-%m-%d')}_{pr_slug}",
                    "filename": fname,
                    "file_extension": Path(fname).suffix.lower(),
                    "language": _detect_file_language(doc_url),
                    "source_url": doc_url,
                    "scraped_at": datetime.utcnow().isoformat(),
                }
            )

            files_uploaded += 1
            logger.info(f"Uploaded ({files_uploaded}/{MAX_FILES}): {vol_path}")
            time.sleep(1.0)

    page += 1

if not uploaded_records:
    raise RuntimeError(
        "No files uploaded in test run. Try relaxing filters or increasing pages."
    )

logger.info(f"Test upload complete. Files uploaded: {len(uploaded_records)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Write / Append Metadata Table

# COMMAND ----------

doc_schema = StructType(
    [
        StructField("volume_path", StringType(), False),
        StructField("relative_path", StringType(), False),
        StructField("section", StringType(), True),
        StructField("press_release_id", StringType(), True),
        StructField("filename", StringType(), False),
        StructField("file_extension", StringType(), True),
        StructField("language", StringType(), True),
        StructField("source_url", StringType(), True),
        StructField("scraped_at", StringType(), True),
    ]
)

new_df = spark.createDataFrame(uploaded_records, schema=doc_schema)

if spark.catalog.tableExists(TABLE_NAME):
    new_df.write.format("delta").mode("append").saveAsTable(TABLE_NAME)
    logger.info(f"Appended {new_df.count()} rows to {TABLE_NAME}")
else:
    new_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_NAME)
    logger.info(f"Created table {TABLE_NAME} with {new_df.count()} rows")

spark.table(TABLE_NAME).select("section", "filename", "language", "file_extension").show(
    20, truncate=60
)

# COMMAND ----------
# MAGIC %md
# MAGIC If this test succeeds, set `MAX_FILES` higher (or remove the limit) for full scrape mode.
