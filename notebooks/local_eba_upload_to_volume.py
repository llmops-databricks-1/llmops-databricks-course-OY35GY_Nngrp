# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1.3b: EBA Knowledge Base — Upload to Unity Catalog Volume
# MAGIC
# MAGIC Provisions a Unity Catalog volume (if needed) and uploads the locally-scraped
# MAGIC `eba_knowledge_base/` folder tree into it.  Also writes a Delta metadata table
# MAGIC (`eba_documents`) that records every file's path, section, press-release ID,
# MAGIC language, and file type — ready for downstream parsing and vector-search indexing.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `1.3_eba_data_scraper.py` first to build `eba_knowledge_base/` locally.

# COMMAND ----------

import json
import os
import re
from pathlib import Path
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from your_custom_package.config import get_config, get_env

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = get_config(env)

CATALOG     = cfg.catalog
SCHEMA      = cfg.schema
VOLUME      = cfg.volume
VOLUME_PATH = cfg.volume_path
LOCAL_KB    = "eba_knowledge_base"
TABLE_NAME  = f"{CATALOG}.{SCHEMA}.eba_documents"

logger.info(f"Env: {env}  Catalog: {CATALOG}  Schema: {SCHEMA}  Volume: {VOLUME}")
logger.info(f"Local KB: {LOCAL_KB}  →  {VOLUME_PATH}")

# COMMAND ----------

_EU_LANG_CODES: frozenset = frozenset({
    "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR",
    "GA", "HR", "HU", "IT", "LT", "LV", "MT", "NL", "PL", "PT",
    "RO", "SK", "SL", "SV",
})
_LANG_PATTERN = re.compile(r"(?<=[_\-\(])([A-Z]{2})(?=[)_\-.]|$)")


def _infer_language(filename: str) -> Optional[str]:
    """Return EN, NL, other EU code, or None (generic) based on the filename."""
    stem = Path(filename).stem.upper()
    if len(stem) == 2 and stem in _EU_LANG_CODES:
        return stem
    for m in _LANG_PATTERN.finditer(stem):
        code = m.group(1)
        if code in _EU_LANG_CODES:
            return code
    for token in re.split(r"[^A-Z]+", stem):
        if len(token) == 2 and token in _EU_LANG_CODES:
            return token
    return None

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Provision Schema and Volume

# COMMAND ----------

from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Helper: Discover Accessible Schemas and Volumes

# COMMAND ----------

try:
    catalogs = [c.name for c in w.catalogs.list()]
    logger.info(f"Accessible catalogs: {catalogs}")
except Exception as exc:
    logger.warning(f"Could not list catalogs: {exc}")

try:
    schemas_in_catalog = [s.name for s in w.schemas.list(catalog_name=CATALOG)]
    logger.info(f"Accessible schemas in {CATALOG}: {schemas_in_catalog}")
except Exception as exc:
    logger.warning(f"Could not list schemas in {CATALOG}: {exc}")
    schemas_in_catalog = []

if SCHEMA in schemas_in_catalog:
    try:
        volumes_in_schema = [v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)]
        logger.info(f"Accessible volumes in {CATALOG}.{SCHEMA}: {volumes_in_schema}")
    except Exception as exc:
        logger.warning(f"Could not list volumes in {CATALOG}.{SCHEMA}: {exc}")
else:
    logger.info(
        f"Schema {CATALOG}.{SCHEMA} not currently visible. "
        "Set EBA_SCHEMA to one from the list above or request access."
    )

# Create schema if absent
existing_schemas = [s.name for s in w.schemas.list(catalog_name=CATALOG)]
if SCHEMA not in existing_schemas:
    try:
        w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        logger.info(f"Schema created: {CATALOG}.{SCHEMA}")
    except PermissionDenied as exc:
        logger.error(f"Missing CREATE SCHEMA on catalog {CATALOG}: {exc}")
        logger.info(f"Existing schemas in {CATALOG}: {existing_schemas}")
        if existing_schemas:
            SCHEMA = existing_schemas[0]
            VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
            TABLE_NAME = f"{CATALOG}.{SCHEMA}.eba_documents"
            logger.warning(
                f"Falling back to accessible schema: {CATALOG}.{SCHEMA}. "
                "Set EBA_SCHEMA to pin a specific schema."
            )
        else:
            raise RuntimeError(
                "No CREATE SCHEMA permission and no accessible schema found. "
                "Ask a UC admin for access."
            ) from exc
else:
    logger.info(f"Schema already exists: {CATALOG}.{SCHEMA}")

# Create volume if absent
existing_volumes = [v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)]
if VOLUME not in existing_volumes:
    try:
        w.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            name=VOLUME,
            volume_type=VolumeType.MANAGED,
        )
        logger.info(f"Volume created: {VOLUME_PATH}")
    except PermissionDenied as exc:
        logger.error(f"Missing CREATE VOLUME on {CATALOG}.{SCHEMA}: {exc}")
        logger.info(f"Existing volumes in {CATALOG}.{SCHEMA}: {existing_volumes}")
        if existing_volumes:
            VOLUME = existing_volumes[0]
            VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
            logger.warning(
                f"Falling back to accessible volume: {VOLUME_PATH}. "
                "Set EBA_VOLUME to pin a specific volume."
            )
        else:
            raise RuntimeError(
                "No CREATE VOLUME permission and no accessible volume found in this schema. "
                "Ask a UC admin for CREATE VOLUME or provide an existing volume."
            ) from exc
else:
    logger.info(f"Volume already exists: {VOLUME_PATH}")

logger.info(f"Using target path: {VOLUME_PATH}")
logger.info(f"Using metadata table: {TABLE_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Upload Files to Volume

# COMMAND ----------

def _resolve_local_kb_path(local_kb: str) -> Path:
    """Resolve local KB path across common workspace execution roots."""
    candidates: list[Path] = []
    raw = Path(local_kb)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        cwd = Path.cwd()
        candidates.extend([
            cwd / raw,
            cwd.parent / raw,
        ])
        project_root = os.environ.get("DATABRICKS_PROJECT_ROOT")
        if project_root:
            candidates.append(Path(project_root) / raw)

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    attempted = [str(c.resolve(strict=False)) for c in candidates] or [str(raw)]
    raise FileNotFoundError(
        "Local knowledge base not found. Tried:\n"
        + "\n".join(attempted)
        + "\nRun notebooks/1.3_eba_data_scraper.py first, or set LOCAL_KB to an absolute path."
    )


local_root = _resolve_local_kb_path(LOCAL_KB)
logger.info(f"Resolved local KB path: {local_root}")

SKIP_FILENAMES = {"metadata.json", "scrape_log.json", "deduplication_manifest.json"}
SKIP_EXTENSIONS = {".py", ".pyc", ".log"}

all_files = [
    f for f in local_root.rglob("*")
    if f.is_file()
    and f.name not in SKIP_FILENAMES
    and f.suffix.lower() not in SKIP_EXTENSIONS
]

logger.info(f"Files to upload: {len(all_files)}")

uploaded = 0
skipped  = 0
failed   = 0

for local_path in all_files:
    relative    = local_path.relative_to(local_root)
    volume_dest = f"{VOLUME_PATH}/{relative.as_posix()}"

    try:
        w.files.get_metadata(volume_dest)
        logger.info(f"  [SKIP] Already in volume: {relative}")
        skipped += 1
        continue
    except Exception:
        pass  # File not present — upload it

    try:
        with local_path.open("rb") as fh:
            w.files.upload(volume_dest, fh, overwrite=True)
        logger.info(f"  [UP] {relative}")
        uploaded += 1
    except Exception as exc:
        logger.error(f"  [FAIL] {relative}: {exc}")
        failed += 1

logger.info(
    f"Upload complete — uploaded: {uploaded}  skipped: {skipped}  failed: {failed}"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Register Delta Metadata Table

# COMMAND ----------

doc_schema = StructType([
    StructField("volume_path",      StringType(), False),
    StructField("relative_path",    StringType(), False),
    StructField("section",          StringType(), True),
    StructField("press_release_id", StringType(), True),
    StructField("filename",         StringType(), False),
    StructField("file_extension",   StringType(), True),
    StructField("language",         StringType(), True),
    StructField("source_url",       StringType(), True),
    StructField("scraped_at",       StringType(), True),
])

records = []

# Press releases
pr_root = local_root / "press_releases"
if pr_root.exists():
    for pr_dir in pr_root.iterdir():
        if not pr_dir.is_dir():
            continue
        meta: dict = {}
        meta_path = pr_dir / "metadata.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as fh:
                meta = json.load(fh)
        for fpath in pr_dir.rglob("*"):
            if not fpath.is_file() or fpath.name in SKIP_FILENAMES or fpath.suffix.lower() in SKIP_EXTENSIONS:
                continue
            rel = fpath.relative_to(local_root).as_posix()
            records.append({
                "volume_path":      f"{VOLUME_PATH}/{rel}",
                "relative_path":    rel,
                "section":          "press_releases",
                "press_release_id": pr_dir.name,
                "filename":         fpath.name,
                "file_extension":   fpath.suffix.lower(),
                "language":         _infer_language(fpath.name),
                "source_url":       meta.get("source_url"),
                "scraped_at":       meta.get("scraped_at"),
            })

# Section pages
for section_key in ("supervisory_reporting", "reporting_frameworks"):
    sec_root = local_root / section_key
    if not sec_root.exists():
        continue
    meta: dict = {}
    meta_path = sec_root / "metadata.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
    for fpath in sec_root.rglob("*"):
        if not fpath.is_file() or fpath.name in SKIP_FILENAMES or fpath.suffix.lower() in SKIP_EXTENSIONS:
            continue
        rel = fpath.relative_to(local_root).as_posix()
        records.append({
            "volume_path":      f"{VOLUME_PATH}/{rel}",
            "relative_path":    rel,
            "section":          section_key,
            "press_release_id": None,
            "filename":         fpath.name,
            "file_extension":   fpath.suffix.lower(),
            "language":         _infer_language(fpath.name),
            "source_url":       meta.get("source_url"),
            "scraped_at":       meta.get("scraped_at"),
        })

logger.info(f"Metadata records: {len(records)}")

df = spark.createDataFrame(records, schema=doc_schema)

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(TABLE_NAME)

logger.info(f"Delta table written: {TABLE_NAME}")
logger.info(f"Total documents registered: {df.count()}")
df.groupBy("section", "file_extension").count().orderBy("section", "count").show(50, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Verify

# COMMAND ----------

docs_df = spark.table(TABLE_NAME)
logger.info(f"Table: {TABLE_NAME}")
logger.info(f"Total rows: {docs_df.count()}")
docs_df.select("section", "press_release_id", "filename", "language", "file_extension") \
    .show(20, truncate=60)

# COMMAND ----------
