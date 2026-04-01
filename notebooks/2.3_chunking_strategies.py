# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.3: Chunking Strategies
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why chunking matters for RAG
# MAGIC - Chunking strategies overview
# MAGIC - Extracting and cleaning EBA document chunks from `eba_parsed_docs`
# MAGIC - Storing final chunks in `eba_chunks` with Change Data Feed enabled
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC eba_parsed_docs  (raw JSON from notebook 2.2)
# MAGIC     ↓  extract elements + clean text
# MAGIC eba_chunks  (paragraph-level chunks with metadata)
# MAGIC     ↓  (next notebook: 2.4)
# MAGIC Vector Search index
# MAGIC ```

# COMMAND ----------

import json
import re
from datetime import datetime

from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql import types as T

from eba_regulatory_agent.config import get_config, get_env

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

env = get_env(spark)
cfg = get_config(env)
catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Catalog: {catalog} | Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Chunking Matters
# MAGIC
# MAGIC **Chunking** breaks documents into smaller pieces for:
# MAGIC
# MAGIC 1. **Embedding Generation**: Most embedding models have token limits (512–8192 tokens)
# MAGIC 2. **Retrieval Precision**: Smaller chunks = more precise retrieval
# MAGIC 3. **Context Window**: LLMs have limited context windows
# MAGIC 4. **Cost Optimization**: Fewer tokens = lower cost
# MAGIC
# MAGIC ### The Chunking Trade-off
# MAGIC - **Large chunks**: More context, but less precise retrieval
# MAGIC - **Small chunks**: More precise, but may lose context
# MAGIC
# MAGIC **Optimal chunk size**: 256–512 tokens for most use cases (~1000–2000 characters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunking Strategies Overview
# MAGIC
# MAGIC | Strategy | How | Best For |
# MAGIC |---|---|---|
# MAGIC | **Fixed-size** | Split by character/token count | Simple, fast |
# MAGIC | **Sentence-based** | Split on sentence boundaries | Preserves sentences |
# MAGIC | **Paragraph-based** | Split on double newlines | Documents with clear structure |
# MAGIC | **AI Parse Documents** | AI identifies elements (text, tables, headings) | Complex regulatory PDFs ✅ |
# MAGIC
# MAGIC We use **paragraph-based** chunking on the TEXT output from `ai_parse_document`,
# MAGIC which already strips layout noise — giving clean, semantic chunks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Parsed Documents from Notebook 2.2

# COMMAND ----------

parsed_table = f"`{catalog}`.`{schema}`.eba_parsed_docs"
parsed_df = spark.table(parsed_table)

logger.info(f"Total parsed documents: {parsed_df.count()}")
parsed_df.select("file_name", "category", "parsed_at").show(truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect the Parsed JSON Structure

# COMMAND ----------

sample_row = parsed_df.select("file_name", "parsed_content").first()
if sample_row:
    parsed_json = json.loads(sample_row["parsed_content"])
    logger.info(f"Document  : {sample_row['file_name']}")
    logger.info(f"Top-level keys: {list(parsed_json.keys())}")

    elements = parsed_json.get("document", {}).get("elements", [])
    if elements:
        from collections import Counter

        type_counts = Counter(e.get("type", "unknown") for e in elements)
        logger.info(f"Element types: {dict(type_counts)}")
    else:
        # Flat TEXT mode — content is a plain string
        content = parsed_json.get("content", str(parsed_json))
        logger.info(f"Content preview (first 500 chars):\n{content[:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Extract and Clean Chunks

# COMMAND ----------


def clean_text(text: str) -> str:
    """Remove excessive whitespace and fix common OCR artefacts."""
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # de-hyphenate line breaks
    return text.strip()


def extract_chunks_from_parsed(parsed_content_str: str) -> list[str]:
    """
    Extract paragraph-level text chunks from ai_parse_document JSON output.

    Handles both:
    - Structured output: document.elements[type=text].content
    - Flat TEXT output: content (plain string)
    """
    try:
        parsed = json.loads(parsed_content_str)
    except Exception:
        return []

    elements = parsed.get("document", {}).get("elements", [])
    if elements:
        texts = [
            clean_text(e.get("content", ""))
            for e in elements
            if e.get("type") == "text" and e.get("content")
        ]
    else:
        # Flat TEXT mode — split on paragraph boundaries
        raw = parsed.get("content", str(parsed))
        texts = [clean_text(p) for p in re.split(r"\n\n+", raw)]

    return [t for t in texts if len(t) > 50]


extract_chunks_udf = F.udf(
    extract_chunks_from_parsed,
    T.ArrayType(T.StringType()),
)

# COMMAND ----------

chunks_df = (
    parsed_df.withColumn("chunks", extract_chunks_udf(F.col("parsed_content")))
    .select(
        "file_name",
        "category",
        "volume_path",
        F.posexplode(F.col("chunks")).alias("chunk_index", "chunk_text"),
    )
    .withColumn("ingestion_timestamp", F.lit(datetime.now().isoformat()))
)

n_chunks = chunks_df.count()
n_docs = parsed_df.count()
logger.info(f"Extracted {n_chunks} chunks from {n_docs} document(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Chunk Statistics

# COMMAND ----------

stats = chunks_df.select(
    F.count("*").alias("total_chunks"),
    F.avg(F.length("chunk_text")).alias("avg_chars"),
    F.min(F.length("chunk_text")).alias("min_chars"),
    F.max(F.length("chunk_text")).alias("max_chars"),
).first()

logger.info(f"Total chunks   : {stats['total_chunks']}")
logger.info(
    f"Avg length     : {stats['avg_chars']:.0f} chars (~{stats['avg_chars'] / 4:.0f} tokens)"
)
logger.info(f"Min / Max      : {stats['min_chars']} / {stats['max_chars']} chars")

chunks_df.groupBy("category").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save to `eba_chunks` Delta Table

# COMMAND ----------

(
    chunks_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(f"`{catalog}`.`{schema}`.eba_chunks")
)

logger.info(f"✅ Saved {n_chunks} chunks → {catalog}.{schema}.eba_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verify

# COMMAND ----------

result_df = spark.table(f"`{catalog}`.`{schema}`.eba_chunks")
result_df.select("file_name", "category", "chunk_index", "chunk_text").show(
    10, truncate=80
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Next Steps
# MAGIC
# MAGIC Chunks are now in `eba_chunks` with Change Data Feed enabled.
# MAGIC In **Notebook 2.4** we will:
# MAGIC - Generate embeddings for each chunk using a Databricks embedding model
# MAGIC - Index them in a Vector Search index for semantic retrieval
