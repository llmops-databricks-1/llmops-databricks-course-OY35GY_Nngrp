# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - AI Parse Documents for intelligent PDF parsing
# MAGIC - Parsing EBA regulatory PDFs from a Unity Catalog Volume
# MAGIC - Storing raw parsed JSON in `eba_parsed_docs` for reuse
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC Volume (EBA PDFs)
# MAGIC     ↓  ai_parse_document(content, 'TEXT')
# MAGIC eba_parsed_docs table  (raw JSON — stored once, parsed once)
# MAGIC     ↓  (next notebook: 2.3)
# MAGIC eba_chunks table
# MAGIC ```

# COMMAND ----------

from databricks.connect import DatabricksSession
from loguru import logger

from eba_regulatory_agent.config import get_config, get_env
from eba_regulatory_agent.data_processor import DataProcessor

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()
logger.info("✅ Spark session ready")

env = get_env(spark)
cfg = get_config(env)

logger.info(f"Environment : {cfg.env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")
logger.info(f"Volume path : {cfg.volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | AI-powered · handles complex layouts · Databricks-native · preserves structure | Databricks-specific · cost per page | Complex documents, tables, multi-column |
# MAGIC | **PyPDF2 / pypdf** | Simple · free · pure Python | Poor with complex layouts · no table extraction | Simple text extraction |
# MAGIC | **pdfplumber** | Good table extraction · layout analysis | Slower · manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | Multi-format support · metadata extraction | Java dependency · heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | ML-powered · good chunking | External service · API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users due to its
# MAGIC integration with Unity Catalog and intelligent layout handling — which matters
# MAGIC for EBA regulatory documents that contain complex tables, numbered paragraphs, and annexes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Review Source PDFs in Volume

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)

# List documents available in the volume (PDF, Word, Excel)
pdf_df = processor.list_volume_documents()
logger.info(f"PDFs found in volume:")
pdf_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parse PDFs with AI Parse Documents
# MAGIC
# MAGIC `DataProcessor.parse_pdfs_with_ai()` will:
# MAGIC 1. Read each PDF as binary from the Volume
# MAGIC 2. Call `ai_parse_document(content, 'TEXT')` — layout-aware AI model
# MAGIC 3. Store the raw JSON output in `eba_parsed_docs` (skips already-parsed files)
# MAGIC
# MAGIC Parsing is expensive (cost per page) — results are stored once and reused in 2.3.

# COMMAND ----------

parsed_count = processor.parse_pdfs_with_ai()
logger.info(f"Documents parsed in this run: {parsed_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect Parsed Output

# COMMAND ----------

import json

parsed_df = spark.table(processor.parsed_table_fqn)
logger.info(f"Total parsed documents: {parsed_df.count()}")
parsed_df.select("file_name", "category", "volume_path", "parsed_at").show(truncate=80)

# COMMAND ----------

# Preview the JSON structure of one document
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
        text_elements = [e for e in elements if e.get("type") == "text"][:3]
        for i, elem in enumerate(text_elements, 1):
            logger.info(f"  Text element {i}: {elem.get('content', '')[:200]}…")
    else:
        # Flat text output (TEXT mode)
        logger.info(f"Content preview: {str(parsed_json)[:400]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Next Steps
# MAGIC
# MAGIC The raw parsed JSON is now stored in `eba_parsed_docs`.
# MAGIC In **Notebook 2.3** we will:
# MAGIC - Extract text elements from the JSON
# MAGIC - Clean and normalise the text
# MAGIC - Split into paragraph-level chunks with metadata
# MAGIC - Store in `eba_chunks` with Change Data Feed enabled (for Vector Search in 2.4)
