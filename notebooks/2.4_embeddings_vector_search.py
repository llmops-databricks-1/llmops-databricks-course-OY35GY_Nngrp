# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - Understanding embeddings and vector representations
# MAGIC - Embedding model comparison
# MAGIC - Creating a Vector Search endpoint and index
# MAGIC - Similarity search, hybrid search, and reranking
# MAGIC - Metadata filtering and search-quality comparison
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC eba_chunks table (from 2.3)
# MAGIC     ↓  Delta Sync + embedding model
# MAGIC Vector Search Index
# MAGIC     ↓  query
# MAGIC Search Results (scores + metadata)
# MAGIC ```

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.vector_search.reranker import DatabricksReranker
from loguru import logger

from eba_regulatory_agent.config import get_config, get_env
from eba_regulatory_agent.vector_search import VectorSearchManager

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

env = get_env(spark)
cfg = get_config(env)
catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Catalog: {catalog} | Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **Vector**: Array of numbers (e.g., [0.1, -0.3, 0.5, ...])
# MAGIC - **Dimension**: Length of the vector (e.g., 384, 768, 1024)
# MAGIC - **Semantic Similarity**: Similar meanings = similar vectors
# MAGIC - **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
# MAGIC
# MAGIC ### How it Works:
# MAGIC
# MAGIC ```
# MAGIC Text: "machine learning"
# MAGIC   ↓  (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, ..., 0.42]  # 1024 dimensions
# MAGIC
# MAGIC Text: "artificial intelligence"
# MAGIC   ↓  (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, ..., 0.40]  # Similar to above!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embedding Models Comparison
# MAGIC
# MAGIC | Model | Dimensions | Max Tokens | Best For |
# MAGIC |-------|-----------|------------|----------|
# MAGIC | **databricks-bge-large-en** | 1024 | 512 | General purpose, high quality |
# MAGIC | **databricks-gte-large-en** | 1024 | 512 | General purpose, fast |
# MAGIC | **text-embedding-ada-002** (OpenAI) | 1536 | 8191 | High quality, expensive |
# MAGIC | **e5-large-v2** | 1024 | 512 | Open source, good quality |
# MAGIC | **all-MiniLM-L6-v2** | 384 | 512 | Fast, smaller, lower quality |
# MAGIC
# MAGIC **For this course, we'll use `databricks-gte-large-en`** - it's fast, high-quality, and free on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Delta Table (eba_chunks)                  │
# MAGIC │  - chunk_index      (primary key)             │
# MAGIC │  - chunk_text       (embedding source)        │
# MAGIC │  - file_name, category, volume_path           │
# MAGIC └──────────────────┬───────────────────────────┘
# MAGIC                    │  Delta Sync (TRIGGERED)
# MAGIC                    ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Vector Search Index                       │
# MAGIC │  - Embeddings generated automatically         │
# MAGIC │  - Stored in optimised ANN format             │
# MAGIC │  - Supports similarity + hybrid search        │
# MAGIC └──────────────────┬───────────────────────────┘
# MAGIC                    │  Query
# MAGIC                    ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Search Results                            │
# MAGIC │  - Most similar chunks with scores            │
# MAGIC │  - Metadata for filtering / display           │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Endpoint & Index

# COMMAND ----------

vs_manager = VectorSearchManager(config=cfg)

logger.info(f"Endpoint       : {vs_manager.endpoint_name}")
logger.info(f"Embedding model: {vs_manager.embedding_model}")
logger.info(f"Index name     : {vs_manager.index_name}")
logger.info(f"Source table   : {vs_manager.source_table}")

# COMMAND ----------

# Create endpoint (if it doesn't already exist)
vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# Create (or retrieve) the Delta Sync index
vs_manager.create_or_get_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration
# MAGIC
# MAGIC | Option | Value | Explanation |
# MAGIC |--------|-------|-------------|
# MAGIC | `pipeline_type` | `TRIGGERED` | Sync on demand — ideal for batch pipelines |
# MAGIC | `primary_key` | `chunk_index` | Positional chunk identifier |
# MAGIC | `embedding_source_column` | `chunk_text` | The cleaned paragraph text |
# MAGIC | `embedding_model_endpoint_name` | `databricks-gte-large-en` | Free Databricks model |

# COMMAND ----------

# Trigger an initial sync so embeddings are computed
vs_manager.sync_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper: Parsing Results
# MAGIC
# MAGIC `VectorSearchManager.parse_results(results)` converts the raw array response
# MAGIC from `similarity_search()` into a list of plain dicts keyed by column name.


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic Search with Similarity
# MAGIC
# MAGIC ### How Semantic Search Works
# MAGIC
# MAGIC 1. **Query Embedding**: Convert your search query to a vector
# MAGIC 2. **Similarity Calculation**: Compare query vector to all document vectors using **cosine similarity**
# MAGIC 3. **Ranking**: Return documents with highest similarity scores
# MAGIC
# MAGIC ### Cosine Similarity
# MAGIC
# MAGIC Measures the angle between two vectors (range: -1 to 1):
# MAGIC - **1.0**: Identical meaning
# MAGIC - **0.8-0.9**: Very similar
# MAGIC - **0.5-0.7**: Somewhat related
# MAGIC - **< 0.5**: Less relevant
# MAGIC
# MAGIC ```
# MAGIC Query: "machine learning techniques"
# MAGIC   ↓  (Embedding)
# MAGIC Vector: [0.2, 0.5, -0.1, ...]
# MAGIC   ↓  (Cosine similarity with all docs)
# MAGIC Results ranked by similarity score
# MAGIC ```

# COMMAND ----------

# Simple similarity search
query = "What are the capital requirements for credit risk under Basel III?"

results = vs_manager.search(query, num_results=5)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. File    : {row.get('file_name', 'N/A')}")
    logger.info(f"   Category: {row.get('category', 'N/A')}")
    logger.info(f"   Chunk   : {row.get('chunk_index', 'N/A')}")
    logger.info(f"   Text    : {row.get('chunk_text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Metadata Filtering
# MAGIC
# MAGIC Narrow results to a specific EBA document category.
# MAGIC
# MAGIC ```python
# MAGIC # Single filter
# MAGIC filters = {"category": "LCR"}
# MAGIC
# MAGIC # Multiple filters (AND)
# MAGIC filters = {"category": "COREP", "file_name": "ITS_2021_03.pdf"}
# MAGIC ```

results = vs_manager.search(query, num_results=3, filters={"category": "LCR"})

logger.info(f"Query: {query}")
logger.info(f"Filter: category = LCR\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('file_name', 'N/A')}")
    logger.info(f"   Category: {row.get('category', 'N/A')}")
    logger.info(f"   Text    : {row.get('chunk_text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Hybrid Search: Semantic + Keyword (BM25)
# MAGIC
# MAGIC ### Why Hybrid Search?
# MAGIC
# MAGIC **Semantic search alone** may miss:
# MAGIC - Exact technical terms (e.g., "GPT-4" vs "language model")
# MAGIC - Acronyms and abbreviations
# MAGIC - Specific product names or codes
# MAGIC
# MAGIC **Hybrid search** combines:
# MAGIC - **Semantic search** (embeddings) â†’ Captures meaning, synonyms
# MAGIC - **Keyword search** (BM25) â†’ Exact term matching, TF-IDF scoring
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. Run both searches in parallel
# MAGIC 2. Get top-k results from each
# MAGIC 3. **Fusion**: Merge and rerank using:
# MAGIC    - Reciprocal Rank Fusion (RRF)
# MAGIC    - Weighted score combination
# MAGIC 4. Return final top-k
# MAGIC
# MAGIC ### BM25 (Best Match 25)
# MAGIC
# MAGIC Keyword scoring algorithm that considers:
# MAGIC - **Term frequency**: How often does the term appear?
# MAGIC - **Document length**: Normalize by doc length
# MAGIC - **Inverse document frequency**: Rare terms = higher weight
# MAGIC
# MAGIC **Result**: Better precision on technical queries with specific terminology.

# COMMAND ----------

# Hybrid search example
query = "own funds requirements COREP reporting templates"

results = vs_manager.search(query, num_results=5, query_type="hybrid")

logger.info(f"Query: {query}")
logger.info("Search Type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('file_name', 'N/A')}")
    logger.info(f"   Text: {row.get('chunk_text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Reranking for Higher Precision
# MAGIC
# MAGIC ### The Two-Stage Retrieval Pattern
# MAGIC
# MAGIC **Stage 1: Fast Retrieval** (Bi-encoder)
# MAGIC - Retrieve top 20-50 candidates quickly
# MAGIC - Uses pre-computed embeddings
# MAGIC - Fast but less accurate
# MAGIC
# MAGIC **Stage 2: Precise Reranking** (Cross-encoder)
# MAGIC - Score each candidate against the query
# MAGIC - More accurate relevance scoring
# MAGIC - Slower, but only runs on candidates
# MAGIC
# MAGIC ### Bi-encoder vs Cross-encoder
# MAGIC
# MAGIC | Aspect | Bi-encoder | Cross-encoder |
# MAGIC |--------|-----------|---------------|
# MAGIC | **Speed** | Very fast | Slower |
# MAGIC | **Accuracy** | Good | Excellent |
# MAGIC | **Use case** | Initial retrieval | Reranking |
# MAGIC | **How it works** | Separate query & doc embeddings | Joint query-doc encoding |
# MAGIC
# MAGIC ### When to Use Reranking
# MAGIC
# MAGIC - **High-stakes queries**: Customer support, legal, medical
# MAGIC - **Complex queries**: Multi-faceted questions
# MAGIC - **When precision matters more than speed**
# MAGIC
# MAGIC ### Trade-offs
# MAGIC
# MAGIC - **Pros**: 10-30% improvement in relevance
# MAGIC - **Cons**: 2-5x slower, higher compute cost

# COMMAND ----------

# Search with reranking
query = "supervisory disclosure requirements under Pillar 3"

results = vs_manager.search(
    query,
    num_results=5,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["chunk_text"]),
)

logger.info(f"Query: {query}")
logger.info("With reranking on: chunk_text\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('file_name', 'N/A')}")
    logger.info(f"   Text: {row.get('chunk_text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Search Quality Comparison

# COMMAND ----------

# Compare different search strategies
query = "minimum capital requirements for market risk"

logger.info(f"Query: {query}\n")

COLS = ["file_name", "category", "chunk_text"]

# Strategy 1: Basic semantic search
results_basic = vs_manager.search(query, num_results=3)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_basic), 1):
    logger.info(f"{i}. [{row.get('category', 'N/A')}] {row.get('file_name', 'N/A')[:60]}")

# Strategy 2: Hybrid search
results_hybrid = vs_manager.search(query, num_results=3, query_type="hybrid")

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_hybrid), 1):
    logger.info(f"{i}. [{row.get('category', 'N/A')}] {row.get('file_name', 'N/A')[:60]}")

# Strategy 3: Hybrid + Reranking
results_reranked = vs_manager.search(
    query,
    num_results=3,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["chunk_text"]),
)

logger.info("\nStrategy 3: Hybrid + Reranking")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_reranked), 1):
    logger.info(f"{i}. [{row.get('category', 'N/A')}] {row.get('file_name', 'N/A')[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Best Practices
# MAGIC
# MAGIC ### ✅ Do:
# MAGIC 1. **Use hybrid search** for better recall
# MAGIC 2. **Add reranking** for critical applications
# MAGIC 3. **Filter by metadata** to narrow results
# MAGIC 4. **Monitor index sync** status
# MAGIC 5. **Use appropriate num_results** (5-10 for most cases)
# MAGIC 6. **Include relevant columns** in results
# MAGIC 7. **Test different embedding models** for your use case
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Retrieve too many results (increases latency)
# MAGIC 2. Ignore index sync status
# MAGIC 3. Use semantic search for exact keyword matches
# MAGIC 4. Forget to handle empty results
# MAGIC 5. Over-rely on similarity scores alone

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Index Monitoring

# COMMAND ----------

# Check index status
index_info = vs_manager.client.get_index(index_name=vs_manager.index_name)

logger.info("Index Information:")
logger.info(f"  Name    : {vs_manager.index_name}")
logger.info(f"  Endpoint: {vs_manager.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Maintenance
# MAGIC
# MAGIC ```python
# MAGIC # Trigger a manual sync (for TRIGGERED pipeline)
# MAGIC vs_manager.sync_index()
# MAGIC
# MAGIC # Delete index (if needed)
# MAGIC # vs_manager.client.delete_index(index_name=vs_manager.index_name)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ Understanding embeddings and vector representations
# MAGIC 2. ✅ Comparing different embedding models
# MAGIC 3. ✅ Creating vector search endpoints
# MAGIC 4. ✅ Creating and syncing vector search indexes
# MAGIC 5. ✅ Basic similarity search
# MAGIC 6. ✅ Advanced features: filters, hybrid search, reranking
# MAGIC 7. ✅ Comparing search strategies
# MAGIC 8. ✅ Best practices and monitoring
# MAGIC
# MAGIC **Next**: Lecture 2.5 - Pipeline Design & Workflow
