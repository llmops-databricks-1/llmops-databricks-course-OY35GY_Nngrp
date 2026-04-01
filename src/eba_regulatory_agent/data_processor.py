"""EBA regulatory PDF processing — parse PDFs from Volume using ai_parse_document."""

from datetime import datetime, timezone

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from eba_regulatory_agent.config import EbaConfig


class DataProcessor:
    """
    Reads EBA regulatory PDFs from a Unity Catalog Volume, parses them with
    ``ai_parse_document``, and stores the raw JSON in ``eba_parsed_docs``.

    Chunking (``eba_chunks``) is handled separately in Notebook 2.3.
    """

    def __init__(self, spark: SparkSession, config: EbaConfig) -> None:
        self.spark = spark
        self.config = config

    @property
    def parsed_table_fqn(self) -> str:
        return f"`{self.config.catalog}`.`{self.config.schema}`.eba_parsed_docs"

    @property
    def chunks_table_fqn(self) -> str:
        return f"`{self.config.catalog}`.`{self.config.schema}`.eba_chunks"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx")

    def list_volume_documents(self) -> DataFrame:
        """
        Return a DataFrame of all supported documents (PDF, Word, Excel)
        found recursively in the EBA volume.

        Columns: file_name, category, volume_path
        """
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

        def _recurse(path: str) -> list[dict]:
            results = []
            for item in w.files.list_directory_contents(path):
                if item.is_directory:
                    results.extend(_recurse(item.path))
                elif any(
                    item.path.lower().endswith(ext)
                    for ext in DataProcessor._SUPPORTED_EXTENSIONS
                ):
                    results.append(
                        {
                            "volume_path": item.path,
                            "file_name": item.path.split("/")[-1],
                            "category": item.path.split("/")[-2],
                        }
                    )
            return results

        records = _recurse(self.config.volume_path)
        logger.info(
            f"Found {len(records)} document(s) in {self.config.volume_path}"
        )
        return self.spark.createDataFrame(records)

    def parse_pdfs_with_ai(self) -> int:
        """
        Parse all supported documents in the volume that have not yet been stored
        in ``eba_parsed_docs``, using ``ai_parse_document(content, map('mode','TEXT'))``.

        Reads files as binary via binaryFile format (executed on the cluster).
        Returns the number of newly parsed documents.
        """
        # Read all supported files as binary (cluster-side, works with Databricks Connect)
        binary_df = (
            self.spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.{pdf,docx,xlsx}")
            .load(self.config.volume_path)
        )

        # Find already-parsed paths to skip (idempotent)
        already_parsed: set[str] = set()
        try:
            existing_df = self.spark.table(self.parsed_table_fqn)
            already_parsed = {
                row.volume_path
                for row in existing_df.select("volume_path").collect()
            }
            logger.info(
                f"{len(already_parsed)} document(s) already parsed — skipping."
            )
        except Exception:
            # The parsed table may not exist on first run.
            already_parsed = set()

        if already_parsed:
            binary_df = binary_df.filter(~F.col("path").isin(already_parsed))

        n_new = binary_df.count()
        if n_new == 0:
            logger.info("All documents already parsed.")
            return 0

        logger.info(f"Parsing {n_new} new document(s)…")

        binary_df = (
            binary_df
            .withColumn("file_name", F.regexp_extract("path", r"[^/]+$", 0))
            .withColumn("category", F.regexp_extract("path", r"/([^/]+)/[^/]+$", 1))
            .withColumnRenamed("path", "volume_path")
        )
        binary_df.createOrReplaceTempView("_eba_new_docs")

        parsed_df = self.spark.sql(f"""
            SELECT
                file_name,
                category,
                volume_path,
                CAST(
                    ai_parse_document(content, map('mode', 'TEXT')) AS STRING
                ) AS parsed_content,
                '{datetime.now(timezone.utc).isoformat()}' AS parsed_at
            FROM _eba_new_docs
        """)

        (
            parsed_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(self.parsed_table_fqn)
        )

        logger.info(f"✅ Stored {n_new} parsed document(s) → {self.parsed_table_fqn}")
        return n_new
