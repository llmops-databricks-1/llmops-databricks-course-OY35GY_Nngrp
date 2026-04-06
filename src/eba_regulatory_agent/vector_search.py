"""Vector search management for EBA regulatory documents."""

import time

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from eba_regulatory_agent.config import EbaConfig


class VectorSearchManager:
    """Manages vector search endpoints and indexes for EBA regulatory chunks."""

    def __init__(
        self,
        config: EbaConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: EbaConfig object
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: ID of the usage policy for the endpoint (optional)
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        # w.config.authenticate() resolves the real bearer token through the
        # SDK's full credential chain — OAuth, metadata-service, PAT, SP —
        # without touching MLflow or creating new tokens. Passing the token
        # explicitly sets _using_user_passed_credentials=True in
        # VectorSearchClient, which bypasses its MLflow resolver entirely.
        w = WorkspaceClient()
        bearer_token = (
            w.config.authenticate().get("Authorization", "").removeprefix("Bearer ")
        )
        self.client = VectorSearchClient(
            workspace_url=w.config.host,
            personal_access_token=bearer_token,
            disable_notice=True,
        )
        self.index_name = f"{self.catalog}.{self.schema}.eba_chunks_index"
        self.source_table = f"{self.catalog}.{self.schema}.eba_chunks"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self) -> object:
        """Create or get vector search index.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()
        source_table = f"{self.catalog}.{self.schema}.eba_chunks"

        # Try to get existing index
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found, will create it")

        # Try to create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="chunk_index",
                embedding_source_column="chunk_text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Index exists but get_index failed earlier (transient) — retry
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Sync the vector search index with the source table."""
        self.create_or_get_index()
        self._wait_for_index_ready()
        index = self.client.get_index(index_name=self.index_name)
        logger.info(f"Syncing vector search index: {self.index_name}")
        index.sync()
        logger.info("✓ Index sync triggered")

    def _wait_for_index_ready(
        self,
        timeout: int = 1800,
        poll_interval: int = 15,
    ) -> None:
        """Poll until the index status is ONLINE or timeout is reached.

        Re-fetches the index object on every iteration so the auth token is
        always fresh (metadata-service tokens are short-lived).

        Args:
            timeout: Maximum seconds to wait (default 30 min — endpoint
                     provisioning can take 15-25 min for a new endpoint)
            poll_interval: Seconds between status checks
        """
        _ready = {"ONLINE", "ONLINE_NO_PENDING_UPDATE", "ONLINE_PIPELINE_RUNNING"}
        elapsed = 0
        while elapsed < timeout:
            index = self.client.get_index(index_name=self.index_name)
            status = index.describe().get("status", {}).get("detailed_state", "")
            logger.info(f"Index status: {status} ({elapsed}s elapsed)")
            if status in _ready:
                return
            if "FAILED" in status:
                raise RuntimeError(f"Index entered failed state: {status}")
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Index not ready after {timeout}s")

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
        query_type: str = "ann",
        reranker: object | None = None,
    ) -> dict:
        """Search the vector index.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional filters to apply
            query_type: "ann" for semantic, "hybrid" for semantic + keyword (BM25)
            reranker: Optional DatabricksReranker instance for two-stage retrieval

        Returns:
            Search results dictionary
        """
        index = self.client.get_index(index_name=self.index_name)
        kwargs: dict[str, object] = {
            "query_text": query,
            "columns": ["chunk_index", "chunk_text", "file_name", "category"],
            "num_results": num_results,
            "filters": filters,
            "query_type": query_type,
        }
        if reranker is not None:
            kwargs["reranker"] = reranker
        return index.similarity_search(**kwargs)

    @staticmethod
    def parse_results(results: dict[str, object]) -> list[dict[str, object]]:
        """Convert raw similarity_search response into a list of flat dicts.

        Args:
            results: Raw dict returned by similarity_search()

        Returns:
            List of dicts keyed by column name.
        """
        data_array = results.get("result", {}).get("data_array", [])
        columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
        return [dict(zip(columns, row, strict=False)) for row in data_array]
