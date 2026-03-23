"""EBA project configuration — catalog, schema, and volume resolution per environment."""

import os
from dataclasses import dataclass

from pyspark.sql import SparkSession


@dataclass
class EbaConfig:
    env: str
    catalog: str
    schema: str
    volume: str
    volume_path: str


# Catalog names per deployment target (matches databricks.yml targets).
_CATALOG_MAP: dict[str, str] = {
    "dev": "gfmnndipdapmlopsdev",
    "acc": "gfmnndipdapmlopsacc",  # update when acc catalog name is confirmed
    "prd": "gfmnndipdapmlopsprd",  # update when prd catalog name is confirmed
}


def get_env(spark: SparkSession) -> str:
    """
    Resolve deployment environment (dev / acc / prd).

    Resolution order:
    1. DATABRICKS_BUNDLE_TARGET env-var (set by the VS Code Databricks extension)
    2. Spark conf key 'bundle.target' (set by Databricks Asset Bundle jobs)
    3. Falls back to 'dev'
    """
    env = os.environ.get("DATABRICKS_BUNDLE_TARGET")
    if env:
        return env
    try:
        return spark.conf.get("bundle.target", "dev")
    except Exception:
        return "dev"


def get_config(env: str = "dev") -> EbaConfig:
    """
    Return EbaConfig for the given environment.

    Every value can be overridden via environment variables for local testing:
        EBA_CATALOG, EBA_SCHEMA, EBA_VOLUME
    """
    catalog = os.environ.get("EBA_CATALOG", _CATALOG_MAP.get(env, _CATALOG_MAP["dev"]))
    schema  = os.environ.get("EBA_SCHEMA",  "eba_regulatory")
    volume  = os.environ.get("EBA_VOLUME",  "eba_knowledge_base")
    return EbaConfig(
        env=env,
        catalog=catalog,
        schema=schema,
        volume=volume,
        volume_path=f"/Volumes/{catalog}/{schema}/{volume}",
    )
