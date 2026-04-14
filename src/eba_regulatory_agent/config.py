"""EBA project configuration — catalog, schema, and volume resolution per environment."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession


def _env_or_default(name: str, default: str) -> str:
    """Return env var when non-empty; otherwise return default."""
    value = os.environ.get(name)
    return value if value else default


def _resolve_config_path(config_path: str) -> Path | None:
    """Resolve config path across common notebook/runtime working directories."""
    raw = Path(config_path)
    candidates = [
        raw,
        Path.cwd() / raw,
        Path.cwd() / "project_config.yml",
        Path(__file__).resolve().parents[2] / "project_config.yml",
        Path(__file__).resolve().parent / "project_config.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_env_block_fallback(yaml_text: str, env: str) -> dict[str, str]:
    """Tiny YAML fallback parser for flat env key/value blocks."""
    env_cfg: dict[str, str] = {}
    in_env = False
    env_indent = 0

    for line in yaml_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped == f"{env}:":
            in_env = True
            env_indent = indent
            continue

        if in_env and indent <= env_indent:
            break

        if in_env and ":" in stripped and not stripped.startswith("-"):
            key, value = stripped.split(":", 1)
            env_cfg[key.strip()] = value.strip().strip('"').strip("'")

    return env_cfg


@dataclass
class EbaConfig:
    env: str
    catalog: str
    schema: str
    volume: str
    volume_path: str
    vector_search_endpoint: str
    embedding_endpoint: str
    llm_endpoint: str
    genie_space_id: str = ""
    warehouse_id: str = "4ad246784b070696"
    warehouse_http_path: str = "/sql/1.0/warehouses/4ad246784b070696"

    @property
    def full_schema_name(self) -> str:
        """Fully qualified schema name (catalog.schema)."""
        return f"{self.catalog}.{self.schema}"


@dataclass
class ProjectConfig(EbaConfig):
    """Backward-compatible config model used by legacy notebooks."""

    system_prompt: str = ""
    usage_policy_id: str = ""
    lakebase_project_id: str = ""
    experiment_name: str = "/Shared/eba-regulatory-agent"

    @property
    def experiment_path(self) -> str:
        """Alias used by some notebooks."""
        return self.experiment_name

    @classmethod
    def from_env(cls, env: str = "dev") -> "ProjectConfig":
        """Build ProjectConfig from current environment variables/defaults."""
        base = get_config(env)
        return cls(
            env=base.env,
            catalog=base.catalog,
            schema=base.schema,
            volume=base.volume,
            volume_path=base.volume_path,
            vector_search_endpoint=base.vector_search_endpoint,
            embedding_endpoint=base.embedding_endpoint,
            llm_endpoint=base.llm_endpoint,
            genie_space_id=base.genie_space_id,
            warehouse_id=base.warehouse_id,
            warehouse_http_path=base.warehouse_http_path,
            system_prompt=_env_or_default("EBA_SYSTEM_PROMPT", ""),
            usage_policy_id=_env_or_default("EBA_USAGE_POLICY_ID", ""),
            lakebase_project_id=_env_or_default("EBA_LAKEBASE_PROJECT_ID", ""),
            experiment_name=_env_or_default(
                "EBA_EXPERIMENT_NAME", "/Shared/eba-regulatory-agent"
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Build ProjectConfig from a legacy project_config.yml file.

        Falls back to environment/default values when file parsing is unavailable.
        """
        base = get_config(env)
        path = _resolve_config_path(config_path)
        data: dict[str, Any] = {}
        env_cfg: dict[str, Any] = {}
        system_prompt = ""

        if path is not None:
            try:
                import yaml  # type: ignore

                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
                    env_cfg = (
                        data.get(env, {}) if isinstance(data.get(env, {}), dict) else {}
                    )
                    system_prompt = str(data.get("system_prompt", ""))
            except Exception:
                env_cfg = _parse_env_block_fallback(path.read_text(encoding="utf-8"), env)

        warehouse_id = _env_or_default(
            "EBA_WAREHOUSE_ID", str(env_cfg.get("warehouse_id", base.warehouse_id))
        )

        return cls(
            env=env,
            catalog=_env_or_default(
                "EBA_CATALOG", str(env_cfg.get("catalog", base.catalog))
            ),
            schema=_env_or_default("EBA_SCHEMA", str(env_cfg.get("schema", base.schema))),
            volume=_env_or_default("EBA_VOLUME", str(env_cfg.get("volume", base.volume))),
            volume_path=f"/Volumes/{_env_or_default('EBA_CATALOG', str(env_cfg.get('catalog', base.catalog)))}/{_env_or_default('EBA_SCHEMA', str(env_cfg.get('schema', base.schema)))}/{_env_or_default('EBA_VOLUME', str(env_cfg.get('volume', base.volume)))}",
            vector_search_endpoint=_env_or_default(
                "EBA_VS_ENDPOINT",
                str(env_cfg.get("vector_search_endpoint", base.vector_search_endpoint)),
            ),
            embedding_endpoint=_env_or_default(
                "EBA_EMBEDDING_ENDPOINT",
                str(env_cfg.get("embedding_endpoint", base.embedding_endpoint)),
            ),
            llm_endpoint=_env_or_default(
                "EBA_LLM_ENDPOINT", str(env_cfg.get("llm_endpoint", base.llm_endpoint))
            ),
            genie_space_id=_env_or_default(
                "EBA_GENIE_SPACE_ID",
                str(env_cfg.get("genie_space_id", base.genie_space_id)),
            ),
            warehouse_id=warehouse_id,
            warehouse_http_path=_env_or_default(
                "EBA_WAREHOUSE_HTTP_PATH", f"/sql/1.0/warehouses/{warehouse_id}"
            ),
            system_prompt=_env_or_default("EBA_SYSTEM_PROMPT", str(system_prompt)),
            usage_policy_id=_env_or_default(
                "EBA_USAGE_POLICY_ID", str(env_cfg.get("usage_policy_id", ""))
            ),
            lakebase_project_id=_env_or_default(
                "EBA_LAKEBASE_PROJECT_ID", str(env_cfg.get("lakebase_project_id", ""))
            ),
            experiment_name=_env_or_default(
                "EBA_EXPERIMENT_NAME",
                str(env_cfg.get("experiment_name", "/Shared/eba-regulatory-agent")),
            ),
        )


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
    catalog = _env_or_default("EBA_CATALOG", _CATALOG_MAP.get(env, _CATALOG_MAP["dev"]))
    schema = _env_or_default("EBA_SCHEMA", "tom_schouten")
    volume = _env_or_default("EBA_VOLUME", "eba_knowledge_base")
    vector_search_endpoint = _env_or_default("EBA_VS_ENDPOINT", "eba_vs_endpoint")
    embedding_endpoint = _env_or_default(
        "EBA_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
    )
    llm_endpoint = _env_or_default(
        "EBA_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
    )
    genie_space_id = _env_or_default("EBA_GENIE_SPACE_ID", "")
    warehouse_id = _env_or_default("EBA_WAREHOUSE_ID", "4ad246784b070696")
    warehouse_http_path = _env_or_default(
        "EBA_WAREHOUSE_HTTP_PATH", f"/sql/1.0/warehouses/{warehouse_id}"
    )
    return EbaConfig(
        env=env,
        catalog=catalog,
        schema=schema,
        volume=volume,
        volume_path=f"/Volumes/{catalog}/{schema}/{volume}",
        vector_search_endpoint=vector_search_endpoint,
        embedding_endpoint=embedding_endpoint,
        llm_endpoint=llm_endpoint,
        genie_space_id=genie_space_id,
        warehouse_id=warehouse_id,
        warehouse_http_path=warehouse_http_path,
    )


def load_config(config_path: str | None = None, env: str = "dev") -> ProjectConfig:
    """Backward-compatible loader used by legacy notebook imports."""
    if config_path:
        return ProjectConfig.from_yaml(config_path, env)
    return ProjectConfig.from_env(env)
