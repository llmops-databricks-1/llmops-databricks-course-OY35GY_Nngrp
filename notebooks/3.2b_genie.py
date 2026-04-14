# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating a SQL warehouse for Genie
# MAGIC - Configuring a Genie space with data sources
# MAGIC - Starting conversations with Genie
# MAGIC - Using Genie for natural language queries
# MAGIC
# MAGIC **What is Genie?**
# MAGIC - Databricks Genie is an AI-powered data analyst
# MAGIC - Converts natural language questions to SQL queries
# MAGIC - Executes queries and returns results
# MAGIC - Can be integrated with agents via MCP

# COMMAND ----------
from pyspark.sql import SparkSession

from eba_regulatory_agent.config import get_config, get_env

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = get_config(env)

catalog = cfg.catalog
schema = cfg.schema


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check for Existing Genie Space
# MAGIC
# MAGIC First, check if we already have a Genie space configured.

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType
from loguru import logger

w = WorkspaceClient()

# Check if genie_space_id is configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    logger.info(f"Using existing Genie Space from config: {cfg.genie_space_id}")
    space_id = cfg.genie_space_id
    USE_EXISTING_SPACE = True
else:
    # List existing spaces so you can pick one
    existing_spaces = w.genie.list_spaces().spaces or []
    if existing_spaces:
        logger.info(f"Found {len(existing_spaces)} existing Genie space(s):")
        for s in existing_spaces:
            logger.info(f"  ID: {s.space_id}  Title: {s.title}")
        logger.info("Set EBA_GENIE_SPACE_ID env var and re-run cfg cell to use one.")
    else:
        logger.info("No existing Genie spaces found — will create a new one.")
    USE_EXISTING_SPACE = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create SQL Warehouse (if needed)
# MAGIC
# MAGIC Genie requires a SQL warehouse to execute queries.
# MAGIC Skip this if using an existing space.

# COMMAND ----------

if not USE_EXISTING_SPACE and not cfg.warehouse_id:
    # Create a new warehouse for the Genie space
    created = w.warehouses.create(
        name="__2XS_EBA_warehouse",
        cluster_size="2X-Small",
        max_num_clusters=1,
        auto_stop_mins=10,
        warehouse_type=CreateWarehouseRequestWarehouseType("PRO"),
        enable_serverless_compute=True,
        tags=sql.EndpointTags(
            custom_tags=[sql.EndpointTagPair(key="Project", value="eba_regulatory_agent")]
        ),
    ).result()
    warehouse_id = created.id
    logger.info(f"Created warehouse: {warehouse_id}")
else:
    # Use warehouse from config
    warehouse_id = cfg.warehouse_id
    logger.info(f"Using existing warehouse: {warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Genie Space
# MAGIC
# MAGIC Define which tables and columns Genie can access.
# MAGIC Skip this if using an existing space.

# COMMAND ----------

# Configure the Genie space with EBA chunks table
serialzed_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.eba_chunks",
                "column_configs": [
                    {
                        "column_name": "category",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "chunk_index", "get_example_values": True},
                    {"column_name": "chunk_text", "get_example_values": True},
                    {
                        "column_name": "file_name",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "ingestion_timestamp", "get_example_values": True},
                ],
            }
        ]
    },
}

if not USE_EXISTING_SPACE:
    space = w.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=json.dumps(serialzed_space),
        title="eba-regulatory-agent-space",
    )
    space_id = space.space_id
    logger.info(f"Created new Genie Space: {space_id}")
else:
    logger.info(f"Using existing Genie Space: {space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Genie Space

# COMMAND ----------

space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
logger.info(f"Genie Space ID: {space_id}")
logger.info(f"Space config: {json.loads(space.serialized_space)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the data.

# COMMAND ----------

conversation = w.genie.start_conversation_and_wait(
    space_id=space.space_id, content="Find the last 10 papers published"
)

conversation.as_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Continue the Conversation
# MAGIC
# MAGIC Ask follow-up questions in the same conversation.

# COMMAND ----------

message = w.genie.create_message_and_wait(
    space_id=space.space_id,
    conversation_id=conversation.conversation_id,
    content="Return the list of authors of the last 10 papers published",
)

message.as_dict()

# COMMAND ----------
