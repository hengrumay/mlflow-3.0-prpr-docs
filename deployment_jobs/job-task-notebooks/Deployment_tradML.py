# Databricks notebook source
# MAGIC %md
# MAGIC This notebook should only be run in a Databricks Job, as part of MLflow 3.0 Deployment Jobs.

# COMMAND ----------

# dbutils.widgets.text("model_name", "")
# dbutils.widgets.text("model_version", "")

# COMMAND ----------

catalog = "mmt"
schema = "mlflow_v3_assessbrickready"
model_fullname = f"{catalog}.{schema}.traditionalML_elasticnet"

# COMMAND ----------

# model_name = dbutils.widgets.get("model_name")
# model_version = dbutils.widgets.get("model_version")

# TODO: Enter serving endpoint name
serving_endpoint_name = model_fullname.split('.')[-1] + "_ep"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Define the model name with catalog and schema
# model_name = f"{catalog}.{schema}.traditionalML_elasticnet"

# List all versions of the model
model_versions = client.search_model_versions(f"name='{model_name}'")

# # Check if version 1 exists
# version_exists = any(mv.version == '1' for mv in model_versions)

if model_versions:
    # Get the latest version of the model
    latest_version_info = model_versions[-1]

    # Print the version information
    print(f"Model Name: {latest_version_info.name}")
    print(f"Version: {latest_version_info.version}")
    print(f"Stage: {latest_version_info.current_stage}")
    print(f"Status: {latest_version_info.status}")
    print(f"Creation Timestamp: {latest_version_info.creation_timestamp}")
    print(f"Last Updated Timestamp: {latest_version_info.last_updated_timestamp}")
    print(f"Description: {latest_version_info.description}")
    print(f"Source: {latest_version_info.source}")
    print(f"Run ID: {latest_version_info.run_id}")
else:
    print("No versions of the model exist.")

# COMMAND ----------

model_version = latest_version_info.version

# COMMAND ----------

# DBTITLE 1,hmm
# InvalidParameterValue: Endpoint name must be maximum 63 characters, and alphanumeric with hyphens and underscores allowed in between.
# File <command-4256738629773894>, line 20

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
  ServedEntityInput,
  EndpointCoreConfigInput
)
from databricks.sdk.errors import ResourceDoesNotExist
w = WorkspaceClient()  # Assumes DATABRICKS_HOST and DATABRICKS_TOKEN are set
served_entities=[
  ServedEntityInput(
    entity_name=model_name,
    entity_version=model_version,
    workload_size="Small",
    scale_to_zero_enabled=True
  )
]

try:
  w.serving_endpoints.update_config(name=serving_endpoint_name, served_entities=served_entities)
except ResourceDoesNotExist:
  w.serving_endpoints.create(name=serving_endpoint_name, config=EndpointCoreConfigInput(served_entities=served_entities))

# COMMAND ----------

## hard to know whether endpoint was actually created -- couldn't see it on serving UI 

# COMMAND ----------

serving_endpoint_name

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Initialize the WorkspaceClient
w = WorkspaceClient()  # Assumes DATABRICKS_HOST and DATABRICKS_TOKEN are set

# Define the serving endpoint name
# serving_endpoint_name = "your_valid_endpoint_name"  # Replace with a valid name

# Retrieve and display the endpoint details
endpoint_details = w.serving_endpoints.get(name=serving_endpoint_name)

# Print the status of the endpoint updates
print(f"Endpoint Name: {endpoint_details.name}")
print(f"State: {endpoint_details.state}")
print(f"Creation Timestamp: {endpoint_details.creation_timestamp}")
print(f"Last Updated Timestamp: {endpoint_details.last_updated_timestamp}")

# COMMAND ----------

import time
from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors import ResourceDoesNotExist, ResourceConflict

# Ensure the serving_endpoint_name is valid
# serving_endpoint_name = "your_valid_endpoint_name"  # Replace with a valid name

w = WorkspaceClient()  # Assumes DATABRICKS_HOST and DATABRICKS_TOKEN are set
served_entities = [
  ServedEntityInput(
    entity_name=model_name,
    entity_version=model_version,
    workload_size="Small",
    scale_to_zero_enabled=True
  )
]

max_retries = 5
retry_delay = 10  # seconds

for attempt in range(max_retries):
  try:
    w.serving_endpoints.update_config(
      name=serving_endpoint_name,
      served_entities=served_entities
    )
    break
  except ResourceConflict:
    if attempt < max_retries - 1:
      time.sleep(retry_delay)
    else:
      raise
  except ResourceDoesNotExist:
    w.serving_endpoints.create(
      name=serving_endpoint_name,
      config=EndpointCoreConfigInput(served_entities=served_entities)
    )
    break

# Retrieve and display the endpoint details
endpoint_details = w.serving_endpoints.get(name=serving_endpoint_name)
pprint(endpoint_details.__dict__)

# COMMAND ----------

## hmmm not exactly clear about how to use these notebooks from the other one 
