# Databricks notebook source
# MAGIC %md
# MAGIC Running this notebook will automatically create a Registered Model in Unity Catalog, as well as a Databricks Job templated as a MLflow 3.0 Deployment Job.

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow-skinny
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
print(username)

# COMMAND ----------

# TODO: Update these values as necessary
catalog = "mmt"
schema = "mlflow_v3_assessbrickready"
model_name = "traditionalML_elasticnet"
model_fullname = f"{catalog}.{schema}.{model_name}"

job_name = "mlflow_brickready_tradML_enet_deployment_job"

# TODO: Create notebooks for each task and populate the notebook path here
evaluation_notebook_path = f"/Workspace/Users/{username}/Evaluation"
approval_notebook_path = f"/Workspace/Users/{username}/Approval"
deployment_notebook_path = f"/Workspace/Users/{username}/Deployment"

# COMMAND ----------

# # Create job with necessary configuration to connect to model as deployment job
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service import jobs

# w = WorkspaceClient()
# job_settings = jobs.JobSettings(
#     name=job_name,
#     tasks=[
#         jobs.Task(
#             task_key="Evaluation",
#             notebook_task=jobs.NotebookTask(notebook_path=evaluation_notebook_path),
#             disable_auto_optimization=True,
#             max_retries=0,
#         ),
#         jobs.Task(
#             task_key="Approval_Check",
#             notebook_task=jobs.NotebookTask(
#                 notebook_path=approval_notebook_path,
#                 base_parameters={"approval_tag_name": "{{task.name}}"}
#             ),
#             depends_on=[jobs.TaskDependency(task_key="Evaluation")],
#             disable_auto_optimization=True,
#             max_retries=0,
#         ),
#         jobs.Task(
#             task_key="Deployment",
#             notebook_task=jobs.NotebookTask(notebook_path=deployment_notebook_path),
#             depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
#             disable_auto_optimization=True,
#             max_retries=0,
#         ),
#     ],
#     parameters=[
#         jobs.JobParameter(name="model_name", default=model_name),
#         jobs.JobParameter(name="model_version", default=""),
#     ],
#     queue=jobs.QueueSettings(enabled=True),
#     max_concurrent_runs=1,
# )

# created_job = w.jobs.create(**job_settings.__dict__)
# job_id = created_job.job_id

# COMMAND ----------

# Create job with necessary configuration to connect to model as deployment job
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
job_settings = jobs.JobSettings(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="Evaluation",
            notebook_task=jobs.NotebookTask(notebook_path=evaluation_notebook_path),
            max_retries=0,
        ),
        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={"approval_tag_name": "{{task.name}}"}
            ),
            depends_on=[jobs.TaskDependency(task_key="Evaluation")],
            max_retries=0,
        ),
        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(notebook_path=deployment_notebook_path),
            depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
            max_retries=0,
        ),
    ],
    parameters=[
        jobs.JobParameter(name="model_name", default=model_name),
        jobs.JobParameter(name="model_version", default=""),
    ],
    queue=jobs.QueueSettings(enabled=True),
    max_concurrent_runs=1,
)

created_job = w.jobs.create(**job_settings.__dict__)
job_id = created_job.job_id

# COMMAND ----------

# created_job = w.jobs.create(**job_settings.__dict__)
# job_id = created_job.job_id
print(job_id)

# COMMAND ----------

## created multiple jobs -- everytime the new job_id is updated -- it's not obvious that these jobs were added -- some status notification in the notebook would be helpful. 

# https://e2-demo-field-eng.cloud.databricks.com/jobs?o=1444828305810485&acl=owned_by_me
# I don't understand why it defualts to serverless. because many ML workflows are not supported by serverless


# COMMAND ----------

# DBTITLE 1,NOPE =S
# Create registered model w/ linked deployment job -- I am not sure if this works -- i had to do that manually in the other folder but i don't think the job got linked. =S 
from mlflow.tracking.client import MlflowClient
import traceback

client = MlflowClient(registry_uri="databricks-uc")
try:
  client.create_registered_model(model_name, deployment_job_id=job_id)
except Exception:
  print(traceback.format_exc())

# COMMAND ----------

# DBTITLE 1,delete and recreate ?
from mlflow.tracking.client import MlflowClient
import traceback

client = MlflowClient(registry_uri="databricks-uc")
# model_name = "your_model_name"  # Replace with the model name you want to delete

try:
    client.delete_registered_model(name=model_name)
    print(f"Model '{model_name}' has been deleted.")
except Exception:
    print(traceback.format_exc())

# COMMAND ----------

# DBTITLE 1,?
import traceback
from mlflow.tracking.client import MlflowClient

# Initialize the MLflow client
client = MlflowClient(registry_uri="databricks-uc")

# Define the model name and job ID
# model_name = "your_model_name"  # Replace with your model name
# job_id = "your_job_id"  # Replace with your job ID

try:
    # Create the registered model
    registered_model = client.create_registered_model(model_name)
    
    # Set the deployment job ID as a tag
    client.set_registered_model_tag(
        name=model_name,
        key="deployment_job_id",
        value=job_id
    )
    print(f"Model '{model_name}' created and deployment job ID set.")
except Exception:
    print(traceback.format_exc())

# COMMAND ----------

registered_model

# COMMAND ----------

model_name

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Define the model name with catalog and schema
model_fullname = f"{catalog}.{schema}.traditionalML_elasticnet"

# List all versions of the model
model_versions = client.search_model_versions(f"name='{model_fullname}'")

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

## the model signature /example input seems to be automatic for inference 
## however, no longer have example code for loading model and predicting


# COMMAND ----------

## I am confused about the jobs here -- how do I get the model to be deployed with the lab using job-task-notebooks?
