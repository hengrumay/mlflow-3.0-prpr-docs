# Databricks notebook source
# MAGIC %md
# MAGIC Running this notebook will automatically create a Registered Model in Unity Catalog, as well as a Databricks Job templated as a MLflow 3.0 Deployment Job.

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow-skinny
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Update these values as necessary
model_name = "main.default.example_model"
job_name = "example_deployment_job"

# TODO: Create notebooks for each task and populate the notebook path here
evaluation_notebook_path = "/Workspace/Users/your.username@databricks.com/Evaluation"
approval_notebook_path = "/Workspace/Users/your.username@databricks.com/Approval"
deployment_notebook_path = "/Workspace/Users/your.username@databricks.com/Deployment"

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
            disable_auto_optimization=True,
            max_retries=0,
        ),
        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={"approval_tag_name": "{{task.name}}"}
            ),
            depends_on=[jobs.TaskDependency(task_key="Evaluation")],
            disable_auto_optimization=True,
            max_retries=0,
        ),
        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(notebook_path=deployment_notebook_path),
            depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
            disable_auto_optimization=True,
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

# Create registered model w/ linked deployment job
from mlflow.tracking.client import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
try:
  client.create_registered_model(model_name, deployment_job_id=job_id)
except Exception as e:
  print(e)