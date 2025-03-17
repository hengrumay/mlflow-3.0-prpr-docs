# Databricks notebook source
# MAGIC %md
# MAGIC # INSTRUCTIONS
# MAGIC Running this notebook will create a Databricks Job templated as a MLflow 3.0 Deployment Job. This job will have three tasks: Evaluation, Approval_Check, and Deployment. The Evaluation task will evaluate the model on a dataset, the Approval_Check task will check if the model has been approved for deployment using UC Tags and the Approval button in the UC Model UI, and the Deployment task will deploy the model to a serving endpoint.
# MAGIC
# MAGIC 1. Copy the [notebooks from the example Github repo](https://github.com/arpitjasa-db/mlflow-3.0-prpr-docs/tree/main/deployment_jobs/job-task-notebooks) into your Databricks Workspace.
# MAGIC 2. Create a UC Model or use an existing one (from the other [MLflow 3 examples](https://github.com/arpitjasa-db/mlflow-3.0-prpr-docs/blob/main/tracking/Traditional_ML_Example.py) for instance).
# MAGIC 3. Update the TODOs/values in the next cell before running the notebook.
# MAGIC 4. After running the notebook, the created job will not be connected to any UC Model. You will still need to **connect the job to a UC Model** in the UC Model UI as indicated in the [documentation](https://docs.google.com/document/d/1bjvwiEckOEMKTupt8ZxAmZLv7Es6Vy0BaAfFbjI1eYc/view?tab=t.0#heading=h.sow1l1a1oeye).

# COMMAND ----------

# TODO: Update these values as necessary
model_name = "main.default.example_model" # The name of the already created UC Model
job_name = "example_deployment_job" # The desired name of the deployment job

# TODO: Create notebooks for each task and populate the notebook path here, replacing the INVALID PATHS LISTED BELOW.
# These paths should correspond to where you put the notebooks templated from the job-task-notebooks directory
# (https://github.com/arpitjasa-db/mlflow-3.0-prpr-docs/tree/main/deployment_jobs/job-task-notebooks)
# in your Databricks workspace.
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
print("Use this Job ID to connect the deployment job to the UC model " + model_name + " as indicated in the UC Model UI/Documentation:")
print(created_job.job_id)
print("\nDocumentation: https://docs.google.com/document/d/1bjvwiEckOEMKTupt8ZxAmZLv7Es6Vy0BaAfFbjI1eYc/view?tab=t.0#heading=h.sow1l1a1oeye")