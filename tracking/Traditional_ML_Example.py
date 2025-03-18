# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow 3 Traditional ML Example
# MAGIC
# MAGIC In this example, we will first run a model training job, which is tracked as an MLflow Run, to produce a trained model, which is tracked as an MLflow Logged Model.
# MAGIC

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow-skinny
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Is this the right version?
import mlflow
mlflow.__version__

# COMMAND ----------

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.entities import Dataset

# Helper function to compute metrics
def compute_metrics(actual, predicted):
    rmse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['quality'] = (iris.target == 2).astype(int)  # Create a binary target for simplicity

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)

# Start a run to represent the training job
with mlflow.start_run() as training_run:
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: Dataset = mlflow.data.from_pandas(train_df, name="train")
    train_x = train_dataset.df.drop(["quality"], axis=1)
    train_y = train_dataset.df[["quality"]]

    # Fit a model to the training dataset
    lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    lr.fit(train_x, train_y)

    # Log the model, specifying its ElasticNet parameters (alpha, l1_ratio)
    # As a new feature, the LoggedModel entity is linked to its name and params
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="elasticnet", ## this is the model name in mlflow tracking | it seems like a model-type.. 
        params={
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
        input_example = train_x ## does this take care of the model signature??
    )

    # Inspect the LoggedModel and its properties
    logged_model = mlflow.get_logged_model(model_info.model_id)
    print(logged_model.model_id, logged_model.params)
    # m-fa4e1bca8cb64971bce2322a8fd427d3, {'alpha': '0.5', 'l1_ratio': '0.5'}

    # Evaluate the model on the training dataset and log metrics
    # These metrics are now linked to the LoggedModel entity
    predictions = lr.predict(train_x)
    (rmse, mae, r2) = compute_metrics(train_y, predictions)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        model_id=logged_model.model_id,
        dataset=train_dataset
    )

    # Inspect the LoggedModel, now with metrics
    logged_model = mlflow.get_logged_model(model_info.model_id)
    print(logged_model.model_id, logged_model.metrics)
    # m-fa4e1bca8cb64971bce2322a8fd427d3, [<Metric: dataset_name='train', key='rmse', model_id='m-fa4e1bca8cb64971bce2322a8fd427d3, value=0.7538635773139717, ...>, ...] 

# COMMAND ----------

# DBTITLE 1,NOTE
# not clear if model signature is required -- I thought UC models require them??

# COMMAND ----------

# DBTITLE 1,print doesn't work ^
logged_model.model_id, logged_model.metrics

# COMMAND ----------

train_x # pandasDF -- would input work for sparkDF? / numpy arrays? maybe the Deeplearning example shows this 

# COMMAND ----------

mlflow.get_logged_model(logged_model.model_id).to_dictionary()

# COMMAND ----------

# MAGIC %md
# MAGIC Some time later, when we get a new evaluation dataset based on the latest production data, we will run a new model evaluation job, which is tracked as a new MLflow Run, to measure the performance of the model on this new dataset.
# MAGIC
# MAGIC This example will produced two MLflow Runs (training_run and evaluation_run) and one MLflow Logged Model (elasticnet). From the resulting Logged Model, we can see all of the parameters and metadata. We can also see all of the metrics linked from the training and evaluation runs.
# MAGIC

# COMMAND ----------

# Start a run to represent the test dataset evaluation job
with mlflow.start_run() as evaluation_run:
  # Load the test dataset with MLflow. We will link test metrics to this dataset.
  test_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(test_df, name="test")
  test_x = test_dataset.df.drop(["quality"], axis=1)
  test_y = test_dataset.df[["quality"]]

  # Load the model
  model = mlflow.sklearn.load_model(f"models:/{logged_model.model_id}")

  # Evaluate the model on the training dataset and log metrics, linking to model
  predictions = model.predict(test_x)
  (rmse, mae, r2) = compute_metrics(test_y, predictions)
  mlflow.log_metrics(
    metrics={
      "rmse": rmse,
      "r2": r2,
      "mae": mae,
    },
    dataset=test_dataset,
    model_id=logged_model.model_id
  )

# COMMAND ----------

# DBTITLE 1,#print doesn't work
print(mlflow.get_logged_model(logged_model.model_id).to_dictionary())

# COMMAND ----------

mlflow.get_logged_model(logged_model.model_id).to_dictionary()

# COMMAND ----------

# MAGIC %md
# MAGIC Now register the model to UC. You can also see the model ID, parameters, and metrics in the UC Model Version page

# COMMAND ----------

# mlflow.register_model(model_info.model_uri, name="catalog.schema.model") ## are we supposed to update this? 

# COMMAND ----------

catalog = "mmt"
# schema = "mlflow_v3_assessbrickready" ## updated but still encountering 
# model_name = 'traditionalML_elasticnet' ## not sure if we are supposed to use the same name (which seems like a model type) in the model logging 
model_name = "tradml_enet"

# COMMAND ----------

mlflow.register_model(model_info.model_uri, name=f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

# can we implement traces in traditionalML or custom pythonFunc mlflow models?

# COMMAND ----------

# DBTITLE 1,NOTES
## I am quite confused hmmm
## TEST https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/runs/1e83a4bd954247ea83563797a805255c?o=1444828305810485
## TRAIN https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/runs/beb5689ffb82447eb4fbd99d62973184?o=1444828305810485 -- this is where the registered model is linked 

## there's no model artifacts like we know of 
# TRAIN https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/runs/beb5689ffb82447eb4fbd99d62973184/artifacts?o=1444828305810485
# TEST https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/runs/1e83a4bd954247ea83563797a805255c/artifacts?o=1444828305810485 

## But for some reason I can't remember how I got to this page
# https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/models?o=1444828305810485
## This is what I was looking for
# https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/models/m-b5c3e5a0bc844ab08f0df904be1a635e/artifacts?o=1444828305810485

## it appears IF you click the MLflow experiments RHS tab (View Experiments) --> takes you to https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79?o=1444828305810485&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D 
# and you have to click the `models PREVIEW` tab to https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/models?o=1444828305810485 
# and the registered model will have the artifacts (?)
# https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/93b6220aaa484e259f3c83b052bb4b79/models/m-b5c3e5a0bc844ab08f0df904be1a635e/artifacts?o=1444828305810485


## example model load/predict is no longer provided? 

# COMMAND ----------

# DBTITLE 1,test another run?
# Start a run to represent the test dataset evaluation job
with mlflow.start_run() as evaluation_run:
  # Load the test dataset with MLflow. We will link test metrics to this dataset.
  test_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(test_df, name="test")
  test_x = test_dataset.df.drop(["quality"], axis=1)
  test_y = test_dataset.df[["quality"]]

  # Load the model
  model = mlflow.sklearn.load_model(f"models:/{logged_model.model_id}")

  # Evaluate the model on the training dataset and log metrics, linking to model
  predictions = model.predict(test_x)
  (rmse, mae, r2) = compute_metrics(test_y, predictions)
  mlflow.log_metrics(
    metrics={
      "rmse": rmse,
      "r2": r2,
      "mae": mae,
    },
    dataset=test_dataset,
    model_id=logged_model.model_id
  )

# COMMAND ----------


