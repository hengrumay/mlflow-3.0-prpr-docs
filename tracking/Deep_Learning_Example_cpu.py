# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow 3 Deep Learning Example
# MAGIC
# MAGIC In this example, we will first run a model training job, which is tracked as an MLflow Run. Every 10 epochs, we will store model checkpoints, which are tracked as MLflow Logged Models. We will then select the best checkpoint for production deployment.

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow-skinny
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3 torch scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# COMMAND ----------

torch.__version__

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pytorch
from mlflow.entities import Dataset
import torch
import torch.nn as nn
import pandas as pd

# Helper function to prepare data
def prepare_data(df):
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return X, y

# Helper function to compute accuracy
def compute_accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy

# Define a basic PyTorch classifier
class IrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        self.relu(x)
        x = self.fc2(x)
        return x

# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)

# Prepare training data
train_dataset = mlflow.data.from_pandas(train_df, name="train")
X_train, y_train = prepare_data(train_dataset.df)

# Define the PyTorch model and move it to the device
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(iris.target_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scripted_model = IrisClassifier(input_size, hidden_size, output_size).to(device)
scripted_model = torch.jit.script(scripted_model)

# Start a run to represent the training job
with mlflow.start_run():
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: Dataset = mlflow.data.from_pandas(train_df, name="train")
    X_train, y_train = prepare_data(train_dataset.df)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scripted_model.parameters(), lr=0.01)

    for epoch in range(101):  # steps
        X_train, y_train = X_train.to(device), y_train.to(device)
        out = scripted_model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log a checkpoint with metrics every 10 epochs
        if epoch % 10 == 0:
            # Each newly created LoggedModel checkpoint is linked with its
            # name, params, and step
            model_info = mlflow.pytorch.log_model(
                pytorch_model=scripted_model,
                name=f"torch-iris-{epoch}",
                params={
                    "n_layers": 3,
                    "activation": "ReLU",
                    "criterion": "CrossEntropyLoss",
                    "optimizer": "Adam"
                },
                step=epoch,
                input_example=X_train.numpy(),
                model_type="pytorch"  # Specify the model type (missing)
            )
            # Log metric on training dataset at step and link to LoggedModel
            mlflow.log_metric(
                key="accuracy",
                value=compute_accuracy(scripted_model, X_train, y_train),
                step=epoch,
                model_id=model_info.model_id,
                dataset=train_dataset
            )

# COMMAND ----------

## hmm each checkpoint has a saved model with weights ... why not nested run with registered model for this? 
# https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/48716754996f4721b29ea1a531ae5221/models?o=1444828305810485

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Set the registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = MlflowClient()

# List all registered models to verify the model name
registered_models = client.search_registered_models()
for model in registered_models:
    print(model.name)

# Specify the name of the registered model you want to delete
model_name = "mmt.mlflow_v3_assessbrickready.torch-iris-100"

# Check if the model exists before attempting to delete
if any(model.name == model_name for model in registered_models):
    # Delete the registered model
    client.delete_registered_model(name=model_name)
else:
    print(f"Model '{model_name}' does not exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC This example produced one MLflow Run (training_run) and 11 MLflow Logged Models, one for each checkpoint (at steps 0, 10, …, 100). Using MLflow’s UI or search API, we can get the checkpoints and rank them by their accuracy.

# COMMAND ----------

# ranked_checkpoints = mlflow.search_logged_models(output_format="list")
# ranked_checkpoints.sort(
#     key=lambda model: next((metric.value for metric in model.metrics if metric.key == "accuracy"), float('-inf')),
#     reverse=True
# )

# best_checkpoint: mlflow.entities.LoggedModel = ranked_checkpoints[0]
# print(best_checkpoint.metrics[0])

# COMMAND ----------

models_dicts

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, MapType

models_list = mlflow.search_logged_models(output_format="list")
models_dicts = [model.to_dictionary() for model in models_list]

# Define the schema
schema = StructType([
    StructField("name", StringType(), True),
    StructField("version", StringType(), True),
    StructField("params", MapType(StringType(), StringType()), True),
    StructField("experiment_id", StringType(), True),
    StructField("model_id", StringType(), True),
    StructField('source_run_id', StringType(), True),
    StructField("artifact_location", StringType(), True),
    StructField("metrics", StringType(), True),
    StructField("tags", MapType(StringType(), StringType()), True),
    StructField("creation_timestamp", StringType(), True),
    StructField("last_updated_timestamp", StringType(), True)
])

ranked_checkpoints_df = spark.createDataFrame(models_dicts, schema=schema)
display(ranked_checkpoints_df)

# COMMAND ----------

ranked_checkpoints.sort(
    key=lambda model: next((metric.value for metric in model.metrics if metric.key == "accuracy"), float('-inf')),
    reverse=True
)

best_checkpoint: mlflow.entities.LoggedModel = ranked_checkpoints[0]
best_checkpoint.metrics[0]

# COMMAND ----------

best_checkpoint

# COMMAND ----------

best_checkpoint.name

# COMMAND ----------

worst_checkpoint: mlflow.entities.LoggedModel = ranked_checkpoints[-1]
worst_checkpoint.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC Once the best checkpoint is selected, that model can be registered to the model registry. You can also see the model ID, parameters, and metrics in the UC Model Version page

# COMMAND ----------

catalog = "mmt"
schema = "mlflow_v3_assessbrickready"
model_name = best_checkpoint.name+"_cpu"

# COMMAND ----------

mlflow.register_model(f"models:/{best_checkpoint.model_id}", name=f"{catalog}.{schema}.{model_name}")

# COMMAND ----------


