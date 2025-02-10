# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow 3 GenAI Example
# MAGIC
# MAGIC In this example, we will create an agent and then evaluate its performance. First, we will define the agent and log it to MLflow.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3-latest langchain databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

# Define the chain
chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
    max_tokens=2000,
)
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a chatbot that can answer questions about Databricks."),
    ("user", "{messages}"),
  ]
)
chain = prompt | chat_model

# Log the chain with MLflow, specifying its parameters
# As a new feature, the LoggedModel entity is linked to its name and params
model_info = mlflow.langchain.log_model(
  lc_model=chain,
  name="basic_chain",
  params={
    "temperature": 0.1,
    "max_tokens": 2000,
    "prompt_template": str(prompt)
  },
  model_type="agent",
  input_example={"messages": "What is MLflow?"},
)

# Inspect the LoggedModel and its properties
logged_model = mlflow.get_logged_model(model_info.model_id)
print(logged_model.model_id, logged_model.params)
# m-123802d4ba324f4d8baa456eb8b5c061, {'max_tokens': '2000', 'prompt_template': "input_variables=['messages'] messages=[SystemMessagePromptTemplate(...), HumanMessagePromptTemplate(...)]", 'temperature': '0.1'}


# COMMAND ----------

# MAGIC %md
# MAGIC Then, we will interactively query the chain in a notebook to make sure that it’s viable enough for further testing. These traces can be viewed in UI, under the Traces tab of the model details page.

# COMMAND ----------

# Enable autologging so that interactive traces from the chain are automatically linked to its LoggedModel
mlflow.langchain.autolog()
loaded_chain = mlflow.langchain.load_model(f"models:/{logged_model.model_id}")
chain_inputs = [
  {"messages": "What is MLflow?"},
  {"messages": "What is Unity Catalog?"},
  {"messages": "What are user-defined functions (UDFs)?"}
]

for chain_input in chain_inputs:
  loaded_chain.invoke(chain_input)

# COMMAND ----------

# MAGIC %md
# MAGIC Assuming the chain is viable, we will run an evaluation job against the agent to determine whether it’s good enough for further QA by subject matter experts. We can construct a dataset from the agent’s responses on various inputs.
# MAGIC
# MAGIC This example will produce one MLflow Run (evaluation_run), one MLflow Logged Model (basic_chain), and traces from the interactive query and evaluation. We can then see all evaluation metrics for the agent.
# MAGIC

# COMMAND ----------

# Start a run to represent the evaluation job
with mlflow.start_run() as evaluation_run:
  eval_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(
    df=eval_df,
    name="eval_dataset",
  )
  # Run the agent evaluation 
  result = mlflow.evaluate(
    model=f"models:/{logged_model.model_id}",
    data=eval_dataset,
    model_type="databricks-agent"
  )
  # Log evaluation metrics and associate with agent
  mlflow.log_metrics(
    metrics=result.metrics,
    dataset=eval_dataset,
    # Specify the ID of the agent logged above
    model_id=logged_model.model_id
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now register the model to UC. You can also see the model ID, parameters, and metrics in the UC Model Version page

# COMMAND ----------

mlflow.register_model(logged_model.model_uri, name="catalog.schema.model")