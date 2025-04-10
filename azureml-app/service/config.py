#import os

SUBSCRIPTION_ID = "696debc0-8b66-4d84-87b1-39f43917d76c"
RESOURCE_GROUP = "pritamd-rg"
AML_WORKSPACE = "rdagent-poc"
ENVIRONMENT = "rdagent:4"
COMPUTE = "rdagent-cluster"
EXPERIMENT_NAME = "agent-triggered-jobs"
MANAGED_IDENTITY_CLIENT_ID = "a45683ae-a07e-4d5b-9b32-d7d64a0cff00"
PROJECT_CONNECTION_STRING="eastus2.api.azureml.ms;696debc0-8b66-4d84-87b1-39f43917d76c;rg-pritamd-7928_ai;pritamd-agent"
MODEL_DEPLOYMENT_NAME="gpt-4"
AGENT_ID="asst_pPTJItMqueYKSOFkAeg8goSf"
AZURE_STORAGE_ACCOUNT_URL="https://rdagentpoc3274969426.blob.core.windows.net/"
AZURE_STORAGE_CONTAINER_NAME="rd-agent"


APP_INSIGHTS_CONNECTION_STRING = (
    "InstrumentationKey=fe75ff06-e2e4-423c-9684-edbf52f1c6ea;"
    "IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;"
    "LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;"
    "ApplicationId=03c78126-c8b9-457d-8de7-80d44f88a44b"
)

COMMAND_EXECUTE = (
    "wget https://github.com/SunsetWolf/rdagent_resource/releases/download/kaggle_data/kaggle_data.zip && "
    "rm -rf git_ignore_folder && "
    "mkdir -p git_ignore_folder && "
    "unzip kaggle_data.zip -d git_ignore_folder/kaggle_data && "
    "source activate kaggle && "
    "dotenv set KG_LOCAL_DATA_PATH \"$(pwd)/git_ignore_folder/kaggle_data\" && "
    "dotenv run -- python -m rdagent.app.data_science.loop --competition sf-crime"
)

ENV_VARS = {
    # "USE_AZURE": "True",
    # "EMBEDDING_OPENAI_API_KEY": "",# Replace with the right key
    # "EMBEDDING_AZURE_API_BASE":"",# Replace with the corresponding base,
    # "EMBEDDING_AZURE_API_VERSION": "2023-05-15",
    # "EMBEDDING_MODEL": "text-embedding-3-small",
    # "CHAT_OPENAI_API_KEY": "",# Replace with the right key
    # "CHAT_AZURE_API_BASE": "",# Replace with the corresponding base,
    # "CHAT_AZURE_API_VERSION": "2025-01-01-preview",
    # "CHAT_MODEL": "gpt-4o",
    # "DS_CODER_COSTEER_ENV_TYPE": "conda"
}
