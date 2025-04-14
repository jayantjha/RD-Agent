#import os

SUBSCRIPTION_ID = "696debc0-8b66-4d84-87b1-39f43917d76c"
RESOURCE_GROUP = "pritamd-rg"
AML_WORKSPACE = "rdagent-poc"
ENVIRONMENT = "rdagent:8"
COMPUTE = "rdagent-cluster"
EXPERIMENT_NAME = "agent-triggered-jobs"
MANAGED_IDENTITY_CLIENT_ID = "a45683ae-a07e-4d5b-9b32-d7d64a0cff00"
PROJECT_CONNECTION_STRING="eastus2.api.azureml.ms;696debc0-8b66-4d84-87b1-39f43917d76c;rg-pritamd-7928_ai;pritamd-agent"
MODEL_DEPLOYMENT_NAME="gpt-4"
AGENT_ID="asst_pPTJItMqueYKSOFkAeg8goSf"
AZURE_STORAGE_ACCOUNT_URL="https://rdagentpoc3274969426.blob.core.windows.net/"
AZURE_STORAGE_CONTAINER_NAME="rd-agent"
FAKE_THREAD_ID="thread_KosU4pD7ZxvUaXLBWeTE1qoC"
FAKE_JOB=True

APP_INSIGHTS_CONNECTION_STRING = (
    "InstrumentationKey=fe75ff06-e2e4-423c-9684-edbf52f1c6ea;"
    "IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;"
    "LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;"
    "ApplicationId=03c78126-c8b9-457d-8de7-80d44f88a44b"
)

COMMAND_EXECUTE = (
    "/workspace/startup.sh "
    "wget https://github.com/SunsetWolf/rdagent_resource/releases/download/kaggle_data/kaggle_data.zip && "
    "rm -rf git_ignore_folder && "
    "mkdir -p git_ignore_folder && "
    "unzip kaggle_data.zip -d git_ignore_folder/kaggle_data && "
    "source activate kaggle && "
    "dotenv set KG_LOCAL_DATA_PATH \"$(pwd)/git_ignore_folder/kaggle_data\" && "
    "dotenv set DS_SESSION_ROOT_PATH \"$(pwd)/rdagent/sessions\" && "
    "dotenv run -- python -m rdagent.app.data_science_msft.loop --competition playground-series-s4e9"
)

ENV_VARS = {
    "EMBEDDING_MODEL": "azure/text-embedding-3-small",
    "CHAT_MODEL": "azure/o3-mini",
    "DS_CODER_COSTEER_ENV_TYPE": "conda",
    "PROJECT_CONN_STRING": "eastus2.api.azureml.ms;696debc0-8b66-4d84-87b1-39f43917d76c;rg-pritamd-7928_ai;pritamd-agent",
    "DS_SESSION_ROOT_PATH": "./git_ignore_folder/rdagent/sessions",
    "DS_USE_MLE_BENCHMARK": "FALSE",
    "DS_ENABLE_MODEL_DUMP": "True",
    "DS_CODER_ON_WHOLE_PIPELINE": "True",
    "DS_IF_USING_MLE_DATA": "FALSE",
    "DS_ENABLE_DOC_DEV": "TRUE",
    "DS_AZURE_STORAGE_ACCOUNT_URL": "https://rdagentpoc3274969426.blob.core.windows.net/",
    "DS_AZURE_STORAGE_CONTAINER_NAME": "rd-agent",
    "THREAD_ID": "thread_KosU4pD7ZxvUaXLBWeTE1qoC",
    "AZURE_API_BASE": "https://ai-pritamdagenthub298066461577.openai.azure.com",
    "AZURE_API_KEY": "",
    "BACKEND": "rdagent.oai.backend.LiteLLMAPIBackend",
    "CHAT_TEMPERATURE": "1",
    "USE_CHAT_CACHE": "True",
    "USE_EMBEDDING_CACHE": "True",
    "COSTEER_V2_QUERY_FORMER_TRACE_LIMIT": "3",
    "MAX_RETRY": "12000",
    "RETRY_WAIT_SECONDS": "5"
}
