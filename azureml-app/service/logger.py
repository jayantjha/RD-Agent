import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
from config import APP_INSIGHTS_CONNECTION_STRING

logger = logging.getLogger("uvicorn")
logger.addHandler(AzureLogHandler(connection_string=APP_INSIGHTS_CONNECTION_STRING))
logger.setLevel(logging.INFO)
