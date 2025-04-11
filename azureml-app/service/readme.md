# AzureML App Service

This folder contains the implementation of a service that integrates with Azure Machine Learning (AzureML). The service is designed to handle requests and provide responses related to AzureML operations.

## Features
- REST API endpoints for AzureML operations.
- Lightweight and fast execution using `uvicorn`.
- Easy setup with a Conda environment.

## Prerequisites
- Python 3.8 or higher.
- Conda installed on your system.

## Setting Up the Environment
To simplify the setup process, a Conda environment is recommended. Follow these steps:

1. Create a Conda environment:
   ```bash
   conda create -n azureml-service python=3.8 -y
   ```

2. Activate the environment:
   ```bash
   conda activate azureml-service
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Service Locally
To run the service locally using `uvicorn`, follow these steps:

1. Ensure the Conda environment is activated:
   ```bash
   conda activate azureml-service
   ```

2. Start the service with `uvicorn`:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000 --reload
   ```

   - Replace `main:app` with the appropriate module and application name if different.
   - The `--reload` flag enables auto-reloading during development.

3. Access the service:
   Open your browser or use a tool like `curl` to access the service at `http://127.0.0.1:8000`.

## Notes
- Ensure that the `requirements.txt` file is up-to-date with all necessary dependencies.
- For production deployment, consider using a production-grade ASGI server like `gunicorn` with `uvicorn` workers.

## Folder Structure
- `main.py`: Entry point for the service.
- `routes/`: Contains API route definitions.
- `models/`: Defines data models used in the service.
- `utils/`: Utility functions and helpers.

Feel free to explore the code and customize it as needed for your use case.
