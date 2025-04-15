from typing import Optional
import requests
import json
from azure.ai.projects.models import FunctionTool
import os
import difflib

competition_mappings = {
    "Predict Podcast Listening Time": "playground-series-s5e4",
    "Regression of Used Car Prices" : "playground-series-s4e9",
    "Binary Prediction with a Rainfall Dataset": "playground-series-s5e3",
    "Exploring Mental Health Data": "playground-series-s4e11",
    "Binary Classification with a Software Defects Dataset": "playground-series-s3e23",
}
def get_kaggle_competition_name(name: Optional[str] = None) -> str:
    """
    Function to get the Kaggle competition name based on user input by making an API call.
    """
    if name:
        try:
            # Get Kaggle credentials from environment variables
            # kaggle_username = os.environ.get("KAGGLE_USERNAME")
            # kaggle_api_key = os.environ.get("KAGGLE_API_KEY")
            
            # if not kaggle_username or not kaggle_api_key:
            #     return "Kaggle API credentials not configured. Please set KAGGLE_USERNAME and KAGGLE_API_KEY environment variables."
                
            # response = requests.get(
            #     f"https://www.kaggle.com/api/v1/competitions/list?search={name}",
            #     auth=(kaggle_username, kaggle_api_key)
            # )
            # response.raise_for_status()
            # competition_data = response.json()

            # if competition_data and len(competition_data) > 0:
            #     val = competition_data[0].get('url', 'https://www.kaggle.com/competitions')
            #     # Extract the last path component from the URL
            #     competition_name = val.rstrip('/').split('/')[-1]
            matches = difflib.get_close_matches(name, competition_mappings.keys(), n=1, cutoff=0.6)
            if matches:
                competition_name = competition_mappings[matches[0]]
                return json.dumps( {'competition_id': competition_name} )
            else:
                return "No competitions found matching your query."
        except requests.exceptions.RequestException as e:
            return f"Error fetching competition details: {e}"
    else:
        return "No specific competition found."

# Initialize function tool with user functions
tool_functions = FunctionTool(functions=[get_kaggle_competition_name])