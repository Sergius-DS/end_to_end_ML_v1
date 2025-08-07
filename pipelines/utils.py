# pipelines/utils.py
import logging
import pandas as pd
from typing import Any
import os

from zenml.client import Client
from zenml.models import ArtifactVersionResponse
# Assuming your DataCleaning and DataPreProcessStrategy are in src/data_cleaning.py
from src.data_cleaning import DataCleaning, DataPreProcessStrategy


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_data_for_test():
    """
    Loads a sample of raw customer churn data, applies initial preprocessing
    (consistent with DataPreProcessStrategy used in training), and returns
    it as a JSON string for inference.
    """
    try:
        # 1. Load your actual customer churn dataset
        # Ensure 'data/Client_Bank.csv' is the correct path to your raw data
        data_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "Credit_card_churn.csv")
        logging.info(f"Loading raw data from: {data_file_path}")
        df = pd.read_csv(data_file_path)

        # 2. Sample data for inference
        # Adjust n to a reasonable number of samples for testing inference
        df = df.sample(n=50, random_state=42) # Use random_state for reproducibility

        # 3. Apply the *same* initial preprocessing as your DataPreProcessStrategy
        # The DataCleaning class expects a DataFrame and a strategy.
        # This will apply the numerical column handling and dropping of initial irrelevant columns.
        # It's important that DataPreProcessStrategy is designed to work on the raw DataFrame
        # and not assume scaling or one-hot encoding has happened yet.
        preprocess_strategy = DataPreProcessStrategy() # This should be the same strategy used in clean_df
        data_cleaning = DataCleaning(df, preprocess_strategy)
        # This handle_data call should perform the initial processing,
        # but NOT the train/test split or final scaling/OHE.
        # DataPreProcessStrategy's handle_data should return a DataFrame ready for the next steps.
        processed_df_for_inference = data_cleaning.handle_data()

        # 4. Drop the target variable and any other columns that should not be fed to the model
        # (e.g., 'Attrition_Flag' if it's the target, or 'CLIENTNUM' if it was dropped)
        # Replace 'review_score' with your actual target column (e.g., 'Attrition_Flag')
        # If 'Attrition_Flag' is the target, ensure it's dropped here.
        if 'Attrition_Flag' in processed_df_for_inference.columns:
              processed_df_for_inference.drop(["Attrition_Flag"], axis=1, inplace=True)
        # Also drop any other columns that were dropped in DataPreProcessStrategy
        # if they somehow persist here. CLIENTNUM was already identified as one.
        if 'CLIENTNUM' in processed_df_for_inference.columns:
            processed_df_for_inference.drop(["CLIENTNUM"], axis=1, inplace=True)


        # 5. Convert to JSON format that your 'predictor' step can consume
        # The 'predictor' step expects a JSON with 'data', 'columns', 'index'.
        result = processed_df_for_inference.to_json(orient="split")
        logging.info("Prepared test data for inference successfully.")
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e


def read_artifact_from_summary(
    client: Client, artifact_summary: ArtifactVersionResponse
) -> Any:
    """
    Reads the data from an artifact summary (ArtifactVersionResponse).

    This helper function handles the two-step process of fetching the full
    artifact view from its summary and then reading its contents.

    Args:
        client: The ZenML client instance.
        artifact_summary: The ArtifactVersionResponse object from a step's outputs.

    Returns:
        The deserialized data from the artifact.
    """
    # Use the artifact's version ID to fetch the full artifact version object
    full_artifact = client.get_artifact_version(artifact_summary.id)
    
    # Now, read the value from the full artifact version object
    return full_artifact.read()