# steps/ingest_data.py
import logging
import pandas as pd
from typing import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def ingest_df(
    data_path: str,
) -> Annotated[pd.DataFrame, "dataset"]:
    """
    Ingests data from the specified CSV file path and returns it as a Pandas DataFrame.

    Args:
        data_path: Path to the CSV data file.

    Returns:
        A Pandas DataFrame containing the ingested data.
    """
    try:
        logger.info(f"Starting data ingestion from path: {data_path}")
        df = pd.read_csv(data_path)
        logger.info("Data ingestion completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error ingesting data from {data_path}: {e}")
        raise e