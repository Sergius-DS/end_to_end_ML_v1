# pipelines/training_pipeline.py
import pandas as pd
from typing import Tuple, Annotated

# Import necessary ZenML components and steps
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import WANDB

# Import your steps
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df, resample_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


# Define the training pipeline
@pipeline(
    enable_cache=False,
    settings={"docker": DockerSettings(required_integrations=[WANDB])},
)
def train_pipeline(data_path: str, min_accuracy: float = 0.6):
    """
    A training pipeline that loads data, cleans it, resamples it,
    trains an XGBoost model, and evaluates it.

    Args:
        data_path: Path to the raw data file.
    """
    # 1. Data Ingestion
    df = ingest_df(data_path=data_path)

    # 2. Data Cleaning and Preprocessing
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessor, trained_features = clean_df(df)
    
    # 3. Data Resampling for imbalanced data
    X_train_resampled, y_train_resampled = resample_data(X_train_scaled=X_train_scaled, y_train=y_train)

    # 4. Model Training
    # CORRECTED: The argument name X_train_scaled is changed to X_train_resampled
    # to match the function definition, and the return values are correctly unpacked.
    bentoml_model_tag, _ = train_model(
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        preprocessor=preprocessor,
        config=ModelNameConfig(),
    )
    
    # 5. Model Evaluation
    auc_score, _, _ = evaluate_model(
        bentoml_model_tag=bentoml_model_tag, 
        X_test=X_test_scaled, 
        y_test=y_test
    )