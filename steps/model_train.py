# steps/model_train.py
import logging
from typing import Any, Tuple, Annotated
import pandas as pd
import xgboost as xgb
from zenml import step
from zenml.logger import get_logger
import bentoml
from steps.config import ModelNameConfig
import pickle  # NEW: Import the pickle library

logger = get_logger(__name__)

@step
def train_model(
    X_train_resampled: Annotated[pd.DataFrame, "X_train_resampled"],
    y_train_resampled: Annotated[pd.Series, "y_train_resampled"],
    preprocessor: Any,
    config: ModelNameConfig,
) -> Tuple[
    Annotated[str, "bentoml_model_tag"],
    Annotated[Any, "preprocessor"],
]:
    """
    Trains an XGBoost model, saves it to BentoML, and also saves the
    model and preprocessor to a local pickle file for the Streamlit app.
    """
    try:
        logger.info(f"Starting model training for {config.model_name}...")

        # Initialize the XGBoost classifier directly
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            n_jobs=4,
            random_state=42,
        )

        # Train the model on the entire resampled training set
        model.fit(X_train_resampled, y_train_resampled)
        logger.info("Model training complete.")
        
        # --- NEW SECTION: Save artifact for Streamlit app ---
        logger.info("Saving model and preprocessor to a local pickle file...")
        artifact_for_app = {
            'model': model,
            'preprocessor': preprocessor
        }
        with open("churn_prediction_artifact.pkl", "wb") as f:
            pickle.dump(artifact_for_app, f)
        logger.info("Artifact saved successfully to churn_prediction_artifact.pkl")
        # --- END NEW SECTION ---

        # Save the model to BentoML store (this part remains the same)
        bentoml_model = bentoml.xgboost.save_model(
            name="xgboost_churn",
            model=model,
            signatures={"predict_proba": {"batchable": False}},
            custom_objects={"preprocessor": preprocessor}
        )
        logger.info(f"BentoML model saved with tag: {bentoml_model.tag}")

        return str(bentoml_model.tag), preprocessor

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise e