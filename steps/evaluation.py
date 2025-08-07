# steps/evaluation.py
import logging
from typing import Any, Tuple, Annotated
import pandas as pd
import bentoml
from sklearn.metrics import roc_auc_score, accuracy_score
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
import wandb

logger = get_logger(__name__)

# NOTE: The experiment_tracker is used by the ZenML step decorator
# so we need to get the active stack's experiment tracker.
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    bentoml_model_tag: Annotated[str, "bentoml_model_tag"],
    X_test: Annotated[pd.DataFrame, "X_test_scaled"],
    y_test: Annotated[pd.Series, "y_test"],
) -> Tuple[
    Annotated[float, "auc_score"],
    Annotated[dict, "evaluation_metrics"],
    Annotated[Any, "model"],
]:
    """
    Evaluates a trained model on the test set and logs metrics to W&B.
    
    Args:
        bentoml_model_tag: The string tag of the BentoML model to be evaluated.
        X_test: The preprocessed and scaled testing features.
        y_test: The testing labels.
    
    Returns:
        A tuple containing:
        - The AUC score.
        - A dictionary of evaluation metrics.
        - The loaded BentoML model object.
    """
    try:
        logger.info(f"Loading model with tag: {bentoml_model_tag}")
        model = bentoml.xgboost.load_model(bentoml_model_tag)
        bentoml_model = bentoml.models.get(bentoml_model_tag)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "AUC": auc,
            "Accuracy": accuracy,
        }

        logger.info(f"Model evaluation metrics: {metrics}")

        # Log metrics and artifacts to W&B
        # We interact with the active W&B run directly if it exists,
        # as ZenML's experiment_tracker sets it up.
        if wandb.run:
            # Log metrics directly to the active W&B run
            wandb.log(metrics)
            logger.info("Metrics logged to W&B.")

            try:
                artifact = wandb.Artifact(
                    # CORRECTED: Use bentoml_model.tag.name to get the model name
                    name=f"{bentoml_model.tag.name}",
                    type="model",
                    description=f"BentoML model: {bentoml_model.tag}"
                )
                # Add the entire directory where the BentoML model is stored
                artifact.add_dir(bentoml_model.path)
                wandb.log_artifact(artifact)
                logger.info(f"BentoML model '{bentoml_model.tag}' logged as a W&B artifact.")
            except Exception as wandb_log_e:
                logger.error(f"Error logging BentoML model to W&B as artifact: {wandb_log_e}")
        else:
            logger.warning("No active W&B run found. Cannot log metrics or BentoML model as artifact directly.")

        return auc, metrics, model

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise e