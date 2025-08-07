# src/evaluation.py
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd # May be useful for plotting if using matplotlib directly within here
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt # For plotting AUC curve

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating the model
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Calculates the score for the Model
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary for accuracy/precision/recall)
            **kwargs: Additional arguments, e.g., y_proba for AUC.
        Returns:
            The calculated score.
        """
        pass

    @abstractmethod
    def display_results(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Displays or logs comprehensive evaluation results (e.g., reports, plots).
        Args:
            y_true: True labels
            y_pred: Predicted labels
            **kwargs: Additional arguments for display (e.g., y_proba, cv_results).
        """
        pass

class ClassificationEvaluation(Evaluation):
    """
    Evaluation Strategy that provides various classification metrics (AUC, Confusion Matrix, Classification Report).
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Calculates the ROC AUC score.
        Args:
            y_true: True labels.
            y_pred: Predicted labels (binary).
            **kwargs: Must contain 'y_proba' (probabilities for the positive class).
        Returns:
            The ROC AUC score.
        """
        try:
            if 'y_proba' not in kwargs:
                logging.error("y_proba is required for AUC calculation.")
                raise ValueError("y_proba is required for AUC calculation.")

            y_proba = kwargs['y_proba'][:, 1] # Get probabilities for the positive class
            auc = roc_auc_score(y_true, y_proba)
            logging.info(f"Calculated ROC AUC Score: {auc}")
            return auc
        except Exception as e:
            logging.error(f"Error in calculating ROC AUC score: {e}")
            raise e

    def display_results(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Displays confusion matrix, classification report, and ROC AUC score.
        Optionally plots AUC curve from cross-validation results.
        Args:
            y_true: True labels.
            y_pred: Predicted labels (binary).
            **kwargs: Can contain 'y_proba' for AUC, and 'cv_results' for plotting.
        """
        try:
            logging.info("\n--- Classification Evaluation Results ---")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            logging.info("\nConfusion Matrix:")
            logging.info(f"\n{cm}")

            # Classification Report
            report = classification_report(y_true, y_pred)
            logging.info("\nClassification Report:")
            logging.info(f"\n{report}")

            # ROC AUC Score (if probabilities are provided)
            if 'y_proba' in kwargs:
                y_proba = kwargs['y_proba'][:, 1]
                auc = roc_auc_score(y_true, y_proba)
                logging.info(f"\nROC AUC Score on Test Set: {auc}")
            else:
                logging.warning("y_proba not provided, cannot calculate ROC AUC score for display.")

            # Plotting AUC from CV results (from training phase)
            cv_results = kwargs.get('cv_results')
            if cv_results is not None and isinstance(cv_results, pd.DataFrame):
                logging.info("\nPlotting AUC of Training vs Validation in Boosting Rounds...")
                iteraciones = range(1, len(cv_results['test-auc-mean']) + 1)
                plt.figure(figsize=(10, 6))
                plt.plot(iteraciones, cv_results['test-auc-mean'], label='Test AUC')
                plt.plot(iteraciones, cv_results['train-auc-mean'], label='Train AUC')
                plt.xlabel('Número de Rondas de Boosting')
                plt.ylabel('Puntaje AUC')
                plt.title('AUC de Entrenamiento vs Validación en Rondas de Boosting')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # Save plot to a file or show it
                plot_path = "auc_boosting_rounds.png"
                plt.savefig(plot_path)
                logging.info(f"AUC Boosting Rounds plot saved to {plot_path}")
                # plt.show() # Uncomment if you want to display the plot directly during execution
                plt.close() # Close the plot to free memory
            else:
                logging.warning("cv_results not provided or not a DataFrame, skipping AUC boosting rounds plot.")

            # Cross-validation scores (if provided)
            cv_scores_best = kwargs.get('cv_scores_best')
            if cv_scores_best is not None and isinstance(cv_scores_best, np.ndarray):
                logging.info(f"\nCross-validation AUC scores (best model): {cv_scores_best}")
                logging.info(f"Average CV AUC: {np.mean(cv_scores_best)}")
                logging.info(f"Standard Deviation of CV AUC: {np.std(cv_scores_best)}")
            else:
                logging.warning("Cross-validation scores not provided for display.")

        except Exception as e:
            logging.error(f"Error in displaying evaluation results: {e}")
            raise e