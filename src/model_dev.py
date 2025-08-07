# src/model_dev.py
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Model(ABC):
    """
    Abstract class for all models.
    """
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Any:
        """
        Trains the model.
        Args:
            X_train: Training data.
            y_train: Training labels.
            **kwargs: Additional keyword arguments for training.
        Returns:
            The trained model.
        """
        pass

class XGBoostModel(Model):
    """
    XGBoostModel for training and hyperparameter tuning.
    """
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params_early_stopping: dict,
        grid_search_params: dict,
        seed: int,
        early_stopping_rounds: int,
        n_splits_cv: int,
        num_boost_round: int,
        n_splits_grid: int
    ) -> Tuple[xgb.XGBClassifier, dict, float]:
        """
        Trains the XGBoost model with early stopping and GridSearchCV.

        Args:
            X_train: Training data.
            y_train: Training labels.
            params_early_stopping: Parameters for the initial XGBoost model with early stopping.
            grid_search_params: Parameters for GridSearchCV.
            seed: Random seed for reproducibility.
            early_stopping_rounds: Number of rounds for early stopping.
            n_splits_cv: Number of folds for cross-validation in early stopping.
            num_boost_round: Maximum number of boosting rounds.
            n_splits_grid: Number of folds for GridSearchCV.

        Returns:
            A tuple containing:
                - The best trained XGBoost model from GridSearchCV.
                - The full cross-validation results from GridSearchCV.
                - The best AUC score found during GridSearchCV.
        """
        try:
            # 1. Early Stopping to find optimal n_estimators
            logging.info("Starting XGBoost early stopping to find optimal n_estimators...")
            dtrain = xgb.DMatrix(X_train, label=y_train)

            cv_results = xgb.cv(
                dtrain=dtrain,
                params=params_early_stopping,
                num_boost_round=num_boost_round,
                nfold=n_splits_cv,
                metrics='auc',
                as_pandas=True,
                seed=seed,
                verbose_eval=10, # Print evaluation results every 10 rounds
                early_stopping_rounds=early_stopping_rounds
            )

            optimal_n_estimators = cv_results['test-auc-mean'].idxmax() + 1
            logging.info(f"Optimal n_estimators found: {optimal_n_estimators}")

            # 2. GridSearchCV for hyperparameter tuning with optimal n_estimators
            logging.info("Starting GridSearchCV for hyperparameter tuning...")
            
            # Initialize XGBClassifier with the optimal n_estimators and other fixed params
            xgb_model = xgb.XGBClassifier(
                n_estimators=optimal_n_estimators,
                objective=params_early_stopping['objective'],
                eval_metric=params_early_stopping['eval_metric'],
                seed=seed,
                # Removed 'use_label_encoder=False' as it's deprecated and not used.
            )

            # Define GridSearchCV
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=grid_search_params,
                scoring='roc_auc',
                cv=StratifiedKFold(n_splits=n_splits_grid, shuffle=True, random_state=seed),
                verbose=1,
                n_jobs=-1 # Use all available CPU cores
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            cv_results_from_training = grid_search.cv_results_
            cv_scores_best_model = grid_search.best_score_

            logging.info(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
            logging.info(f"Best AUC score from GridSearchCV: {cv_scores_best_model}")
            logging.info("Model training and hyperparameter tuning completed successfully.")

            return best_model, cv_results_from_training, cv_scores_best_model

        except Exception as e:
            logging.error(f"Error in training XGBoost model: {e}")
            raise e