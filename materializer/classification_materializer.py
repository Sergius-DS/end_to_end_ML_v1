# materializer/classification_materializer.py
import os
import pickle
from typing import Any, Type, Union

import numpy as np
import pandas as pd

# Import common classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier # Assuming you use XGBoost for classification
from catboost import CatBoostClassifier # If you use CatBoost for classification
from lightgbm import LGBMClassifier # If you use LightGBM for classification


from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

# Define a default filename for the serialized artifact
DEFAULT_FILENAME = "classification_model_artifact"


class classification_materializer(BaseMaterializer):
    """
    Custom materializer for classification projects.
    It handles serialization and deserialization of common classification models,
    as well as standard data types like pandas DataFrames and NumPy arrays.
    """

    # Define the types that this materializer can handle.
    # Add any other types your pipeline steps might return that you want
    # to handle with this custom materializer.
    ASSOCIATED_TYPES = (
        str,                # For string artifacts
        np.ndarray,         # For NumPy arrays
        pd.Series,          # For pandas Series
        pd.DataFrame,       # For pandas DataFrames
        LogisticRegression, # Scikit-learn Logistic Regression model
        SVC,                # Scikit-learn Support Vector Classifier
        RandomForestClassifier, # Scikit-learn Random Forest Classifier
        GradientBoostingClassifier, # Scikit-learn Gradient Boosting Classifier
        DecisionTreeClassifier, # Scikit-learn Decision Tree Classifier
        XGBClassifier,      # XGBoost Classifier model
        CatBoostClassifier, # CatBoost Classifier model
        LGBMClassifier,     # LightGBM Classifier model
        # If you also need to handle regression models with this materializer,
        # you can add them here as well (e.g., XGBRegressor, RandomForestRegressor)
    )

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[
        str,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        LogisticRegression,
        SVC,
        RandomForestClassifier,
        GradientBoostingClassifier,
        DecisionTreeClassifier,
        XGBClassifier,
        CatBoostClassifier,
        LGBMClassifier,
    ]:
        """
        Loads the artifact from the artifact store and returns it as a Python object.

        Args:
            data_type: The type of the object to be loaded. ZenML uses this
                       to ensure type compatibility.
        Returns:
            The deserialized Python object.
        """
        # Call the base class's handle_input to ensure proper ZenML internal handling
        super().handle_input(data_type)

        # Construct the full file path within the artifact URI
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        # Open the file in binary read mode and load the object using pickle
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(
        self,
        obj: Union[
            str,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            LogisticRegression,
            SVC,
            RandomForestClassifier,
            GradientBoostingClassifier,
            DecisionTreeClassifier,
            XGBClassifier,
            CatBoostClassifier,
            LGBMClassifier,
        ],
    ) -> None:
        """
        Saves the given Python object to the artifact store.

        Args:
            obj: The Python object to be saved.
        """
        # Call the base class's handle_return to ensure proper ZenML internal handling
        super().handle_return(obj)

        # Construct the full file path within the artifact URI
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        # Open the file in binary write mode and save the object using pickle
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)