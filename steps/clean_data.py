# steps/clean_data.py
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Annotated, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_scaled"],
    Annotated[pd.DataFrame, "X_test_scaled"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[ColumnTransformer, "preprocessor"],
    Annotated[list, "trained_feature_names"],
]:
    """
    Cleans, preprocesses, and splits the data.
    - Automatically detects numerical and categorical features.
    - Handles feature engineering and preprocessing.
    """
    try:
        logger.info("Starting data cleaning and preprocessing...")

        # Drop columns identified as irrelevant. Using errors='ignore' for robustness.
        df = data.drop(
            columns=[
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                'CLIENTNUM',
            ],
            axis=1,
            errors='ignore'
        )

        # Handle potential datetime values by converting them to numeric
        for col in ['Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)
                logger.info(f"Cleaned and imputed column: {col}")

        # Map 'Attrition_Flag' to numerical values
        df['Attrition_Flag'].replace({'Existing Customer': 0, 'Attrited Customer': 1}, inplace=True)

        # Select the influential columns as specified in the notebook
        influential_cols = [
            'Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Total_Revolving_Bal',
            'Total_Trans_Ct', 'Attrition_Flag', 'Gender', 'Marital_Status', 'Income_Category', 'Education_Level'
        ]
        
        # Ensure only columns that exist in the DataFrame are selected
        df = df[[col for col in influential_cols if col in df.columns]]

        # Separate the target variable
        y = df['Attrition_Flag']
        X = df.drop(columns=['Attrition_Flag'])

        # Explicitly ensure y is a Series to prevent materialization errors
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                raise ValueError("The target variable 'y' is a DataFrame with multiple columns.")
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Automatically detect numerical and categorical variables from the features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Detected numerical features: {numerical_features}")
        logger.info(f"Detected categorical features: {categorical_features}")

        # Create the preprocessor with ColumnTransformer
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            # ADD THIS LINE TO FIX THE PICKLING ERROR
            n_jobs=1,
            remainder='passthrough'
        )

        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)

        # Transform the training and test data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Convert sparse matrices to dense arrays
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        # Get feature names after one-hot encoding
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        trained_feature_names = numerical_features + list(ohe_feature_names)

        # Convert transformed data back to a DataFrame with correct feature names
        X_train_scaled = pd.DataFrame(X_train_transformed, columns=trained_feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_transformed, columns=trained_feature_names, index=X_test.index)

        logger.info("Data cleaning and preprocessing completed.")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor, trained_feature_names

    except Exception as e:
        logger.error(f"Error during data cleaning step: {e}")
        raise e

@step
def resample_data(
    X_train_scaled: pd.DataFrame, y_train: pd.Series
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_resampled"],
    Annotated[pd.Series, "y_train_resampled"],
]:
    """
    Resamples the training data using SMOTE to handle class imbalance.
    """
    try:
        logger.info("Starting data resampling with SMOTE...")
        smote = SMOTE(random_state=42)
        
        # smote.fit_resample returns numpy arrays, so we explicitly convert them back.
        X_train_resampled_arr, y_train_resampled_arr = smote.fit_resample(X_train_scaled, y_train)
        
        # Convert back to DataFrame and Series with proper indices
        X_train_resampled = pd.DataFrame(X_train_resampled_arr, columns=X_train_scaled.columns)
        
        # This explicit conversion is key to ensuring the output is a single-column Series
        y_train_resampled = pd.Series(y_train_resampled_arr, name=y_train.name)
        
        logger.info(f"Resampled training data shape: {X_train_resampled.shape}")
        
        return X_train_resampled, y_train_resampled
    except Exception as e:
        logger.error(f"Error during data resampling step: {e}")
        raise e