# scr/data_cleaning.py
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataStrategy(ABC):
    """
    Abstract class defining a strategy for handling data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Abstract method to handle data.
        """
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by handling specific columns, filling missing values,
        dropping irrelevant columns, and mapping/encoding categorical values.
        """
        try:
            logging.info("Starting data preprocessing...")

            # Handle 'Total_Amt_Chng_Q4_Q1' column - convert timestamps/non-numeric to NaN, then fill
            data['Total_Amt_Chng_Q4_Q1'] = pd.to_numeric(data['Total_Amt_Chng_Q4_Q1'], errors='coerce')
            data['Total_Amt_Chng_Q4_Q1'] = data['Total_Amt_Chng_Q4_Q1'].fillna(value=data['Total_Amt_Chng_Q4_Q1'].mean())
            logging.info("Handled 'Total_Amt_Chng_Q4_Q1'.")

            # Handle 'Total_Ct_Chng_Q4_Q1' column - convert timestamps/non-numeric to NaN, then fill
            data['Total_Ct_Chng_Q4_Q1'] = pd.to_numeric(data['Total_Ct_Chng_Q4_Q1'], errors='coerce')
            data['Total_Ct_Chng_Q4_Q1'] = data['Total_Ct_Chng_Q4_Q1'].fillna(value=data['Total_Ct_Chng_Q4_Q1'].mean())
            logging.info("Handled 'Total_Ct_Chng_Q4_Q1'.")

            # Drop columns identified as irrelevant
            cols_to_drop = [
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                'CLIENTNUM'
            ]
            data = data.drop(columns=cols_to_drop, axis=1, errors='ignore')
            logging.info(f"Dropped irrelevant columns: {cols_to_drop}.")

            # Map 'Attrition_Flag' (target column)
            if 'Attrition_Flag' in data.columns:
                data['Attrition_Flag'] = data['Attrition_Flag'].replace({'Existing Customer': 0, 'Attrited Customer': 1})
                logging.info("Mapped 'Attrition_Flag' column.")
            else:
                logging.warning("Attrition_Flag column not found for mapping.")

            # Select influential columns
            columns_to_keep = [
                'Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Total_Revolving_Bal',
                'Total_Trans_Ct', 'Gender', 'Marital_Status',
                'Income_Category', 'Education_Level', 'Attrition_Flag',
                'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1'
            ]
            actual_cols_to_keep = [col for col in columns_to_keep if col in data.columns]
            data = data[actual_cols_to_keep]
            logging.info(f"Selected columns: {actual_cols_to_keep}.")

            # Handle remaining categorical columns using one-hot encoding
            categorical_cols = ['Gender', 'Marital_Status', 'Income_Category', 'Education_Level']
            cols_to_encode = [col for col in categorical_cols if col in data.columns and data[col].dtype == 'object']
            
            if cols_to_encode:
                logging.info(f"One-hot encoding categorical columns: {cols_to_encode}.")
                data = pd.get_dummies(data, columns=cols_to_encode, dummy_na=False)
            else:
                logging.info("No categorical columns found for one-hot encoding.")

            # Iterate through all columns to ensure they are numeric (float64) and handle NaNs
            for col in data.columns:
                # If the column is still an object type, attempt to convert to numeric
                if data[col].dtype == 'object':
                    logging.warning(f"Column '{col}' is still of object type after one-hot encoding. Attempting to convert to numeric.")
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                # Explicitly cast to float64 immediately after numeric conversion.
                # This is crucial to prevent Pandas nullable types (like Float64)
                # from potentially leading to MaskedArrays later.
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].astype(np.float64)
                    logging.debug(f"Column '{col}' explicitly cast to float64.")
                else:
                    logging.warning(f"Column '{col}' is not numeric and cannot be converted to float64. Dropping column.")
                    data = data.drop(columns=[col])
                    continue # Skip further processing for this column if it was dropped

                if data[col].isnull().any():
                    logging.warning(f"Column '{col}' contains NaN values. Filling NaNs.")
                    # Fill NaNs with the mean. If the column is all NaNs, mean() will be NaN,
                    # so we need to handle that case by dropping the column.
                    if not pd.isna(data[col].mean()):
                        data[col] = data[col].fillna(data[col].mean())
                    else:
                        logging.warning(f"Column '{col}' became entirely NaN after coercion and could not be filled. Dropping column.")
                        data = data.drop(columns=[col])
            
            # Final safeguard: Drop any rows with *any* remaining NaN values across the entire DataFrame
            if data.isnull().any().any():
                rows_with_nan_before_drop = data.isnull().any(axis=1).sum()
                logging.warning(f"Found {rows_with_nan_before_drop} rows with NaN values after preprocessing. Dropping these rows to ensure numeric input.")
                data = data.dropna()
                if data.empty:
                    raise ValueError("DataFrame is empty after dropping rows with NaN values. Check data quality or preprocessing steps.")

            logging.info("Data preprocessing completed.")
            return data # Return pd.DataFrame as expected by DataDivideStrategy
        except Exception as e:
            logging.error(f"Error in Preprocessing: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test sets, applying encoding,
    oversampling, and scaling.
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divides data into training and testing sets, applies
        SMOTE for oversampling, and StandardScaler for feature scaling.
        Converts results to Pandas DataFrames/Series before returning.
        """
        try:
            logging.info("Starting data division and scaling...")
            # Separate features (X) and target (y)
            if 'Attrition_Flag' not in data.columns:
                raise ValueError("Target column 'Attrition_Flag' not found in DataFrame after preprocessing.")
            
            X = data.drop(labels='Attrition_Flag', axis=1)
            y = data['Attrition_Flag']

            # Define SEED for reproducibility
            SEED = 88

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)

            # Convert to NumPy arrays for SMOTE and StandardScaler (these libraries prefer NumPy arrays)
            X_train_np = X_train.values.astype(np.float64)
            X_test_np = X_test.values.astype(np.float64)
            y_train_np = y_train.values.astype(np.float64)
            y_test_np = y_test.values.astype(np.float64)

            # Apply SMOTE for oversampling
            logging.info("Applying SMOTE for oversampling...")
            method = SMOTE(random_state=SEED)
            X_train_resampled_np, y_train_resampled_np = method.fit_resample(X_train_np, y_train_np)
            logging.info(f"Original training samples: {len(X_train_np)}, Resampled training samples: {len(X_train_resampled_np)}")

            # Define and apply StandardScaler
            logging.info("Applying StandardScaler for feature scaling...")
            scaler = StandardScaler()
            X_train_scaled_np = scaler.fit_transform(X_train_resampled_np)
            X_test_scaled_np = scaler.transform(X_test_np)
            logging.info("Data division and scaling completed.")

            # Convert back to Pandas DataFrames/Series before returning
            # This is the crucial step to ensure Pandas materializers are used.
            X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns).astype(np.float64)
            X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns).astype(np.float64)
            y_train_resampled_series = pd.Series(y_train_resampled_np, name='Attrition_Flag', dtype=np.float64)
            y_test_series = pd.Series(y_test_np, name='Attrition_Flag', dtype=np.float64)

            return (
                X_train_scaled_df,
                X_test_scaled_df,
                y_train_resampled_series,
                y_test_series
            )
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e

class DataCleaning:
    """
    Class for cleaning data which processes data and divides it into train and test.
    """
    def __init__(self, data_path: str, strategy: DataStrategy):
        """
        Initializes the DataCleaning class with the path to the data and a strategy.
        Loads the data from a CSV file.
        """
        self.data = self._load_data(data_path)
        self.strategy = strategy

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        """
        try:
            logging.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: The file at {data_path} was not found.")
            raise
        except pd.errors.EmptyDataError:
            logging.error(f"Error: The file at {data_path} is empty.")
            raise
        except Exception as e:
            logging.error(f"Error loading data from CSV: {e}")
            raise

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Handles data using the specified strategy.
        """
        try:
            return self.strategy.handle_data(self.data.copy())
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
