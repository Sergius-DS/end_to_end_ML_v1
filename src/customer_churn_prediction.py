# src/customer_churn_prediction.py
import bentoml
import pandas as pd
from typing import Any
import numpy as np # numpy is imported here as it's used for .tolist()

@bentoml.service(
    name="customer_churn_prediction_service",
    resources={"cpu": "2"}  # Optional: Customize resources if needed
)
class CustomerChurnPrediction:
    # Define the runner as a class attribute.
    # BentoML will automatically load this runner when the service starts.
    # Note: Lowercased "xgboost_churn" to match common conventions and potential log warnings.
    churn_runner = bentoml.xgboost.get("xgboost_churn:latest").to_runner()

    # Define the prediction API as a class method.
    # The 'self' parameter is required for class methods.
    @bentoml.api
    async def predict(self, input_df: pd.DataFrame) -> Any:
        """
        Prediction endpoint that takes a raw Pandas DataFrame, applies the
        necessary preprocessing, and returns the prediction probabilities.
        """
        try:
            # Access the preprocessor from the runner's custom_objects.
            # BentoML ensures this custom_object is available once the runner is loaded.
            preprocessor = self.churn_runner.custom_objects['preprocessor']
                        
            # Apply the preprocessor to the input DataFrame.
            # This ensures the data has the same features and scaling as the training data.
            processed_df = preprocessor.transform(input_df)

            # Make predictions using the preprocessed data.
            # Use the runner's async_run method for asynchronous execution,
            # which is recommended for I/O-bound operations.
            prediction_probas = await self.churn_runner.predict_proba.async_run(processed_df)
                        
            # Convert the numpy array of probabilities to a list for JSON serialization.
            # This is necessary because JSON does not directly support numpy arrays.
            prediction_probas_list = prediction_probas.tolist()
                        
            return {"predictions": prediction_probas_list}

        except Exception as e:
            # Provide a more informative error message if something goes wrong during prediction.
            # This helps in debugging issues when the service is running.
            return {"error": f"An error occurred during prediction: {str(e)}"}