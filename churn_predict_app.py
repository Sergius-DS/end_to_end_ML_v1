# churn_predict_app.py
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os # Import the os module to check for file existence

# --- CONFIGURATION ---
ARTIFACT_PATH = "churn_prediction_artifact.pkl"
HIGH_LEVEL_IMAGE_PATH = "_assets/high_level_overview.png"
WHOLE_PIPELINE_IMAGE_PATH = "_assets/training_and_deployment_pipeline_updated.png"


# --- HELPER FUNCTION TO LOAD THE MODEL ---
@st.cache_resource
def load_prediction_artifact(path):
    """
    Loads the pickled model and preprocessor artifact.
    The @st.cache_resource decorator ensures this is loaded only once.
    """
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        return artifact['model'], artifact['preprocessor']
    except FileNotFoundError:
        st.error(f"Error: The artifact file was not found at '{path}'.")
        st.info("Please run the deployment pipeline (`py -m steps.run_deployment --config deploy`) to train a model and generate the artifact file.")
        return None, None

# --- MAIN APP ---
def main():
    st.set_page_config(layout="wide")
    st.title("Customer Churn Prediction (Local Demo)")

    # Load the model and preprocessor
    model, preprocessor = load_prediction_artifact(ARTIFACT_PATH)
    if model is None or preprocessor is None:
        st.stop() # Stop the app if artifacts can't be loaded

    # Check for and display the high-level overview image if it exists
    if os.path.exists(HIGH_LEVEL_IMAGE_PATH):
        high_level_image = Image.open(HIGH_LEVEL_IMAGE_PATH)
        st.image(high_level_image, caption="High Level Pipeline Overview")

    # Check for and display the whole pipeline image if it exists
    if os.path.exists(WHOLE_PIPELINE_IMAGE_PATH):
        whole_pipeline_image = Image.open(WHOLE_PIPELINE_IMAGE_PATH)
        st.image(whole_pipeline_image, caption="Training and Deployment Pipeline")

    st.markdown(
        """
    #### Problem Statement
    The objective here is to predict whether a customer will **churn** (attrite) or remain an **existing customer** based on their banking behavior and demographic features. This app uses a locally loaded model and preprocessor to make predictions.
    """
    )
    st.markdown(
        """
    #### Description of Features
    This app is designed to predict customer churn. Please input the following **influential customer and banking features** to get a prediction:

    | Feature Name | Description | Example Values |
    |---|---|---|
    | **Contacts Count (12 mon)** | Number of contacts in the last 12 months. | 0-6 |
    | **Months Inactive (12 mon)** | Number of months inactive in the last 12 months. | 0-6 |
    | **Total Revolving Balance** | Total revolving balance on the credit card. | Numeric value |
    | **Total Transaction Count** | Total number of transactions in the last 12 months. | Numeric value |
    | **Gender** | Customer's gender. | Male, Female |
    | **Marital Status** | Customer's marital Status. | Married, Single, Divorced, Unknown |
    | **Income Category** | Customer's income category. | < $40K, $40K - $60K, $60K - $80K, $80K - $120K, $120K + , Unknown |
    | **Education Level** | Customer's education level. | High School, Graduate, Uneducated, College, Post-Graduate, Doctorate, Unknown |
    """
    )


    st.sidebar.header("Input Customer Data")

    contacts_count_12_mon = st.sidebar.number_input("Contacts Count (12 mon)", min_value=0, max_value=10, value=2, step=1)
    months_inactive_12_mon = st.sidebar.number_input("Months Inactive (12 mon)", min_value=0, max_value=12, value=2, step=1)
    total_revolving_bal = st.sidebar.number_input("Total Revolving Balance", value=1000.0)
    total_trans_ct = st.sidebar.number_input("Total Transaction Count", value=50, step=1)
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"])
    income_category = st.sidebar.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
    education_level = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Uneducated", "College", "Post-Graduate", "Doctorate", "Unknown"])

    if st.button("Predict Churn"):
        # 1. Create a raw DataFrame from the inputs
        raw_df = pd.DataFrame({
            "Contacts_Count_12_mon": [contacts_count_12_mon],
            "Months_Inactive_12_mon": [months_inactive_12_mon],
            "Total_Revolving_Bal": [total_revolving_bal],
            "Total_Trans_Ct": [total_trans_ct],
            "Gender": [gender],
            "Marital_Status": [marital_status],
            "Income_Category": [income_category],
            "Education_Level": [education_level],
        })

        try:
            # 2. Transform the raw data using the loaded preprocessor
            processed_data = preprocessor.transform(raw_df)

            # 3. Make a prediction using the loaded model
            prediction_probas = model.predict_proba(processed_data)

            # 4. Interpret the result
            prediction_prob = prediction_probas[0][1]
            if prediction_prob > 0.5:
                st.error(f"Prediction: This customer is **likely to churn** (Probability: {prediction_prob:.2f})")
            else:
                st.success(f"Prediction: This customer is **unlikely to churn** (Probability of Staying: {1-prediction_prob:.2f})")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()