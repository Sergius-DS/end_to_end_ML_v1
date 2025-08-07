# steps/run_pipeline.py
import click
import json  # Import the json library
import os
from zenml.client import Client
from pipelines.training_pipeline import train_pipeline
# Import ZenML's universal file I/O utility
from zenml.io import fileio


@click.command()
@click.option(
    "--data-path",
    type=str,
    default="data/Credit_card_churn.csv",  # Or adjust to your preferred default
    help="Path to the data CSV file for training.",
)
def main(data_path: str):
    """
    ZenML script to run the training pipeline and fetch the final AUC score.
    """
    print("Starting ZenML training pipeline...")
    
    # Run the training pipeline.
    # This returns a PipelineRunResponse object.
    pipeline_run_response = train_pipeline(data_path=data_path)
    
    # After the pipeline run is complete, fetch the run object
    client = Client()
    
    # Use the ID from the response to get the full run view
    run_view = client.get_pipeline_run(pipeline_run_response.id)
    
    # Now, get the 'evaluate_model' step from the run view
    evaluate_step = run_view.steps["evaluate_model"]
    
    # Get the summary of the output artifact from the specific run
    auc_artifact_summary = evaluate_step.outputs["auc_score"][0]
    
    # THE DEFINITIVE SOLUTION: Manually read the artifact by inspecting its contents.
    
    # 1. List the files inside the artifact's URI directory to find the data file.
    try:
        artifact_files = fileio.listdir(auc_artifact_summary.uri)
    except Exception as e:
        print(f"Error listing files in artifact URI '{auc_artifact_summary.uri}': {e}")
        return

    # ZenML's materializers can save data in various formats. We will check for common names.
    data_file_name = None
    if "data.json" in artifact_files:
        data_file_name = "data.json"
    elif "data" in artifact_files:
        data_file_name = "data"
    elif "value" in artifact_files:
        data_file_name = "value"
    else:
        # If no common file name is found, raise an error
        raise FileNotFoundError(
            f"Could not find a known data file ('data.json', 'data', or 'value') in the "
            f"artifact directory {auc_artifact_summary.uri}. "
            f"Found files: {artifact_files}"
        )

    # 2. Construct the full path to the data file.
    artifact_data_path = os.path.join(auc_artifact_summary.uri, data_file_name)
    
    print(f"Reading artifact from path: {artifact_data_path}")

    # 3. Use ZenML's fileio to open the file and load the data.
    # We will try loading as JSON first, as that is what was found.
    try:
        with fileio.open(artifact_data_path, "r") as f:
            # Since the file is data.json, we use the json library
            auc_score = json.load(f)
    except Exception as e:
        # Fallback for other potential formats, though json is most likely
        print(f"Failed to load artifact with json. Error: {e}")
        print("Attempting to read as plain text...")
        with fileio.open(artifact_data_path, "r") as f:
            auc_score = float(f.read())

    print(f"Pipeline finished successfully.")
    print(f"Final Model AUC Score: {auc_score:.4f}")


if __name__ == "__main__":
    main()