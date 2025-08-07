# steps/run_deployment.py
import click
import subprocess
import os
import re
from zenml.client import Client
from zenml.logger import get_logger
from zenml.services import BaseService

# Importing the pipelines from the project
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
    prediction_service_loader,
)

logger = get_logger(__name__)


@click.command(
    help="""
    ZenML deployment script for the customer churn project.

    Run with 'deploy' to create a new model deployment.
    Run with 'predict' to run inference on the deployed model.
    """
)
@click.option(
    "--config",
    type=click.Choice(["deploy", "predict"]),
    default="deploy",
    help="Choose whether to deploy a new model or run predictions on an existing deployment.",
)
@click.option(
    "--data-path",
    type=str,
    default="data/credit_card_churn.csv",
    help="Path to the data file for deployment.",
)
@click.option(
    "--min-accuracy",
    type=float,
    default=0.6,
    help="Minimum accuracy threshold to trigger a new deployment.",
)
@click.option(
    "--pipeline-name",
    type=str,
    default="continuous_deployment_pipeline",
    help="Name of the deployment pipeline to run.",
)
@click.option(
    "--pipeline-step-name",
    type=str,
    default="deploy_bentoml_service",
    help="Name of the step that deploys the service.",
)
def main(
    config: str,
    data_path: str,
    min_accuracy: float,
    pipeline_name: str,
    pipeline_step_name: str,
):
    """
    Main function to run the deployment or prediction pipelines.
    """
    if config == "deploy":
        logger.info("Running the continuous deployment pipeline to build the BentoML model...")
        
        continuous_deployment_pipeline(
            data_path=data_path,
            min_accuracy=min_accuracy
        )
        
        logger.info("Deployment pipeline finished.")
        
        try:
            logger.info("Building the BentoML service bundle...")
            
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            
            # THE DEFINITIVE FIX: Use the '-o tag' flag with 'bentoml build'
            # This flag forces BentoML to output only the tag to stdout,
            # which is far more reliable than parsing the full output.
            build_process = subprocess.run(
                ["bentoml", "build", "-o", "tag"], 
                check=True,  # Raise CalledProcessError if the command fails
                cwd=project_root,
                capture_output=True, # Capture stdout and stderr
                text=True, # Decode as text
                encoding="utf-8" # Specify universal encoding for cross-platform compatibility
            )
            
            # The output is now just the tag, so we clean it and use it directly.
            bento_tag = build_process.stdout.strip()
            
            if not bento_tag:
                 raise ValueError("BentoML build command did not return a tag.")

            logger.info("="*80)
            logger.info("✅ Pipeline finished and Bento was built successfully!")
            logger.info(f"   Bento Tag: {bento_tag}")
            logger.info("\n🚀 To start the prediction server manually, run the following command:")
            logger.info(f"   python steps/serve_bentoml.py \"{bento_tag}\"")
            logger.info("="*80)

        except subprocess.CalledProcessError as e:
            # If the subprocess fails, print its output for easier debugging
            logger.error(f"Failed to build the Bento. Return code: {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            logger.error("Please ensure you have a `bentofile.yaml` and the `bentoml` CLI is installed and configured correctly.")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return

    elif config == "predict":
        # This part of the code remains the same but relies on the server being started manually.
        logger.info("Running inference pipeline...")
        logger.warning("Please ensure the BentoML server is running in a separate terminal before running predictions.")
        
        client = Client()
        # The prediction_service_loader is now a step in the pipeline, which is not called directly.
        # This part of the code should be updated to use the inference pipeline properly.
        # However, for this update, we will assume the manual prediction service loader is still in use.
        service = prediction_service_loader(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            running=True
        )

        if isinstance(service, BaseService) and service.is_running:
            logger.info(f"Found a running service at endpoint: {service.prediction_url}")
            inference_pipeline().run()
        else:
            logger.error("No running service found. Please run `bentoml serve <bento_tag> --port 3000` manually or use the 'serve_bentoml.py' script first.")


if __name__ == "__main__":
    main()