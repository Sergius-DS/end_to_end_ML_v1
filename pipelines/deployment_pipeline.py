# pipelines/deployment_pipeline.py
import bentoml
import pandas as pd
from typing import Tuple, Annotated, Any
import os

# --- ZenML Integration Imports ---
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import BENTOML, WANDB
from zenml.integrations.bentoml.model_deployers import BentoMLModelDeployer
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import (
    BentoMLLocalDeploymentService,
)
from zenml.integrations.bentoml.services import (
    BentoMLLocalDeploymentConfig,
)
from zenml.client import Client
from zenml.services import BaseService

# Import your steps from the 'steps' directory
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df, resample_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig
from steps.deployment_trigger import deployment_trigger


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, pipeline_step_name: str, running: bool = True
) -> BaseService:
    """Get the active prediction service from the BentoML deployer."""
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=running,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML deployment service found for pipeline '{pipeline_name}' "
            f"and step '{pipeline_step_name}'. Please run the deployment pipeline first."
        )
    return services[0]


@step(enable_cache=False)
def deploy_bentoml_service(
    bentoml_model_tag: Annotated[str, "bentoml_model_tag"],
    deploy_decision: bool,
    model_name: str = "XGBoost_Churn",  # This will be the ZenML service name
    port: int = 3000,
) -> Annotated[BentoMLLocalDeploymentService | None, "bentoml_service"]:
    """Deploys a BentoML service using the BentoMLModelDeployer."""
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    if deploy_decision:
        print(
            f"Deploying new BentoML service for service name '{model_name}' using "
            f"BentoML model tag '{bentoml_model_tag}'..."
        )
        
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="deploy_bentoml_service",
            running=True,
        )
        if existing_services:
            print("Found an existing service. Stopping it...")
            existing_services[0].stop(timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT)

        # Use the correct, specific configuration class for a local deployment.
        deploy_config = BentoMLLocalDeploymentConfig(
            model_name=model_name,
            model_uri=bentoml_model_tag,
            bento_tag=bentoml_model_tag,
            service_name=model_name,
            port=port,
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="deploy_bentoml_service",
            working_dir=os.getcwd(),
            daemon=False,  # ADDED: Explicitly disable daemon mode for Windows compatibility
        )

        # Deploy the new service by passing the config object and the service type.
        service = model_deployer.deploy_model(
            config=deploy_config,
            service_type=BentoMLLocalDeploymentService.SERVICE_TYPE,
            timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT,
        )
        print(f"Service deployed successfully at {service.prediction_url}")
        return service
    else:
        print("Deployment trigger was not activated. Skipping deployment.")
        return None

# Main continuous deployment pipeline
@pipeline(
    enable_cache=False, 
    settings={
        "docker": DockerSettings(required_integrations=[BENTOML, WANDB]),
        "experiment_tracker": {"name": "wandb_tracker"}
    }
)
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.6,
    model_name: str = "XGBoost_Churn",
    port: int = 3000,
) -> Annotated[BentoMLLocalDeploymentService | None, "bentoml_service"]:
    """A continuous deployment pipeline that trains, evaluates, and deploys a model."""
    df = ingest_df(data_path=data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessor, _ = clean_df(data=df)
    X_train_resampled, y_train_resampled = resample_data(X_train_scaled=X_train_scaled, y_train=y_train)

    bentoml_model_tag, _ = train_model(
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        preprocessor=preprocessor,
        config=ModelNameConfig(),
    )
    
    auc_score, _, _ = evaluate_model(
        bentoml_model_tag=bentoml_model_tag, 
        X_test=X_test_scaled, 
        y_test=y_test
    )

    deploy_decision = deployment_trigger(auc=auc_score, min_accuracy=min_accuracy)

    service = deploy_bentoml_service(
        bentoml_model_tag=bentoml_model_tag,
        deploy_decision=deploy_decision,
        model_name=model_name,
        port=port,
    )
    
    return service

@pipeline(
    enable_cache=False, 
    settings={
        "docker": DockerSettings(required_integrations=[BENTOML, WANDB]),
        "experiment_tracker": {"name": "wandb_tracker"}
    }
)
def inference_pipeline():
    """Inference pipeline that loads a deployed model service and performs predictions."""
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="deploy_bentoml_service",
    )