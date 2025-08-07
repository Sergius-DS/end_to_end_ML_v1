# steps/deployment_trigger.py
from zenml import step

@step
def deployment_trigger(auc: float, min_accuracy: float) -> bool:
    """
    A step that checks if the model's AUC score is above a minimum threshold.

    This step is used to decide whether to trigger a model deployment.
    
    Args:
        auc: The Area Under the Curve score of the trained model.
        min_accuracy: The minimum required accuracy for deployment.

    Returns:
        A boolean indicating whether to deploy the model.
    """
    if auc >= min_accuracy:
        print(f"Model AUC of {auc} is greater than or equal to the minimum accuracy of {min_accuracy}. Deployment will be triggered.")
        return True
    else:
        print(f"Model AUC of {auc} is less than the minimum accuracy of {min_accuracy}. Deployment will be skipped.")
        return False
