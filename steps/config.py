# steps/config.py
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model Configuration.
    """
    model_name: str = "xgboost-model"
    n_splits: int = 5
    random_state: int = 42