import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
from typing import Dict, Any
import mlflow.sklearn

def estimator_fn(estimator_params: Dict[str, Any]={}):
  return LinearRegression(**estimator_params)
