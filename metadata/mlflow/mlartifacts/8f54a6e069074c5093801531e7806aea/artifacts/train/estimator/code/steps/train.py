from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Any

# Map estimator names to their respective classes
ESTIMATOR_REGISTRY = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "DecisionTreeRegressor": DecisionTreeRegressor,
}

def estimator_fn(estimator_params: Dict[str, Any] = {}) -> Any:
    """
    Returns the selected estimator with the provided parameters.

    Args:
        estimator_params (dict): Parameters for the estimator.

    Returns:
        estimator: A scikit-learn estimator object.
    """
    estimator_name = estimator_params.get("estimator", "LinearRegression")  # Default to LinearRegression if not specified
    estimator_class = ESTIMATOR_REGISTRY.get(estimator_name)

    if estimator_class is None:
        raise ValueError(f"Unsupported estimator: {estimator_name}")
    
    # Return the estimator with the parameters provided in the recipe.yaml
    return estimator_class(**estimator_params)
