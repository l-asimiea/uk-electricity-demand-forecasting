"""
Model Development
This script contains code to train ML models using processed electricity data
in search for an adequate forecasting model.
"""

# Imported Libraries
import os
import warnings
from typing import Literal

import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.models.model_utils import (
    model_evaluator,
)
from src.visualisation.plot_utils import plotly_user_standard_settings

plotly_user_standard_settings(pio, px)

# Settings
warnings.filterwarnings("ignore")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
plot_save_path = os.path.join(project_root, "reports/figures/")

MODEL_MAPPING = {
    "random_forest": RandomForestRegressor,
    "gradient_boost": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
}


class Model(BaseModel):
    training_data: pl.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    class ModelConfig(BaseModel):
        mode: Literal["predict_only", "train_and_predict", "retrain_and_predict"] = Field(
            default="predict_only",
            description="Execution mode — controls whether training is triggered.",
        )

        model_type: Literal["lightgbm", "xgboost", "random_forest"] = Field(
            default="lightgbm",
            description="ML base model to use.",
        )
        model_params: dict = Field(
            default={"n_estimators": 100, "max_depth": None, "random_state": 42},
            description="model type and hyperparameters for the selected model type.",
        )
        features: list[str] = Field(..., description="List of feature column names to use for training and prediction.")
        ...

    settings: ModelConfig = Field(default_factory=ModelConfig)

    def setup_model(self):
        """Initializes the model based on the specified type and parameters."""
        model = MODEL_MAPPING.get(self.settings.model_type)
        return model(**self.settings.model_params)

    def input_data_schema():
        """Defines the expected schema for the input data."""
        return ...

    def model_evaluator(fold, y_test, y_pred, model):
        """
        This function takes in model training information involving
        cross validation (cv) splits and calculates model metrics.

        Args:
        fold:       the fold count from the cv split
        y_test:     portion of the target variable used for testing
        y_pred:     portion of the target varibale used for prediction
        model:      the trained model
        model_name: the name of the trained modelß

        Return:
        {model_name, model, fold, RMSE, MAPE, R2}
        """
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        return {
            "model": model,
            "fold": fold,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 2),
        }

    def get_training_features_and_target(self):
        """Extracts features and target variables from the training data."""

        df = self.training_data.sort("settlement_datetime")

        features = df.select(self.settings.features)
        target = df.select(pl.col("transmission_demand"))

        return features, target

    def train_model(self, model, features, target):
        """Trains a model on the training_data."""

        # Apply time-series split cross validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=48 * 365 * 1, gap=48)

        # Loop through cv folds
        for fold, (train_id, test_id) in enumerate(tscv.split(features), start=1):
            print(f"Running Fold {fold} ...")
            print("Splitting data intro training and testing subsets ...")
            x_train, x_test = features.take(train_id), features.take(test_id)
            y_train, y_test = target.take(train_id), target.take(test_id)

            model_result = []
            try:
                model.fit(x_train, y_train)
                model_pred = model.predict(x_test)
                model_result.append(model_evaluator(fold, y_test, model_pred, model))
            except Exception as e:
                print(f"{self.settings.model_type} model training failed on fold {fold}: {e}")

        return model_result

    def predict():
        """Generates prediction(s) using a trained model."""
        return ...

    def save_model():
        """Saves the trained model to a file."""
        return ...

    def load_model():
        """Loads a model from a file."""
        return ...

    def execute(self):
        features, target = self.get_training_features_and_target()
        model = self.setup_model()
        model_results = self.train_model(model, features, target)

        return model_results


# # Feature Importance - Best Model...
# print("Generating feature importance from best model")
# best_model_importance, fig = model_feature_importance(X, best_model_vars)
# fig.show()


"""
Note:
Feature importance plots of the best 2 models (gradient boost and random forest at 5th folds)
showed that the lag features offered the most importance to the model training
especially lag1
"""
# --------------------------------------------------------------------------
# Visualise Model Verification Performance
# model_vars = best_model_vars
# fig = plot_actual_vs_model_pred(model_vars, X, y)
# file_name = f"{plot_save_path}Actual_vs_Predicted_TSD.html"
# fig.show()
# fig.write_html(file_name)

"""
Note:
The actual TSD was compared to predictions made using the best model(Gradient Boost fold 5) and
the second best model (Random Forest fold 5).
The comparison showed that the gradient boost and random forest exibited 5.6% and 6%
average absolute error respectively with mean absolute error of 1705.6MW and 1617MW respectively.
Visually, both models exhbited regions of mainly over prediction than under predictions
at the peaks of the actual TSD with gradient boost performing better (i.e., being closer
to the actual TSD).
"""
# ------------------------------------------------------------------
# Model Optimisation using GridSearchCV
# Define hyperparameter grid for best Gradient Boost model
"""
# Not Done due to high computational cost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
}
# Perform Grid Search
best_model_grid_search = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=best_model_grid_search,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=2,
    n_jobs=-1)

grid_search.fit(X_train,y_train)
print("Best Gradient Boost Parameters:", grid_search.best_params_)
"""


# --------------------------------------------------------------------------
# Save best model only
# best_model = best_model_vars['model']
# joblib.dump(
#     best_model_vars["model"],
#     f"{project_root}/models/{best_model_vars['model_name']}_best_model.pkl",
# )

# # alternate - save model and its metrics
# model_save_filepath = ".pkl"
# with open(model_save_filepath, "wb") as file:
#     pickle.dump(best_model_vars)
