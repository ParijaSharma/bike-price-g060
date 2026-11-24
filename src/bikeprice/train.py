import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from bikeprice.clean_rows import clean_rows
from bikeprice.clean_columns import clean_columns
from bikeprice.features import create_pipeline


TARGET = "resale_thousand_rs"

FEATURE_COLUMNS = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]


def get_data_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def train_models(data_path="data/raw/train.csv"):
    data_path = Path(data_path)

    # Read original training data
    df = pd.read_csv(data_path)

    # Apply both teammates' cleaning functions
    df = clean_rows(df)
    df = clean_columns(df)

    # Remove rows without a target
    df = df.dropna(subset=[TARGET]).copy()

    # Select model features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # MLflow configuration
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mp295-bike-price")

    data_sha256 = get_data_hash(data_path)

    experiments = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": None},
    ]

    best_mae = float("inf")
    best_run_id = None

    for settings in experiments:
        model = RandomForestRegressor(
            n_estimators=settings["n_estimators"],
            max_depth=settings["max_depth"],
            random_state=42,
            n_jobs=-1,
        )

        pipeline = create_pipeline(model)

        with mlflow.start_run() as run:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_valid)
            mae = mean_absolute_error(y_valid, predictions)

            mlflow.log_params(settings)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_metric("val_mae", mae)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
            )

            print("Run ID:", run.info.run_id)
            print("Settings:", settings)
            print("Validation MAE:", mae)
            print("-" * 40)

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id

    print("\\nBest run ID:", best_run_id)
    print("Best validation MAE:", best_mae)

    return best_run_id, best_mae


if __name__ == "__main__":
    train_models()
