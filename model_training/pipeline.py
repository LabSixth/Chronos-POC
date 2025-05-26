"""
Unified machine learning pipeline for stock price prediction.

This module orchestrates the entire ML workflow including:
- Data loading and cleaning
- Exploratory data analysis
- Model training (Prophet and Chronos)
- Model evaluation and visualization
- Artifact management
"""

import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
from src.aws_utils import aws_data_ingress, aws_data_outgress
from src.data_processing.data_cleaning import (
    clean_time_series, save_processed_data, train_test_split
)
from src.data_processing.data_loading import load_data
from src.data_processing.eda import run_eda
from src.models.chronos_model import ChronosModel
from src.models.prophet_model import ProphetModel
from src.utils.model_evaluation import (
    evaluate_from_config,
    prediction_visualization,
    save_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str):
    """
    Execute the ML pipeline based on configuration.

    Args:
        config_path: Path to the configuration YAML file
    """

    # Ingress data from S3
    aws_data_ingress.data_ingress()

    # Load config
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    run_config = config.get("run_config", {})
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "artifacts")) / str(now)
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.dump(config, file)

    # Run EDA
    logger.info("Starting Exploratory Data Analysis...")
    run_eda(config, artifacts / "eda")
    logger.info("EDA completed")

    # Data loading & cleaning
    data_cfg = config["data_processing"]["data_loading"]
    clean_cfg = config["data_processing"]["data_cleaning"]

    # Determine data source from config
    use_s3 = data_cfg.get("use_s3", False)
    stock_data = load_data(data_cfg, local=not use_s3, use_s3=use_s3)
    logger.info("Data loaded from %s source", 'S3' if use_s3 else 'local')

    clean_data = clean_time_series(stock_data, clean_cfg)
    train_df, test_df = train_test_split(clean_data, clean_cfg)

    # Save processed data
    save_processed_data(
        train_df,
        test_df,
        artifacts / "processed_data",
        prefix="stock"
    )

    # Prophet Model
    prophet = ProphetModel(config)
    prophet.hyperparameter_optimization(train_df)
    prophet.fit(train_df)
    prophet_preds = prophet.predict(test_df)
    prophet.save_model(artifacts / "prophet_model.pkl")
    prophet.save_prediction(prophet_preds, artifacts / "prophet_preds.csv")

    # Chronos Model
    chronos = ChronosModel(config)
    chronos.fit(train_df)
    chronos_preds = chronos.predict(test_df)
    chronos.save_prediction(chronos_preds, artifacts / "chronos_preds.csv")

    # Evaluation & Visualization
    metrics_cfg = config["model_evaluation"]
    prophet_metrics = evaluate_from_config(test_df["y"], prophet_preds, metrics_cfg)
    chronos_metrics = evaluate_from_config(test_df["y"], chronos_preds, metrics_cfg)
    save_metrics(prophet_metrics, artifacts / "prophet_metrics.yaml")
    save_metrics(chronos_metrics, artifacts / "chronos_metrics.yaml")
    logger.info("Prophet metrics: %s", prophet_metrics)
    logger.info("Chronos metrics: %s", chronos_metrics)
    prediction_visualization(test_df["y"], prophet_preds, "Prophet", artifacts)
    prediction_visualization(test_df["y"], chronos_preds, "Chronos", artifacts)

    # Complete pipeline by uploading data to S3
    aws_data_outgress.data_outgress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ML pipeline")
    parser.add_argument(
        "--config",
        default="config/default-config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)
