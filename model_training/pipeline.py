import argparse
import logging.config
from pathlib import Path
import yaml
import datetime
from src.data_processing.data_loading import load_data
from src.data_processing.data_cleaning import clean_time_series, train_test_split, save_processed_data
from src.data_processing.eda import run_eda
from src.models.prophet_model import ProphetModel
from src.models.chronos_model import ChronosModel
from src.utils.model_evaluation import compute_metrics, save_metrics, prediction_visualization, evaluate_from_config
from src.utils.aws_utils import upload_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_config = config.get("run_config", {})
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "artifacts")) / str(now)
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Run EDA
    logger.info("Starting Exploratory Data Analysis...")
    run_eda(config, artifacts / "eda")
    logger.info("EDA completed")

    # Data loading & cleaning
    data_cfg = config["data_processing"]["data_loading"]
    clean_cfg = config["data_processing"]["data_cleaning"]
    
    # Determine data source from config
    use_s3 = data_cfg.get("use_s3", False)
    df = load_data(data_cfg, local=not use_s3, s3=use_s3)
    logger.info(f"Data loaded from {'S3' if use_s3 else 'local'} source")
    
    df_clean = clean_time_series(df, data_cfg)
    train_df, test_df = train_test_split(df_clean, clean_cfg)
    
    # Save processed data
    save_processed_data(train_df, test_df, artifacts / "processed_data", prefix="stock")

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
    logger.info(f"Prophet metrics: {prophet_metrics}")
    logger.info(f"Chronos metrics: {chronos_metrics}")
    prediction_visualization(test_df["y"], prophet_preds, "Prophet", artifacts)
    prediction_visualization(test_df["y"], chronos_preds, "Chronos", artifacts)

    # Optionally upload artifacts to S3
    aws_config = config.get("aws")
    if aws_config and aws_config.get("upload", False):
        upload_artifacts(artifacts, aws_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ML pipeline")
    parser.add_argument("--config", default="config/default-config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
