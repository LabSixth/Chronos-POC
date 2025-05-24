"""
Lambda function for aggregating stock data from Polygon API.

This module retrieves daily stock data for a specific ticker over a date range
and uploads the aggregated results to S3.
"""

from datetime import datetime
import yaml

from src.aggregation import upload_quarter_to_s3
from utils.helpers import (
    load_api_key,
    ensure_bucket_exists,
    get_daily_record,
    business_days,
    sleep_with_rate_limit
)

def main():
    """
    Main entry point for the data aggregation process.

    Loads configuration, retrieves stock data for specified dates,
    and uploads aggregated results to S3.
    """
    with open("config/config.yaml", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    api_key = load_api_key(config["polygon"]["api_key_ssm_param"])
    ticker = config["polygon"]["ticker"]
    start_date = datetime.strptime(config["polygon"]["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["polygon"]["end_date"], "%Y-%m-%d")
    s3_bucket = config["aws"]["s3_bucket"]
    output_prefix = config["aws"]["output_prefix"]
    sleep_seconds = config["rate_limit"]["sleep_seconds"]
    profile_name = config["aws"].get("iam_profile", None)

    ensure_bucket_exists(s3_bucket)

    all_data = []
    for date_str in business_days(start_date, end_date):
        record = get_daily_record(ticker, date_str, api_key)
        if record and record.get("status") == "OK":
            all_data.append(record)
        sleep_with_rate_limit(sleep_seconds)

    # Prepare upload configuration
    upload_config = {
        "ticker": ticker,
        "start_str": config["polygon"]["start_date"],
        "s3_bucket": s3_bucket,
        "output_prefix": output_prefix,
        "profile_name": profile_name
    }

    upload_quarter_to_s3(all_data, upload_config)

if __name__ == "__main__":
    main()
