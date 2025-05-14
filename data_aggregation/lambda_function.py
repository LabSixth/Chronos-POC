import yaml
from datetime import datetime
from utils.helpers import (
    load_api_key,
    ensure_bucket_exists,
    get_daily_record,
    business_days,
    sleep_with_rate_limit
)
from src.aggregation import upload_quarter_to_s3

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    api_key = load_api_key(config["polygon"]["api_key_ssm_param"])
    ticker = config["polygon"]["ticker"]
    start_date = datetime.strptime(config["polygon"]["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["polygon"]["end_date"], "%Y-%m-%d")
    iam_profile = config["aws"]["iam_profile"]
    s3_bucket = config["aws"]["s3_bucket"]
    output_prefix = config["aws"]["output_prefix"]
    sleep_seconds = config["rate_limit"]["sleep_seconds"]

    ensure_bucket_exists(s3_bucket)

    all_data = []
    for date_str in business_days(start_date, end_date):
        record = get_daily_record(ticker, date_str, api_key)
        if record and record.get("status") == "OK":
            all_data.append(record)
        sleep_with_rate_limit(sleep_seconds)

    upload_quarter_to_s3(all_data, ticker, config["polygon"]["start_date"], s3_bucket, output_prefix)

if __name__ == "__main__":
    main()
