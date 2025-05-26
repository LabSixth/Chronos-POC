
from datetime import datetime
import yaml
import logging
from pathlib import Path
from src.polygon_data.aggregation import upload_quarter_to_s3
from src.polygon_data.helpers import (
    load_api_key,
    ensure_bucket_exists,
    get_daily_record,
    business_days,
    sleep_with_rate_limit
)

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def query_polygon(start_date, end_date) -> None:
    """
    Queries historical data for a specific ticker symbol within a given date range, handles
    rate-limited API calls, and uploads the processed data to an AWS S3 bucket.

    Parameters:
        start_date : str
            The start date of the data query in 'YYYY-MM-DD' format.
        end_date : str
            The end date of the data query in 'YYYY-MM-DD' format.

    Raises:
        FileNotFoundError
            If the specified configuration file is not found.
        KeyError
            If mandatory configuration keys are missing from the loaded configuration.
        ValueError
            If API response data format is invalid or incompatible with further processing.
    """

    # Get configurations
    config_file = Path(__file__).joinpath("..", "..", "configs", "config.yaml").resolve()
    with open(config_file, encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    # Modify start date and end date to actual dates
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    api_key = load_api_key(config["polygon"]["api_key_ssm_param"], config["aws"]["iam_profile"])
    ticker = config["polygon"]["ticker"]
    s3_bucket = config["aws"]["s3_bucket"]
    output_prefix = config["aws"]["output_prefix"]
    sleep_seconds = config["rate_limit"]["sleep_seconds"]
    profile_name = config["aws"].get("iam_profile", None)

    ensure_bucket_exists(s3_bucket, profile_name)

    all_data = []
    for date_str in business_days(start_date, end_date):
        logger.info(f"Processing {ticker} for {date_str}")
        record = get_daily_record(ticker, date_str, api_key)

        if record and record.get("status") == "OK":
            all_data.append(record)
        sleep_with_rate_limit(sleep_seconds)

    # Prepare upload configuration
    upload_config = {
        "ticker": ticker,
        "start_str": start_date,
        "s3_bucket": s3_bucket,
        "output_prefix": output_prefix,
        "profile_name": profile_name
    }

    upload_quarter_to_s3(all_data, upload_config)


def main() -> None:
    """
    Main function to process and query stock data for the previous year's quarters.

    This function calculates the current and previous year, then constructs a list of
    quarters (each represented as a dictionary with start and end dates) for the previous
    year. It loops through these quarters and queries stock data for each quarter using
    the `query_polygon` function.

    Raises:
        Any exceptions that may occur during date operations or querying stock data through
        the `query_polygon` function. These should be handled or explicitly raised by the
        `query_polygon` implementation.
    """

    # Get the current year and previous year
    current_year = datetime.now().year
    previous_year = current_year - 1

    # Create a list of quarters for previous years
    quarters = [
        {"start_date": f"{previous_year}-01-01", "end_date": f"{previous_year}-03-31"},
        {"start_date": f"{previous_year}-04-01", "end_date": f"{previous_year}-06-30"},
        {"start_date": f"{previous_year}-07-01", "end_date": f"{previous_year}-09-30"},
        {"start_date": f"{previous_year}-10-01", "end_date": f"{previous_year}-12-31"},
    ]

    # For each of these quarters, loop through and query the stock data from Polygon
    for quarter in quarters:
        query_polygon(quarter["start_date"], quarter["end_date"])


def handler(event, context) -> None:
    """
    AWS Lambda handler.
    Delegates to the main() function which handles all the logic.
    """
    main()
