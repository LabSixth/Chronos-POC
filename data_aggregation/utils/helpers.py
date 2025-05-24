"""
Helper utilities for stock data aggregation.

This module provides utility functions for API interaction, S3 operations,
date handling, and rate limiting for the stock data aggregation pipeline.
"""

from datetime import timedelta
import time

import boto3
import requests
from botocore.exceptions import ClientError

def load_api_key(param_name):
    """
    Load API key from AWS Systems Manager Parameter Store.

    Args:
        param_name: Name of the parameter containing the API key

    Returns:
        str: The API key value
    """
    ssm = boto3.client("ssm")
    return ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]

def ensure_bucket_exists(bucket):
    """
    Verify that an S3 bucket exists and is accessible.

    Args:
        bucket: Name of the S3 bucket to check

    Raises:
        RuntimeError: If the bucket doesn't exist or isn't accessible
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as exception:
        error_msg = f"Bucket '{bucket}' does not exist or is not accessible: {exception}"
        raise RuntimeError(error_msg) from exception

def get_daily_record(ticker, date_str, api_key):
    """
    Retrieve daily stock data for a specific ticker and date.

    Args:
        ticker: Stock symbol
        date_str: Date in YYYY-MM-DD format
        api_key: API key for Polygon.io

    Returns:
        dict: JSON response from the API or None if request failed
    """
    url = f"https://api.polygon.io/v1/open-close/{ticker}/{date_str}?adjusted=true&apiKey={api_key}"
    res = requests.get(url, timeout=30)
    if res.ok:
        return res.json()

    print(f"Failed for {date_str}: {res.status_code}")
    return None

def business_days(start_date, end_date):
    """
    Generate business days (Monday-Friday) between start and end dates.

    Args:
        start_date: Starting date (datetime object)
        end_date: Ending date (datetime object)

    Yields:
        str: Each business day in YYYY-MM-DD format
    """
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday (0-4)
            yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)

def sleep_with_rate_limit(seconds):
    """
    Sleep for the specified number of seconds to respect API rate limits.

    Args:
        seconds: Number of seconds to sleep
    """
    time.sleep(seconds)
