
import time
import boto3
import requests
from botocore.exceptions import ClientError
from datetime import timedelta


def load_api_key(param_name, profile_name):
    """
    Loads an API key securely from AWS SSM Parameter Store. This function uses the provided
    AWS profile name to create a session and fetches the parameter value associated with
    the given parameter name. The parameter is retrieved with decryption enabled, ensuring
    that secure strings are handled properly.

    Args:
        param_name (str): The name of the parameter to retrieve from the AWS SSM Parameter Store.
        profile_name (str): The AWS profile name to use for establishing the session.

    Returns:
        str: The retrieved parameter value, which represents the API key.
    """

    session = boto3.Session()
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]


def ensure_bucket_exists(bucket, profile_name):
    """
    Ensures the specified Amazon S3 bucket exists and is accessible. If the bucket
    does not exist or is not accessible, a RuntimeError is raised with the
    appropriate message detailing the issue.

    Parameters:
    bucket : str
        The name of the S3 bucket to check for existence.
    profile_name : str
        The AWS profile to use for authentication with the S3 service.

    Raises:
    RuntimeError
        If the specified bucket does not exist or the client does not have
        sufficient access permissions.
    """

    session = boto3.Session()
    s3_client = session.client("s3")
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