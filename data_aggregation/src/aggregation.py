"""
Data aggregation and S3 upload utilities.

This module handles the aggregation of stock data and uploading to S3 storage.
"""

import json
import boto3

def upload_quarter_to_s3(data, config):
    """
    Upload aggregated quarterly stock data to S3.

    Args:
        data: List of daily stock records
        config: Dictionary containing:
            - ticker: Stock symbol
            - start_str: Start date string in YYYY-MM-DD format
            - s3_bucket: Target S3 bucket name
            - output_prefix: S3 key prefix for the output file
            - profile_name: Optional AWS profile name
    """
    ticker = config["ticker"]
    start_str = config["start_str"]
    s3_bucket = config["s3_bucket"]
    output_prefix = config["output_prefix"]
    profile_name = config.get("profile_name")

    year = start_str[:4]
    month = int(start_str[5:7])
    quarter = (month - 1) // 3 + 1
    key = f"{output_prefix}/{ticker}_Q{quarter}_{year}.json"

    # Use profile if provided, otherwise use default credentials
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        s3_client = session.client("s3")
    else:
        s3_client = boto3.client("s3")

    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )
    print(f"Uploaded {len(data)} records to s3://{s3_bucket}/{key}")
