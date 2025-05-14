import boto3
import os
import json
import requests
import time
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

def load_api_key(param_name):
    ssm = boto3.client("ssm")
    return ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]

def ensure_bucket_exists(bucket):
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        raise RuntimeError(f"Bucket '{bucket}' does not exist or is not accessible: {e}")

def get_daily_record(ticker, date_str, api_key):
    url = f"https://api.polygon.io/v1/open-close/{ticker}/{date_str}?adjusted=true&apiKey={api_key}"
    res = requests.get(url)
    if res.ok:
        return res.json()
    else:
        print(f"Failed for {date_str}: {res.status_code}")
        return None

def business_days(start_date, end_date):
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current.strftime('%Y-%m-%d')
        current += timedelta(days=1)

def sleep_with_rate_limit(seconds):
    time.sleep(seconds)
