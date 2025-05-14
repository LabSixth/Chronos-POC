import json
import boto3

def upload_quarter_to_s3(data, ticker, start_str, s3_bucket, output_prefix, profile_name):
    year = start_str[:4]
    month = int(start_str[5:7])
    quarter = (month - 1) // 3 + 1
    key = f"{output_prefix}/{ticker}_Q{quarter}_{year}.json"
    session = boto3.Session(profile_name=profile_name)
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )
    print(f"Uploaded {len(data)} records to s3://{s3_bucket}/{key}")