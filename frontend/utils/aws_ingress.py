
import yaml
import os
import boto3
from pathlib import Path

CONFIG_PATH = Path(__file__).joinpath("..", "..", "config.yaml").resolve()
CONFIGS = yaml.safe_load(open(CONFIG_PATH, "r").read())


def get_s3_data() -> None:
    """
    Downloads the latest files from a specified S3 bucket to a local directory.

    This function ensures the existence of a local data folder, establishes an
    Amazon S3 session, and retrieves the latest files from a predefined folder
    within an S3 bucket based on configurations. The files are then stored
    locally in the specified directory.

    Raises:
        KeyError: If expected keys like Bucket_Name, Prefix, or Files are not found in
           the CONFIGS["AWS_Configurations"] dictionary.
        NoCredentialsError: If AWS credentials are not configured.
        ValueError: If the prefix contains no valid subfolders in the S3 bucket.
    """

    # Make sure that the data folder exists
    data_path = Path(__file__).joinpath("..", "..", "data").resolve()
    os.makedirs(data_path, exist_ok=True)

    # Create a S3 session
    session = boto3.Session()
    s3 = session.client("s3")

    # From the S3 bucket, find the latest folder with the artifacts needed
    bucket_name = CONFIGS["AWS_Configurations"]["Bucket_Name"]
    prefix = CONFIGS["AWS_Configurations"]["Prefix"]
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    subfolders = [cp["Prefix"].split("/")[1] for cp in response.get("CommonPrefixes", [])]

    # Get the latest folder and download the files locally
    latest = max(subfolders, key=int)
    files_listing = CONFIGS["AWS_Configurations"]["Files"]

    for file in files_listing:
        s3_prefix = f"{prefix}{latest}/{file}"
        local_path = data_path.joinpath(file)
        s3.download_file(bucket_name, s3_prefix, local_path)
