from pathlib import Path
import logging
import os

import boto3

logger = logging.getLogger(__name__)


def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """Upload all the artifacts in the specified directory to S3.

    Args:
        artifacts: The configured artifact directory.
        config: Dictionary with keys 'bucket_name' and 'prefix'.

    Returns:
        A list of S3 URIs for the uploaded files. URI identifies the location of a resource in S3.
    """
    s3 = boto3.client("s3")  # initialize the S3 client with boto3
    bucket = os.getenv("AWS_BUCKET", config["bucket_name"])  # get the bucket from environment variable or config
    region = os.getenv("AWS_REGION", config.get("region", "us-east-1")) # get the region from environment variable or config
    prefix = os.getenv("AWS_PREFIX", config.get("prefix", "")) # get the region from environment variable or config

    uploaded_uris = []

    for path in artifacts.rglob("*"):  # iterate over all the file paths in the artifacts directory
        if path.is_file():  # check if the current path is a file
            relative_path = path.relative_to(artifacts)  # extract the relative path of the file
            s3_key = f"{prefix}/{artifacts.name}/{relative_path.as_posix()}"  # create the S3 key for uploading the file
            try:
                s3.upload_file(str(path), bucket, s3_key)  # upload the file to S3 (through boto3 object)
                uri = f"s3://{bucket}/{s3_key}"  # create the S3 URI for the uploaded file
                uploaded_uris.append(uri)  # add the S3 URI to the list of uploaded URIs
                logger.info("Uploaded %s to %s", path, uri)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to upload %s: %s", path, e)

    logger.info("Uploaded %d files in total to s3://%s/%s", len(uploaded_uris), bucket, prefix)

    return uploaded_uris
