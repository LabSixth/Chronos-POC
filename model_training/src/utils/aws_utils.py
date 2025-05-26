"""
AWS Utilities Module.

This module provides functions for interacting with AWS services,
primarily for uploading machine learning artifacts to S3.
"""

from pathlib import Path
import logging
import os
from typing import List

import boto3

logger = logging.getLogger(__name__)


def upload_artifacts(artifacts: Path, config: dict) -> List[str]:
    """Upload all the artifacts in the specified directory to S3.

    Args:
        artifacts: The configured artifact directory.
        config: Dictionary with keys 'bucket_name' and 'prefix'.

    Returns:
        A list of S3 URIs for the uploaded files. URI identifies the location of a resource in S3.
    """
    s3_client = boto3.client("s3")  # initialize the S3 client with boto3
    bucket = os.getenv("AWS_BUCKET", config["bucket_name"])  # get the bucket from env or config
    prefix = os.getenv("AWS_PREFIX", config.get("prefix", ""))  # get the prefix from env or config

    uploaded_uris = []

    # iterate over all the file paths in the artifacts directory
    for path in artifacts.rglob("*"):
        if path.is_file():  # check if the current path is a file
            # extract the relative path of the file
            relative_path = path.relative_to(artifacts)
            # create the S3 key for uploading the file
            s3_key = f"{prefix}/{artifacts.name}/{relative_path.as_posix()}"
            try:
                # upload the file to S3 (through boto3 object)
                s3_client.upload_file(str(path), bucket, s3_key)
                # create the S3 URI for the uploaded file
                uri = f"s3://{bucket}/{s3_key}"
                # add the S3 URI to the list of uploaded URIs
                uploaded_uris.append(uri)
                logger.info("Uploaded %s to %s", path, uri)
            except Exception as exception:  # pylint: disable=broad-except
                logger.error("Failed to upload %s: %s", path, exception)

    logger.info(
        "Uploaded %d files in total to s3://%s/%s",
        len(uploaded_uris),
        bucket,
        prefix
    )

    return uploaded_uris
