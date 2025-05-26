
import yaml
import os
import boto3
import logging
import botocore.exceptions
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_FILE = Path(__file__).joinpath("..", "..", "..", "config", "default-config.yaml").resolve()
CONFIGS = yaml.safe_load(open(CONFIG_FILE, "r").read())


def data_outgress() -> None:
    """
    Handles data egress by uploading local artifacts to an Amazon S3 bucket.

    This function identifies local artifacts and organizes them in a specified
    Amazon S3 bucket based on folder and file structure. Each sub-directory of
    the artifacts directory is treated as a separate folder in the configured S3
    location, and its files are uploaded accordingly.

    The S3 bucket name and prefix are retrieved from the configuration settings.
    In case of any errors related to S3 operations or unexpected exceptions, the
    function logs the corresponding error and raises the exception for further
    handling.

    Raises:
        botocore.exceptions.ClientError: If an S3 client error occurs during
            the upload process.
        Exception: If any other unexpected errors occur.
    """

    # Get the folder with all the artifacts
    artifacts_folder = Path(__file__).joinpath("..", "..", "..", "artifacts").resolve()
    time_folders = os.listdir(artifacts_folder)

    # Configure to S3
    try:
        session = boto3.Session()
        s3 = session.client("s3")

        # For each of the artifact folder, create one in S3 and upload the data into S3
        bucket_name = CONFIGS["aws_outgress"]["bucket_name"]
        prefix = CONFIGS["aws_outgress"]["prefix"]
        for folder in time_folders:
            local_folder = artifacts_folder.joinpath(folder).resolve()
            s3_prefix = f"{prefix}/{folder}/"

            # Loop through all the contents in the local folder and upload them into S3
            for file_path in local_folder.rglob("*"):
                if file_path.is_file():
                    s3_key = s3_prefix + str(file_path.relative_to(local_folder)).replace("\\", "/")
                    s3.upload_file(str(file_path), bucket_name, s3_key)
                    logger.info(f"Uploaded {file_path} to S3 bucket {bucket_name} with key {s3_key}.")

    except botocore.exceptions.ClientError as e:
        logger.error(f"Failed to upload file to S3: {e.response['Error']['Message']}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error occurred during S3 upload: {str(e)}")
        raise
