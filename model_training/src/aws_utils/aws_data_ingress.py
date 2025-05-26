
import yaml
import os
import boto3
import logging
import botocore.exceptions
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_FILE = Path(__file__).joinpath("..", "..", "..", "config", "default-config.yaml").resolve()
CONFIGS = yaml.safe_load(open(CONFIG_FILE, "r").read())


def data_ingress() -> None:
    """
    Handles the data ingress process by creating a local folder structure, if required,
    and downloading a specified file from an AWS S3 bucket to the local system.

    This function uses configurations from a predefined `CONFIGS` dictionary for setting
    the S3 bucket name, the file name, and the local storage location. It ensures that
    the local directory structure exists before attempting the download. In case of any
    error during the S3 download process, relevant exceptions are raised and logged.

    Raises:
        botocore.exceptions.ClientError: If an error response is returned by the AWS services.
        Exception: For any other unexpected errors.
    """

    # Get the location for data ingress, create one if it does not exist
    folder_name = CONFIGS["data_processing"]["data_loading"]["local_path"].split("/")[0]
    file_name = CONFIGS["data_processing"]["data_loading"]["local_path"].split("/")[-1]
    data_folder = Path(__file__).joinpath("..", "..", "..", folder_name).resolve()
    os.makedirs(data_folder, exist_ok=True)

    # Configurate to S3
    try:
        session = boto3.Session()
        s3 = session.client("s3")

        # Download the file locally
        s3.download_file(
            CONFIGS["data_processing"]["data_loading"]["bucket_name"],
            CONFIGS["data_processing"]["data_loading"]["file_name"],
            data_folder.joinpath(file_name).resolve().__str__()
        )

    except botocore.exceptions.ClientError as e:
        logger.error(f"Failed to download file from S3: {e.response['Error']['Message']}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error occurred during S3 download: {str(e)}")
        raise
