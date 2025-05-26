"""
Glue job to process and consolidate Polygon stock data.

This script reads quarterly JSON files from S3, processes them,
and consolidates the data into a single parquet file.
"""

import datetime
import re
from typing import List, Tuple

import boto3
from awsglue.context import GlueContext  # pylint: disable=import-error
from awsglue.job import Job  # pylint: disable=import-error
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, input_file_name, lit, struct


def initialize_glue_job() -> Tuple[GlueContext, Job]:
    """
    Initialize SparkContext, GlueContext, and a Glue Job.
    """
    spark_context = SparkContext()
    glue_context = GlueContext(spark_context)
    job = Job(glue_context)
    return glue_context, job


def load_and_enrich_json(
    glue_context: GlueContext,
    input_paths: List[str]
) -> DataFrame:
    """
    Read multiline JSON arrays from S3, nest all original fields
    under a "json" struct, and add pull_time + source_path metadata.
    """
    spark = glue_context.spark_session
    raw = (
        spark.read
        .option("multiLine", True)
        .json(input_paths)
    )
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    nested = struct(*raw.columns).alias("json")

    return (
        raw
        .withColumn("json", nested)
        .withColumn("pull_time", lit(today))
        .withColumn("source_path", input_file_name())
        .select("json", "pull_time", "source_path")
    )


def flatten_json_struct(dataframe: DataFrame, struct_col: str = "json") -> DataFrame:
    """
    Expand every field inside struct_col into its own top-level column,
    preserving pull_time and source_path.
    """
    fields = dataframe.schema[struct_col].dataType.fieldNames()
    select_cols = [
        col(f"{struct_col}.{f}").alias(f) for f in fields
    ] + [col("pull_time"), col("source_path")]

    return dataframe.select(*select_cols)


def write_coalesced_and_copy_and_cleanup(
    dataframe: DataFrame,
    temp_prefix: str,
    final_uri: str
) -> None:
    """
    1) dataframe.coalesce(1).write.parquet(temp_prefix)
    2) copy the single part-*.parquet to final_uri
    3) delete ALL temp_prefix objects
    """
    # write a single file to S3 temp prefix
    dataframe.coalesce(1).write.mode("overwrite").parquet(temp_prefix)

    # parse bucket/key
    def parse_s3(uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and key parts."""
        _, rest = uri.split("://", 1)
        bucket, *key = rest.split("/", 1)
        return bucket, key[0] if key else ""

    temp_bucket, temp_key_prefix = parse_s3(temp_prefix)
    final_bucket, final_key = parse_s3(final_uri)

    s3_client = boto3.client("s3")

    # list temp objects
    resp = s3_client.list_objects_v2(Bucket=temp_bucket, Prefix=temp_key_prefix)
    contents = resp.get("Contents", [])

    # find the part file
    part = next(
        obj["Key"]
        for obj in contents
        if re.match(
            rf"^{re.escape(temp_key_prefix)}/part-.*\.parquet$",
            obj["Key"])
    )

    # copy part → final
    s3_client.copy_object(
        Bucket=final_bucket,
        CopySource={"Bucket": temp_bucket, "Key": part},
        Key=final_key
    )

    # clean up temp (delete every object under the temp prefix)
    for obj in contents:
        s3_client.delete_object(Bucket=temp_bucket, Key=obj["Key"])


def main() -> None:
    """
    Glue job entrypoint:
      - Read quarterly JSONs from s3://polygon-stock-data-cloud/full/
      - Enrich + nest into json struct
      - Flatten into individual columns (no company column)
      - Write combined_data.parquet under s3://polygon-stock-data-cloud/output/
      - Remove the leftover temp files under output/_temp_write/
    """
    glue_context, job = initialize_glue_job()

    input_paths = [
        "s3://polygon-stock-data-cloud/full/AAPL_Q1_2024.json",
        "s3://polygon-stock-data-cloud/full/AAPL_Q2_2024.json",
        "s3://polygon-stock-data-cloud/full/AAPL_Q3_2024.json",
        "s3://polygon-stock-data-cloud/full/AAPL_Q4_2024.json",
    ]
    enriched = load_and_enrich_json(glue_context, input_paths)
    flat = flatten_json_struct(enriched)

    temp_output = "s3://polygon-stock-data-cloud/output/_temp_write"
    final_output = "s3://polygon-stock-data-cloud/output/combined_data.parquet"

    write_coalesced_and_copy_and_cleanup(flat, temp_output, final_output)

    job.commit()


if __name__ == "__main__":
    main()