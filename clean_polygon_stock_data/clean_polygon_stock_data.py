import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize Spark and Glue contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Load raw data from AWS Glue Data Catalog (points to S3 JSON)
raw_df = glueContext.create_dynamic_frame.from_catalog(
    database="polygon_data_catalog",                    # your Glue database name
    table_name="polygon_polygon_data_stock",            # your Glue table name
    transformation_ctx="raw_df"
)

# TODO: Add cleaning logic here if needed
# For now, we use the raw data directly as "cleaned"
cleaned_df = raw_df

# Write cleaned data to S3 in Parquet format (Lakehouse-friendly)
glueContext.write_dynamic_frame.from_options(
    frame=cleaned_df,
    connection_type="s3",
    connection_options = {"path": "s3://polygon-cleaned-stock/output/"},
    format="parquet"
)

# Commit the job (required for Glue jobs)
job.commit()
