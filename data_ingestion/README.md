
# Stock Data Microservice

A microservice designed to extract stock data from the Polygon API and save it to AWS S3. This service is containerized with Docker,
pushed to AWS ECR, and deployed as an AWS Lambda function.

This microservice extracts stock market data from Polygon.io API and stores it in AWS S3 for further analysis or processing.
The solution is fully cloud-native, leveraging AWS services for deployment and operations.

---

## Features

- Fetches stock data from Polygon.io API
- Processes and transforms the data as needed
- Stores the processed data in AWS S3 buckets
- Runs as a serverless function on AWS Lambda
- Containerized for consistent deployment and execution

## Architecture

1. **Data Source**: Polygon.io API for stock market data
2. **Processing**: AWS Lambda function (containerized)
3. **Storage**: AWS S3 buckets
4. **Container Registry**: AWS ECR (Elastic Container Registry)

---

## Project Structure

```markdown
Stock-Data-Microservice/
├── src/
│   ├── configs/
│   │   └── config.yaml        # Configuration settings
│   ├── polygon_data/
│   │   ├── __init__.py
│   │   ├── aggregation.py     # Data aggregation functionality
│   │   ├── helpers.py         # Helper functions
│   │   └── lambda_function.py # AWS Lambda entry point
│   └── __init__.py
├── Dockerfile                 # Container definition
├── .dockerignore              # Files to exclude from container
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Python dependencies
├── .python-version            # Python version specification
└── README.md                  # Project documentation
```

---

## Prerequisites
- AWS Account with appropriate permissions
- Polygon.io API key
- Docker installed locally for development and testing
- Python 3.11 or higher

> [!Note]
> Polygon.io API should be saved in AWS System Manager as `/polygon/apiKey`

---

## Docker Build and Push

1. Build the Docker image:

```shell
docker build -t stock-data-service .
```

2. Tag the image for ECR:

``` bash
   docker tag stock-data-service:latest [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/stock-data-service:latest
```

3. Push to ECR:
``` bash
   aws ecr get-login-password --region [REGION] | docker login --username AWS --password-stdin [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com
   docker push [AWS_ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com/stock-data-service:latest
```

## AWS Lambda Deployment

1. Create a new Lambda function using the container image
2. Configure environment variables (if needed)
3. Set up appropriate IAM roles for S3 access
4. Configure Lambda triggers (CloudWatch Events, etc.)

## Configuration
Update the file with your specific settings: `src/configs/config.yaml`
- Polygon API key
- S3 bucket names
- Data parameters (ticker symbols, date ranges, etc.)
