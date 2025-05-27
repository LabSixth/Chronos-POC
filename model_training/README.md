
# Chronos & Prophet Training Microservice

This microservice is responsible for training time-series forecasting models using
[Amazon's Chronos](https://github.com/amazon-science/chronos-forecasting) and
[Facebook Prophet](https://facebook.github.io/prophet/). It is packaged as a container, pushed to Amazon Elastic Container Registry (ECR),
and executed using Amazon ECS Fargate.

---

## ⚙️ Prerequisite

1. AWS account and AWS CLI, and locally authenticated AWS for pushing Docker image to AWS ECR
2. Python virtual environment created, synced, and activated

---

## 🧱 Project Structure

```
.
├── config/                        # Configuration files
├── src/                           # Source code
│   ├── api/                       # API endpoints and server code
│   │   └── main                   # Fast API calls
│   ├── aws_utils/                 # AWS integration utilities
│   │   └── aws_data_ingress.py    # Download data from AWS S3
|   |   └── aws_data_outgress.py   # Export artifacts to AWS S3
│   ├── data_processing/           # Data processing modules
│   │   ├── data_cleaning.py       # Clean and prepare time series data
│   │   ├── data_loading.py        # Load data from various sources
│   │   └── eda.py                 # Exploratory data analysis utilities
│   ├── models/                    # Model implementation
│   │   ├── chronos_model.py       # Amazon Chronos model implementation
│   │   └── prophet_model.py       # Facebook Prophet model implementation
│   ├── utils/                     # Utility functions
│   │   └── model_evaluation.py    # Model evaluation functions
├── test/                          # Test suite
│   └── test_data_processing.py    # Tests for data processing module
│   └── test_evaluation.py         # Tests for model evaluation module
│   └── test_models.py             # Tests for modeling module
├── .dockerignore                  # Files to exclude from Docker context
├── .python-version                # Python version specification
├── Dockerfile                     # Container definition
├── README.md                      # Project documentation
├── pipeline.py                    # Main pipeline entry point
├── pyproject.toml                 # Project metadata and dependencies
└── requirements.txt               # Direct dependencies list
```

---

## 🚀 Key Functionality

1. `pipeline.py`

- Acts as the entrypoint for model training.
- Accepts CLI or environment variables pointing to a YAML config file.
- Loads configuration and routes to either Chronos or Prophet trainer.
- Uploads all model artifacts and metrics to a specified S3 bucket.

2. `src/models/chronos_model.py`

- Loads time-series data.
- Configures Chronos model based on YAML input.
- Trains Chronos model and logs metrics (e.g. MAE, RMSE).
- Saves model to disk.

3. `src/models/prophet_trainer.py`

- Similar to Chronos trainer but using Facebook Prophet.
- Handles Prophet model setup, forecasting, and metric evaluation.
- Outputs include forecast visualizations and metrics YAML.

3. `src/utils/model_evaluation.py`

- Computes accuracy metrics like MAE, RMSE, MAPE for model evaluation.

### Running Pipeline Locally

To run the pipeline locally, run the command below in command line.

```shell
python3 pipeline.py --config config/default-config.yaml
```

---

## 🐳 Dockerization

### Dockerfile

- Based on Python 3.11 Slim base image (compatible with ECS Fargate).
- Installs dependencies using `uv` and `pyproject.toml`.

To build and push:

```bash
docker build -t chronos-application .
aws ecr get-login-password | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<region>.amazonaws.com
docker tag chronos-application:latest <your-ecr-uri>
docker push <your-ecr-uri>
```

---

## ☁️ ECS Fargate Deployment

- This image is deployed via ECS Fargate.
- It pulls the image from ECR, loads config from S3, trains model, and exports results.
