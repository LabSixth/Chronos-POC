# Stock Price Prediction Pipeline

This project implements a machine learning pipeline for stock price prediction using time series forecasting models. The pipeline includes data processing, model training, and evaluation components.

## Project Structure

```
model_training/
├── artifacts/               # Output artifacts (models, metrics, visualizations)
├── config/                  # Configuration files
│   ├── default-config.yaml  # Main configuration
│   └── logging/            # Logging configuration
│       └── local.conf      # Logging settings
├── data/                   # Data directory
│   └── combined_data.parquet  # Combined Stock price data (only for local dev purpose, should be removed)
├── dockerfiles/            # Docker configuration files
│   ├── Dockerfile.dev      # Development environment
│   └── Dockerfile.deploy   # Production deployment
├── src/                    # Source code
│   ├── data_processing/    # Data processing modules
│   │   ├── data_loading.py
│   │   ├── data_cleaning.py
│   │   └── eda.py
│   ├── models/            # Model implementations
│   │   ├── prophet_model.py
│   │   └── chronos_model.py
│   ├── utils/             # Utility functions
│   │   ├── model_evaluation.py
│   │   └── aws_utils.py
│   └── api/               # API service
│       └── main.py
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── pipeline.py           # Main pipeline script
```

## Required Environment Variables

The pipeline supports both local and AWS S3 data storage. For AWS functionality, the following environment variables should be set:

- `AWS_ACCESS_KEY_ID` (*required* for S3 data loading and artifact upload)
- `AWS_SECRET_ACCESS_KEY` (*required* for S3 data loading and artifact upload)
- `AWS_DEFAULT_REGION` (*optional*, defaults to `us-east-1` in config)
- `AWS_BUCKET` (*optional*, set in `default-config.yaml` under `aws.bucket_name`)
- `AWS_PREFIX` (*optional*, set in `default-config.yaml` under `aws.prefix`)

Users can set these variables in two ways:
1. Create a `.env` file in the project root
2. Pass them directly when running the Docker container

## Workflow & Usage

### 1. Dev Stage - Local Shell Command

For local development, debugging, and testing.

**Requirements:** Python 3.10, dependencies installed, data file ready.

```bash
cd model_training
# Install core dependencies
pip install -r requirements.txt
# Install Amazon Chronos (for the chronos_model)
pip install git+https://github.com/amazon-science/chronos-forecasting.git
python pipeline.py --config config/default-config.yaml
```

---

### 2. Dev Stage - Local Docker

For local environment isolation and production simulation.

**Requirements:** Docker Desktop installed, data file ready.

```bash
cd model_training
# Build the development image
docker build -f dockerfiles/Dockerfile.dev -t stock-training .
# Run the container (Windows)
docker run -v %cd%/artifacts:/app/artifacts -v %cd%/data:/app/data stock-training
# Run the container (Mac/Linux)
docker run -v $(pwd)/artifacts:/app/artifacts -v $(pwd)/data:/app/data stock-training
```

---

### 3. Deploy Stage - Remote Docker

# TODO: Add remote/production deployment instructions

---