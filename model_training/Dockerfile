# Use the AWS Lambda Python 3.11 base image
FROM python:3.11-slim

# Install git and uv
RUN apt update && apt install -y curl git tar
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    find /root -name uv -type f -executable -exec mv {} /usr/local/bin/uv \;

# Set the working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv pip install --no-cache-dir -r pyproject.toml --system
RUN pip cache purge

# Copy source code
COPY . .

# Add src/ to PYTHONPATH so imports work like: from src.module import ...
ENV PYTHONPATH="/app/src"

# Define the Lambda handler (must be `handler(event, context)` in pipeline.py)
CMD ["python", "pipeline.py", "--config", "config/default-config.yaml"]