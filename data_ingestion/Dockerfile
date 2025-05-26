FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source as src/ so src.polygon_data works
COPY src/ ./src/

# Make sure Python can import `src.*`
ENV PYTHONPATH="/var/task"

# Set the Lambda handler path (must match imports)
CMD ["src.polygon_data.lambda_function.handler"]