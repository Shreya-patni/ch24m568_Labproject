# Step 1: Base image (Python + dependencies)
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy project files into container
COPY . /app

# Step 4: Install system dependencies (for Spark/Java, if needed)
RUN apt-get update && apt-get install -y openjdk-17-jdk curl && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirement.txt
#RUN dvc pull

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r training/requirements.txt \
    && pip install mlflow fastapi uvicorn pyspark

# Step 6: Expose ports (FastAPI + MLflow)
EXPOSE 8000
EXPOSE 5000

# Step 7: Environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Step 8: Default command → run MLflow & FastAPI together
CMD python -m mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./artifacts \
      --host 0.0.0.0 \
      --port 5000 & \
    uvicorn deployment.app:app --host 0.0.0.0 --port 8000
