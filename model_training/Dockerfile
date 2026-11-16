FROM python:3.14-slim

# XGBoost needs the OpenMP library, which the slim image doesn't include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY model_training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_training/ ./model_training/

# Anything you add after "docker compose run trainer" is appended here,
# so you can pass options like --no-use-smote.
ENTRYPOINT ["python", "model_training/train.py", \
            "--data-path", "/data/creditcard.csv", \
            "--output-dir", "/models", \
            "--tracking-uri", "http://mlflow:5000"]