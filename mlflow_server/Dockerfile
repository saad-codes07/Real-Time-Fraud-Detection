FROM python:3.14-slim

# Same MLflow version as your training environment, so client and server agree.
RUN pip install --no-cache-dir mlflow==3.18.0

WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]