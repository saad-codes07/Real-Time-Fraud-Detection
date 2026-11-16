FROM python:3.14-slim

# XGBoost needs the OpenMP library, which the slim image doesn't include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install libraries first. Docker caches this step, so rebuilding after a
# code change is fast because the requirements file didn't change.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

# The model files are attached at run time (see docker-compose.yml).
ENV MODEL_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]