FROM python:3.14-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY frontend/ ./frontend/

# Inside Compose, the API is reachable by its service name, "api".
ENV API_URL=http://api:8000
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]