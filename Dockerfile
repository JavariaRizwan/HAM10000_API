# Dockerfile for HAM10000 Skin Cancer API
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and app
COPY model.h5 .
COPY norm_stats.json .
COPY class_names.json .
COPY label_map.json .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
