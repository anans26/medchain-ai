# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional stage with full dependencies
FROM base AS full
COPY requirements.txt ./
RUN pip install --no-cache-dir torch transformers

# Final image – choose lightweight by default
FROM base
COPY . /app
CMD ["python", "test_ai_inference.py"]
