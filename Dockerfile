FROM python:3.11-slim

# eccodes C library required by cfgrib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-dev libeccodes-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8051
CMD ["gunicorn", "--worker-class=gthread", "--workers=1", "--threads=4", "--bind=0.0.0.0:8051", "--timeout=120", "app:server"]
