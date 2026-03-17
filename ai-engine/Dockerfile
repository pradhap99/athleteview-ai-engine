FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM base AS dev
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

FROM base AS production
COPY . .
EXPOSE 8001
HEALTHCHECK --interval=15s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
