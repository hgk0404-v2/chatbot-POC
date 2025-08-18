# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev python3-pip build-essential cmake git pkg-config libgomp1 libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# heavy deps (cuBLAS wheel을 내려받기 위해 extra-index-url 사용)
COPY requirements-heavy.txt /app/requirements-heavy.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefer-binary \
        --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/ \
        -r requirements-heavy.txt

# 나머지 deps
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
