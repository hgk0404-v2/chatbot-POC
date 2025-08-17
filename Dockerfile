FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# 미리 빌드된 CUDA wheel 설치 (torch 2.3 + CUDA 12.1 대응)
RUN pip install llama-cpp-python==0.2.90 \
    --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/torch2.3/cu121

# requirements.txt 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]