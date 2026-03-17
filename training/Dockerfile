# AthleteView AI Engine - Training Container
# NVIDIA CUDA 12.2 with Python 3.11
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="AthleteView AI <engineering@athleteview.ai>"
LABEL description="Training container for AthleteView AI models"
LABEL version="1.0.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir -r /app/training/requirements.txt

# Install PyTorch with CUDA 12.2 support (override requirements.txt version)
RUN pip install --no-cache-dir \
    torch==2.2.0+cu122 \
    torchvision==0.17.0+cu122 \
    --index-url https://download.pytorch.org/whl/cu122

# Copy config files
COPY configs/ /app/training/configs/

# Copy training scripts
COPY scripts/ /app/training/scripts/

# Copy dataset utilities and docs
COPY datasets/ /app/training/datasets/

# Copy notebooks
COPY notebooks/ /app/training/notebooks/

# Create directories for model weights, outputs, data, and logs
RUN mkdir -p /app/weights /app/outputs /app/data /app/logs /app/checkpoints

# Set default environment variables
ENV WANDB_DIR=/app/logs
ENV TORCH_HOME=/app/weights
ENV HF_HOME=/app/weights/huggingface
ENV YOLO_CONFIG_DIR=/app/training/configs
ENV ATHLETEVIEW_TRAINING_DIR=/app/training
ENV ATHLETEVIEW_WEIGHTS_DIR=/app/weights
ENV ATHLETEVIEW_DATA_DIR=/app/data
ENV ATHLETEVIEW_OUTPUT_DIR=/app/outputs

# Set working directory to training
WORKDIR /app/training

# Health check - verify CUDA is available
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["scripts/evaluate.py", "--help"]
