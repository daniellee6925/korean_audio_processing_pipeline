FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv ffmpeg git \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools first
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set up working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Set environment variables for GPU
ENV PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["/bin/bash"]