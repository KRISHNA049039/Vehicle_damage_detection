FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose API port
EXPOSE 8080

# Default: run API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
