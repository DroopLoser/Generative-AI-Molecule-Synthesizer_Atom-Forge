FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    default-jre \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install torch CPU version first (smaller, no CUDA needed)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric

# Install everything else
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]