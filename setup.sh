#!/bin/bash
# Optimized setup script for faster deployment
set -e

echo "🚀 Starting optimized setup..."

# Upgrade pip with cache
pip install --upgrade --cache-dir /tmp/pip-cache pip setuptools wheel

# Install CPU-only PyTorch for smaller size (optional but recommended)
# Uncomment the next line to use CPU-only PyTorch (saves ~600MB and 5-7 minutes)
# pip install --cache-dir /tmp/pip-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📦 Installing dependencies..."
pip install --cache-dir /tmp/pip-cache -r requirements.txt

# Download model if MODEL_URL is set (optional)
if [ ! -z "$MODEL_URL" ] && [ ! -f "models/hybrid_model_v2.pth" ]; then
    echo "📥 Downloading model from $MODEL_URL..."
    python download_model.py
fi

echo "✅ Setup completed successfully!"
