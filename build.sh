#!/bin/bash
# Optimized build script for faster deployment

set -e

echo "🚀 Starting optimized build..."

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install dependencies with cache
echo "📦 Installing dependencies..."
pip install --cache-dir /tmp/pip-cache -r requirements.txt

# Verify critical packages
python -c "import torch; import torchvision; print(f'PyTorch {torch.__version__} installed successfully')"

echo "✅ Build completed successfully!"
