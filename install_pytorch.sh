#!/bin/bash
# Install CPU-only PyTorch for faster deployment (smaller size)
# This reduces PyTorch size from ~800MB to ~200MB

echo "Installing CPU-only PyTorch for faster deployment..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "PyTorch CPU-only installation complete!"
