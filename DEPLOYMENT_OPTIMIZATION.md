# Deployment Optimization Guide

## Issues Fixed

### 1. **Large Model File (61.16 MB)**
- **Problem**: Model file was being tracked in git, uploaded on every deployment
- **Solution**: Added `*.pth` to `.gitignore`
- **Action**: Use external storage (S3, Google Drive) or download model on first run

### 2. **Heavy PyTorch Dependencies**
- **Problem**: Full PyTorch + torchvision = ~800MB, takes 5-10 minutes to install
- **Solution**: Use CPU-only PyTorch (~200MB, installs in 1-2 minutes)
- **Action**: Run `install_pytorch.sh` or use CPU-only wheels

### 3. **No Build Optimizations**
- **Problem**: No caching, no parallel installs
- **Solution**: Added `build.sh` with pip cache
- **Action**: Use build script for faster installs

### 4. **Inefficient Gunicorn Config**
- **Problem**: Too many workers, no preload
- **Solution**: Optimized Procfile with preload
- **Action**: Already updated

## Quick Deployment Tips

### For Render/Railway/Heroku:

1. **Remove model from git** (if not already):
   ```bash
   git rm --cached models/hybrid_model_v2.pth
   git commit -m "Remove model file from git"
   ```

2. **Use CPU-only PyTorch** (add to build script):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Download model on first run** (optional):
   - Store model in cloud storage (S3, Google Drive)
   - Download on app startup if not present

### For Streamlit Cloud:

1. Model file should be in repository (they handle it)
2. Use optimized requirements.txt
3. Consider using model caching

## Expected Deployment Times

- **Before**: 10-15 minutes (PyTorch + model file)
- **After**: 3-5 minutes (CPU-only PyTorch, no model in git)

## Model File Handling

Since model is now in `.gitignore`, you have options:

1. **External Storage** (Recommended):
   - Upload to S3/Google Drive
   - Download on first app start
   - Cache locally

2. **Git LFS** (if supported):
   - Use Git Large File Storage
   - Keeps model in repo but optimized

3. **Build-time download**:
   - Add to build script
   - Download from URL during build
