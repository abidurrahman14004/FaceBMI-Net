# Docker Setup Guide

Complete Docker setup for BMI Predictor application with all dependencies handled automatically.

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: Using Helper Scripts

**On Linux/Mac:**
```bash
chmod +x docker-run.sh
./docker-run.sh
```

**On Windows:**
```cmd
docker-run.bat
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -t bmi-predictor .

# Run the container
docker run -d \
  --name bmi-predictor-app \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/samples:/app/samples:ro \
  bmi-predictor
```

## 📋 What's Included

The Docker setup automatically handles:

✅ **All Python Dependencies** from `requirements.txt`:
- Streamlit (web framework)
- PyTorch & torchvision (ML framework)
- MediaPipe 0.10.13 (facial landmark extraction)
- NumPy, Pandas, SciPy (data processing)
- scikit-learn (ML utilities)
- Pillow (image processing)
- All other required packages

✅ **System Dependencies**:
- OpenGL libraries (for MediaPipe)
- Graphics libraries
- Build tools (during build phase)

✅ **Optimizations**:
- Multi-stage build (smaller final image)
- Layer caching (faster rebuilds)
- Health checks
- Proper volume mounts

## 📁 Directory Structure

```
BMI project/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files to exclude from build
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Main application
├── models/                 # Model files (mounted as volume)
│   └── hybrid_model_v2.pth
└── samples/                # Sample images (mounted as volume)
    └── dataset.csv
```

## 🔧 Configuration

### Port Configuration

Default port is **8501**. To change it, edit `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:8501"  # Change YOUR_PORT to desired port
```

### Model File

The model file `models/hybrid_model_v2.pth` is mounted as a read-only volume. Make sure it exists before running:

```bash
# Check if model file exists
ls models/hybrid_model_v2.pth
```

### Environment Variables

You can add environment variables in `docker-compose.yml`:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - STREAMLIT_SERVER_PORT=8501
  - YOUR_VAR=value
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find what's using port 8501
# Linux/Mac:
lsof -i :8501

# Windows:
netstat -ano | findstr :8501

# Change port in docker-compose.yml
```

### Model File Not Found

```bash
# Check if model file exists
ls -la models/hybrid_model_v2.pth

# Check container logs
docker-compose logs bmi-predictor

# Verify volume mount
docker exec bmi-predictor-app ls -la /app/models
```

### MediaPipe Errors

All dependencies are installed during build. If issues persist:

```bash
# Rebuild without cache
docker-compose build --no-cache

# Check installed packages
docker exec bmi-predictor-app pip list | grep mediapipe
```

### Container Won't Start

```bash
# Check logs
docker-compose logs

# Run interactively to see errors
docker run -it --rm bmi-predictor /bin/bash
```

### Permission Issues (Linux)

```bash
# Fix permissions
sudo chown -R $USER:$USER models/ samples/

# Or run with user ID
docker run -u $(id -u):$(id -g) ...
```

## 📊 Container Management

### View Logs
```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Stop and Remove
```bash
# Stop container
docker-compose stop

# Stop and remove
docker-compose down

# Remove with volumes
docker-compose down -v
```

### Rebuild After Changes
```bash
# Rebuild and restart
docker-compose up --build -d

# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

### Access Container Shell
```bash
# Execute command in container
docker exec -it bmi-predictor-app /bin/bash

# Check Python version
docker exec bmi-predictor-app python --version

# Check installed packages
docker exec bmi-predictor-app pip list
```

## 🚀 Production Deployment

### Optimizations for Production

1. **Use specific image tags:**
   ```dockerfile
   FROM python:3.12-slim@sha256:...
   ```

2. **Add resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

3. **Use reverse proxy (nginx):**
   - Add nginx container in docker-compose.yml
   - Configure SSL/TLS
   - Set up proper logging

4. **Environment-specific configs:**
   - Use `.env` file for secrets
   - Separate docker-compose files for dev/prod

### Security Best Practices

1. **Don't run as root:**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

2. **Scan for vulnerabilities:**
   ```bash
   docker scan bmi-predictor
   ```

3. **Use secrets management:**
   - Docker secrets
   - Environment variables (not in code)
   - External secret managers

## 📝 Docker Commands Reference

```bash
# Build
docker build -t bmi-predictor .
docker-compose build

# Run
docker run -p 8501:8501 bmi-predictor
docker-compose up

# Stop
docker stop bmi-predictor-app
docker-compose down

# Logs
docker logs bmi-predictor-app
docker-compose logs -f

# Remove
docker rm bmi-predictor-app
docker-compose down -v

# Clean up
docker system prune -a
```

## ✅ Verification

After starting the container, verify everything works:

1. **Check container is running:**
   ```bash
   docker ps | grep bmi-predictor
   ```

2. **Check health:**
   ```bash
   docker inspect bmi-predictor-app | grep Health
   ```

3. **Access the app:**
   - Open browser: http://localhost:8501
   - Should see the BMI Predictor interface

4. **Test prediction:**
   - Upload a face image
   - Should get BMI prediction

## 🎯 Next Steps

1. **Place model file:**
   - Ensure `models/hybrid_model_v2.pth` exists

2. **Start container:**
   ```bash
   docker-compose up --build
   ```

3. **Access app:**
   - Open http://localhost:8501

4. **Monitor:**
   - Watch logs: `docker-compose logs -f`
   - Check health: `docker ps`

## 📞 Support

If you encounter issues:
1. Check container logs
2. Verify model file exists
3. Ensure all dependencies in requirements.txt
4. Check Docker version compatibility

---

**All dependencies from `requirements.txt` are automatically installed during Docker build. No manual installation needed!** 🎉
