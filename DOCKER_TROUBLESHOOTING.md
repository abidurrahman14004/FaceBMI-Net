# Docker Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Docker Desktop Not Running

**Error:**
```
unable to get image 'bmiproject-bmi-predictor': error during connect: 
Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/images/...": 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**Solution:**
1. **Start Docker Desktop:**
   - Open Docker Desktop application
   - Wait for it to fully start (whale icon in system tray should be steady)
   - Check if Docker is running: `docker info`

2. **Verify Docker is running:**
   ```powershell
   docker info
   ```
   If this works, Docker is running. If not, start Docker Desktop.

3. **Restart Docker Desktop (if needed):**
   - Right-click Docker Desktop icon in system tray
   - Select "Restart"
   - Wait for it to start completely

### Issue 2: Version Warning in docker-compose.yml

**Warning:**
```
the attribute `version` is obsolete, it will be ignored
```

**Solution:**
- Already fixed! The `version` field has been removed from `docker-compose.yml`
- This is just a warning and won't prevent Docker from working

### Issue 3: Port Already in Use

**Error:**
```
Error: bind: address already in use
```

**Solution:**
1. **Find what's using the port:**
   ```powershell
   netstat -ano | findstr :8501
   ```

2. **Kill the process (replace PID with actual process ID):**
   ```powershell
   taskkill /PID <PID> /F
   ```

3. **Or change the port in docker-compose.yml:**
   ```yaml
   ports:
     - "8502:8501"  # Use port 8502 instead
   ```

### Issue 4: Model File Not Found

**Error:**
```
Model file not found at models/hybrid_model_v2.pth
```

**Solution:**
1. **Verify model file exists:**
   ```powershell
   ls models/hybrid_model_v2.pth
   ```

2. **Check file permissions:**
   - Ensure the file is readable
   - Check if file path is correct

3. **Verify volume mount:**
   - Check `docker-compose.yml` has volume mount for models
   - Ensure path is correct (relative to docker-compose.yml location)

### Issue 5: Build Fails

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c pip install..." did not complete successfully
```

**Solution:**
1. **Check internet connection:**
   - Docker needs internet to download packages

2. **Clear Docker cache:**
   ```powershell
   docker system prune -a
   ```

3. **Rebuild without cache:**
   ```powershell
   docker-compose build --no-cache
   ```

### Issue 6: MediaPipe Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.13
```

**Solution:**
1. **Check Python version in Dockerfile:**
   - Should be Python 3.12 (already set)

2. **Try updating pip first:**
   - Already handled in Dockerfile

3. **Check if version exists:**
   ```powershell
   docker run --rm python:3.12-slim pip index versions mediapipe
   ```

### Issue 7: Permission Denied

**Error:**
```
Permission denied: /app/models
```

**Solution:**
1. **Check file permissions:**
   ```powershell
   icacls models\hybrid_model_v2.pth
   ```

2. **Run Docker with proper user (if needed):**
   - Usually not needed on Windows
   - Check Docker Desktop settings

### Issue 8: Container Exits Immediately

**Error:**
```
Container exited with code 1
```

**Solution:**
1. **Check logs:**
   ```powershell
   docker-compose logs bmi-predictor
   ```

2. **Run interactively to debug:**
   ```powershell
   docker run -it --rm bmi-predictor /bin/bash
   ```

3. **Check if Streamlit can start:**
   ```powershell
   docker run -it --rm bmi-predictor streamlit --version
   ```

## Quick Diagnostic Commands

```powershell
# Check Docker is running
docker info

# Check Docker Compose version
docker-compose version

# List running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs
docker-compose logs -f

# Check image exists
docker images | grep bmi-predictor

# Remove old containers/images
docker-compose down
docker system prune -a
```

## Step-by-Step Startup

1. **Start Docker Desktop:**
   - Open Docker Desktop application
   - Wait for "Docker Desktop is running" message

2. **Verify Docker is running:**
   ```powershell
   docker info
   ```

3. **Navigate to project directory:**
   ```powershell
   cd "E:\BMI project"
   ```

4. **Build and start:**
   ```powershell
   docker-compose up --build
   ```

5. **Access the app:**
   - Open browser: http://localhost:8501

## Still Having Issues?

1. **Check Docker Desktop logs:**
   - Docker Desktop → Troubleshoot → View logs

2. **Restart Docker Desktop:**
   - Right-click system tray icon → Restart

3. **Update Docker Desktop:**
   - Check for updates in Docker Desktop settings

4. **Verify system requirements:**
   - Windows 10/11 64-bit
   - WSL 2 enabled (for Docker Desktop)
   - Virtualization enabled in BIOS
