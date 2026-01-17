# 🐳 How to Start Docker Desktop

## The Error You're Seeing

```
error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/...": 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**This means Docker Desktop is not running!**

## Quick Fix

### Step 1: Start Docker Desktop

1. **Open Docker Desktop:**
   - Press `Windows Key` and type "Docker Desktop"
   - Click on "Docker Desktop" to launch it
   - OR look for Docker Desktop icon in your Start Menu

2. **Wait for Docker to Start:**
   - You'll see a Docker whale icon in your system tray (bottom right)
   - Wait until it shows "Docker Desktop is running"
   - This may take 30-60 seconds

3. **Verify Docker is Running:**
   ```powershell
   docker info
   ```
   If this command works (shows Docker info), you're good to go!

### Step 2: Run Your App

Once Docker Desktop is running:

```powershell
cd "E:\BMI project"
docker-compose up --build
```

## If Docker Desktop Won't Start

### Check Prerequisites:

1. **WSL 2 is enabled:**
   ```powershell
   wsl --status
   ```
   If not installed, install WSL 2:
   ```powershell
   wsl --install
   ```

2. **Virtualization is enabled:**
   - Check BIOS settings
   - Enable "Virtualization Technology" or "VT-x"

3. **Windows Updates:**
   - Make sure Windows is up to date
   - Restart your computer

### Reinstall Docker Desktop (if needed):

1. Uninstall Docker Desktop
2. Download latest from: https://www.docker.com/products/docker-desktop
3. Install and restart computer

## Quick Verification

Run these commands to verify everything is working:

```powershell
# 1. Check Docker is running
docker info

# 2. Check Docker Compose works
docker-compose version

# 3. Check you're in the right directory
pwd
# Should show: E:\BMI project

# 4. Check model file exists
ls models\hybrid_model_v2.pth

# 5. Build and run
docker-compose up --build
```

## Expected Output

When Docker Desktop is running and you execute `docker-compose up --build`, you should see:

```
[+] Building ...
[+] Running ...
bmi-predictor-app  | 
bmi-predictor-app  |   You can now view your Streamlit app in your browser.
bmi-predictor-app  | 
bmi-predictor-app  |   Local URL: http://localhost:8501
```

Then open **http://localhost:8501** in your browser!

---

**TL;DR: Start Docker Desktop first, then run `docker-compose up --build`** 🚀
