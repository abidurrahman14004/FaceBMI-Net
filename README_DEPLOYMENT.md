# Deployment Guide for BMI Predictor Flask App

## Current Status
✅ **App works perfectly on localhost**  
⚠️ **Streamlit Cloud is not ideal for Flask apps** - Consider alternatives below

## Recommended Deployment Platforms

### Option 1: Render.com (Recommended - Free Tier)
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new "Web Service"
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.11
5. Deploy!

### Option 2: Railway.app
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Railway auto-detects Flask apps
4. Add `Procfile` (already included)
5. Deploy!

### Option 3: Heroku
1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`
4. Uses `Procfile` automatically

## If You Must Use Streamlit Cloud

Streamlit Cloud expects Python 3.11 or 3.12. The current configuration includes:
- `runtime.txt` - Specifies Python 3.11.9
- `packages.txt` - Specifies python3.11
- `requirements.txt` - Updated with compatible versions

**Note**: Streamlit Cloud may still try to use Python 3.13. If this happens:
1. Go to Streamlit Cloud settings
2. Manually set Python version to 3.11
3. Or consider using one of the recommended platforms above

## Local Development
```bash
pip install -r requirements.txt
python app.py
```
App runs on http://localhost:5000

## Files Included for Deployment
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku/Railway deployment
- `runtime.txt` - Python version specification
- `packages.txt` - System packages (for Streamlit Cloud)
- `setup.sh` - Setup script
