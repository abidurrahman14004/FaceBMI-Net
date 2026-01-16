# Streamlit Cloud Deployment Setup

## Requirements.txt Configuration

The `requirements.txt` file is configured for Streamlit Cloud deployment with Python 3.13 compatibility.

### Key Dependencies:
- `mediapipe>=0.10.30,<0.11.0` - Required for Python 3.13 on Streamlit Cloud
- `numpy>=1.26.0,<2.2.0` - Compatible with TensorFlow 2.19.0
- All other dependencies use flexible version constraints

## Deployment Checklist

✅ **requirements.txt** includes `mediapipe>=0.10.30,<0.11.0`
✅ **No runtime.txt** - Streamlit Cloud manages Python version automatically
✅ **No packages.txt** - Not needed for Streamlit Cloud
✅ **Model file** should be in `models/hybrid_model_v2.pth`

## Troubleshooting

If MediaPipe is not detected after deployment:

1. Check deployment logs for MediaPipe installation
2. Verify `requirements.txt` is in the root directory
3. Ensure the file includes: `mediapipe>=0.10.30,<0.11.0`
4. Check that Python 3.13 is being used (MediaPipe 0.10.30+ required)

## Local vs Cloud

- **Local (Python 3.12)**: Can use `mediapipe==0.10.13` or `>=0.10.30`
- **Streamlit Cloud (Python 3.13)**: Requires `mediapipe>=0.10.30`

The current `requirements.txt` uses `>=0.10.30` which works for both.
