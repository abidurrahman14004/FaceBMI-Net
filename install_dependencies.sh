#!/bin/bash
echo "Installing BMI Predictor Dependencies..."
echo ""

echo "Installing MediaPipe..."
pip install mediapipe

echo ""
echo "Installing OpenCV..."
pip install opencv-python

echo ""
echo "Installing SciPy..."
pip install scipy

echo ""
echo "Installing all requirements..."
pip install -r requirements.txt

echo ""
echo "Done! All dependencies installed."
echo "Please restart your Streamlit app."
