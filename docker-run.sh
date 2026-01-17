#!/bin/bash
# Helper script to run Docker container

set -e

echo "🚀 Starting BMI Predictor Docker Container..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the image
echo "📦 Building Docker image..."
docker build -t bmi-predictor:latest .

# Run the container
echo "🏃 Starting container..."
docker run -d \
    --name bmi-predictor-app \
    -p 8501:8501 \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/samples:/app/samples:ro" \
    --restart unless-stopped \
    bmi-predictor:latest

echo "✅ Container started successfully!"
echo "🌐 Access the app at: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  View logs:    docker logs -f bmi-predictor-app"
echo "  Stop:         docker stop bmi-predictor-app"
echo "  Remove:       docker rm bmi-predictor-app"
