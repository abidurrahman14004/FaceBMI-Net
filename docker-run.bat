@echo off
REM Helper script to run Docker container on Windows

echo Starting BMI Predictor Docker Container...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Build the image
echo Building Docker image...
docker build -t bmi-predictor:latest .

REM Run the container
echo Starting container...
docker run -d ^
    --name bmi-predictor-app ^
    -p 8501:8501 ^
    -v "%CD%\models:/app/models:ro" ^
    -v "%CD%\samples:/app/samples:ro" ^
    --restart unless-stopped ^
    bmi-predictor:latest

if errorlevel 1 (
    echo Error: Failed to start container
    exit /b 1
)

echo Container started successfully!
echo Access the app at: http://localhost:8501
echo.
echo Useful commands:
echo   View logs:    docker logs -f bmi-predictor-app
echo   Stop:         docker stop bmi-predictor-app
echo   Remove:       docker rm bmi-predictor-app

pause
