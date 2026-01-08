# BMI Predictor - Flask Web Application

A complete Flask web application that predicts BMI (Body Mass Index) from uploaded images using machine learning.

## Features

- 🖼️ **Image Upload**: Drag & drop or click to upload images
- 🤖 **ML Model Integration**: Easy integration with your trained ML model
- 📊 **BMI Display**: Beautiful visualization of BMI results with category classification
- 🎨 **Modern UI**: Responsive and user-friendly interface
- ⚡ **Real-time Processing**: Fast prediction with loading indicators

## Project Structure

```
BMI project/
├── app.py                 # Main Flask application
├── models/
│   ├── __init__.py
│   └── bmi_predictor.py  # ML model wrapper class
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
├── uploads/              # Uploaded images (auto-created)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Integration

✅ **Your PyTorch model is already integrated!**

The application is configured to use your `hybrid_model_v2.pth` model located in the `models/` directory. The model loading and prediction code is already set up in `models/bmi_predictor.py`.

**Current Configuration:**
- Model Path: `models/hybrid_model_v2.pth`
- Framework: PyTorch
- Image Preprocessing: 224x224 RGB with ImageNet normalization
- Device: Auto-detects CPU/GPU

**If you need to customize preprocessing:**
Edit the `preprocess_image` method in `models/bmi_predictor.py` to match your model's training preprocessing:
- Image size (currently 224x224)
- Normalization values (currently ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Any other preprocessing steps your model requires

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Upload an image by either:
   - Dragging and dropping an image onto the upload area
   - Clicking the upload area to browse and select a file
3. Click "Predict BMI" to get your result
4. View your BMI value, category, and recommendations

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- WEBP

Maximum file size: 16MB

## API Endpoints

### `GET /`
Main page with the upload interface.

### `POST /predict`
Upload an image and get BMI prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
    "success": true,
    "bmi": 24.5,
    "category": "Normal weight",
    "message": "You have a healthy weight. Keep up the good work!"
}
```

### `GET /health`
Health check endpoint.

## Customization

### Change Port
Edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port number
```

### Modify Upload Settings
Edit `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Change max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}  # Add/remove formats
```

### Customize UI
Edit `static/style.css` and `templates/index.html` to match your branding.

## Troubleshooting

### Model Loading Issues
- Ensure your model file path is correct
- Check that all required ML framework dependencies are installed
- Verify your model file format matches the loading method

### Image Processing Errors
- Check that images are in supported formats
- Verify image preprocessing matches your model's requirements
- Ensure PIL/Pillow is properly installed

### Port Already in Use
Change the port in `app.py` or stop the process using port 5000.

## Important Notes

⚠️ **This is a demonstration application. For accurate BMI assessment, please consult a healthcare professional.**

✅ **Your PyTorch model (`hybrid_model_v2.pth`) is integrated and ready to use!** The application will automatically load your model when it starts.

## License

This project is provided as-is for educational and demonstration purposes.
