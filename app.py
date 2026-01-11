from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import pandas as pd
from models.bmi_predictor import BMIPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize BMI predictor with model path (lazy loading)
model_path = os.path.join('models', 'hybrid_model_v2.pth')
bmi_predictor = None

def get_bmi_predictor():
    """Lazy initialization of BMI predictor"""
    global bmi_predictor
    if bmi_predictor is None:
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                print("Creating BMIPredictor instance - model will be loaded on first prediction attempt")
            
            # Create predictor instance (it won't fail even if model doesn't exist)
            bmi_predictor = BMIPredictor(model_path=model_path)
            
            # If model was successfully loaded, print confirmation
            if bmi_predictor.model_loaded:
                print("BMI Predictor initialized successfully!")
            else:
                print(f"BMI Predictor created but model not loaded yet. Error: {bmi_predictor.load_error}")
                
        except Exception as e:
            print(f"Error initializing BMI predictor: {e}")
            import traceback
            traceback.print_exc()
            # Create a dummy predictor that returns errors
            class DummyPredictor:
                def predict(self, image_bytes):
                    return {
                        'success': False,
                        'error': f'Model not available: {str(e)}. Please ensure the model file exists at {model_path}'
                    }
            bmi_predictor = DummyPredictor()
    return bmi_predictor

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WEBP'}), 400
        
        # Read image file
        image_bytes = file.read()
        
        # Process image and predict BMI
        predictor = get_bmi_predictor()
        result = predictor.predict(image_bytes)
        
        if result['success']:
            return jsonify({
                'success': True,
                'bmi': result['bmi'],
                'category': result['category'],
                'message': result.get('message', '')
            })
        else:
            return jsonify({'error': result.get('error', 'Prediction failed')}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/samples')
@app.route('/samples/')
def samples():
    """Display sample images with true BMI values"""
    return render_template('samples.html')

@app.route('/privacy-policy')
@app.route('/privacy-policy/')
def privacy_policy():
    """Display privacy policy and terms"""
    return render_template('privacy_policy.html')

@app.route('/api/samples')
def get_samples():
    """API endpoint to get samples data"""
    try:
        csv_path = os.path.join('samples', 'dataset.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': f'Samples CSV not found at {csv_path}'}), 404
        
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if 'image_filename' not in df.columns:
            return jsonify({'error': 'Column "image_filename" not found in CSV'}), 400
        if 'BMI' not in df.columns:
            return jsonify({'error': 'Column "BMI" not found in CSV'}), 400
        
        # Select columns to return (include id if available)
        columns_to_return = ['image_filename', 'BMI']
        if 'id' in df.columns:
            columns_to_return.insert(0, 'id')
        
        # Clean data - remove rows with missing values in required columns
        df_clean = df[columns_to_return].copy()
        df_clean = df_clean.dropna(subset=['image_filename', 'BMI'])
        
        # Convert BMI to float and remove invalid values
        df_clean['BMI'] = pd.to_numeric(df_clean['BMI'], errors='coerce')
        df_clean = df_clean.dropna(subset=['BMI'])
        
        # Check if images directory exists
        images_dir = os.path.join('samples', 'front')
        images_exist = os.path.exists(images_dir)
        
        # Filter samples to only include those with existing image files
        samples_with_images = []
        if images_exist:
            try:
                # Get list of all files in the images directory (normalized to lowercase for matching)
                available_files_lower = {}
                for file in os.listdir(images_dir):
                    file_lower = file.lower()
                    if file_lower not in available_files_lower:
                        available_files_lower[file_lower] = file  # Store original filename
                
                # Check each sample for matching image file
                for _, row in df_clean.iterrows():
                    image_filename = row['image_filename']
                    found = False
                    
                    # Try exact match first
                    image_path = os.path.join(images_dir, image_filename)
                    if os.path.exists(image_path):
                        samples_with_images.append(row.to_dict())
                        found = True
                        continue
                    
                    # Try case-insensitive match
                    filename_lower = image_filename.lower()
                    if filename_lower in available_files_lower:
                        samples_with_images.append(row.to_dict())
                        found = True
                        continue
                    
                    # Try with different extensions if filename ends with .jpg or .jpeg
                    if filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg'):
                        base_name = os.path.splitext(image_filename)[0]
                        base_name_lower = base_name.lower()
                        
                        # Check for various extensions
                        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.Jpg', '.Jpeg']:
                            test_filename = base_name + ext
                            test_filename_lower = test_filename.lower()
                            if test_filename_lower in available_files_lower:
                                samples_with_images.append(row.to_dict())
                                found = True
                                break
            except Exception as e:
                print(f"Error reading images directory: {e}")
                samples_with_images = []
        else:
            # If images directory doesn't exist, return empty list
            samples_with_images = []
        
        return jsonify({
            'samples': samples_with_images,
            'total': len(samples_with_images),
            'images_available': len(samples_with_images) > 0,
            'images_dir': images_dir if images_exist else None,
            'message': f'Successfully loaded {len(samples_with_images)} samples with available images'
        })
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty'}), 400
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_samples: {error_details}")
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 500

@app.route('/samples/front/<filename>')
def sample_image(filename):
    """Serve sample images"""
    try:
        images_dir = os.path.join('samples', 'front')
        if not os.path.exists(images_dir):
            # Return a 404 with a helpful message
            return jsonify({
                'error': 'Images directory not found',
                'message': f'Image directory "{images_dir}" does not exist. Please ensure sample images are placed in this directory.',
                'expected_path': os.path.abspath(images_dir)
            }), 404
        
        # Check if file exists
        file_path = os.path.join(images_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': f'Image file "{filename}" not found'}), 404
            
        return send_from_directory(images_dir, filename)
    except Exception as e:
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.errorhandler(404)
def not_found(error):
    # Only return index.html for actual 404s, not for valid routes
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
