import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd
import io
import sys

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bmi_predictor import BMIPredictor

# Page configuration
st.set_page_config(
    page_title="BMI Predictor from Face Image",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 5px;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .bmi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .category {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .underweight { color: #3498db; }
    .normal { color: #2ecc71; }
    .overweight { color: #f39c12; }
    .obese { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_predictor():
    """Load the BMI predictor model (cached)"""
    try:
        model_path = os.path.join('models', 'hybrid_model_v2.pth')
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model file not found at {model_path}")
            return None, f"Model file not found at {model_path}"
        
        predictor = BMIPredictor(model_path=model_path)
        
        if predictor.model_loaded:
            return predictor, None
        else:
            return None, predictor.load_error
            
    except Exception as e:
        return None, str(e)

def get_bmi_category_class(category):
    """Get CSS class for BMI category"""
    category_lower = category.lower()
    if 'underweight' in category_lower:
        return 'underweight'
    elif 'normal' in category_lower:
        return 'normal'
    elif 'overweight' in category_lower:
        return 'overweight'
    elif 'obese' in category_lower:
        return 'obese'
    return ''

def predict_bmi(image_bytes, predictor):
    """Predict BMI from image bytes"""
    try:
        result = predictor.predict(image_bytes)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }

def main():
    # Sidebar
    with st.sidebar:
        st.title("🏥 BMI Predictor")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Home", "Sample Images", "Privacy Policy"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        ### About
        This app predicts BMI (Body Mass Index) from facial images using deep learning.
        
        ### How to use:
        1. Upload a clear frontal face image
        2. Click 'Predict BMI'
        3. View your results
        
        ### Supported formats:
        - PNG, JPG, JPEG, GIF, WEBP
        - Max file size: 16MB
        """)
    
    # Main content based on page selection
    if page == "Home":
        show_home_page()
    elif page == "Sample Images":
        show_samples_page()
    else:
        show_privacy_policy()

def show_home_page():
    """Main prediction page"""
    st.title("BMI Prediction from Face Image")
    st.markdown("Upload a clear frontal face image to predict BMI")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            predictor, error = load_predictor()
            
            if predictor is not None:
                st.session_state.predictor = predictor
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error(f"❌ Failed to load model: {error}")
                st.info("Please ensure the model file exists at `models/hybrid_model_v2.pth`")
                return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        help="Upload a clear frontal face image (PNG, JPG, JPEG, GIF, or WEBP)"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Predict button
            if st.button("🔮 Predict BMI", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Get image bytes
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    # Predict
                    result = predict_bmi(image_bytes, st.session_state.predictor)
                    
                    if result['success']:
                        # Display results
                        bmi = result['bmi']
                        category = result['category']
                        category_class = get_bmi_category_class(category)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="text-align: center;">
                                <div class="bmi-value">{bmi:.2f}</div>
                                <div class="category {category_class}">{category}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # BMI Categories reference
                        st.markdown("---")
                        st.markdown("### BMI Categories Reference")
                        st.markdown("""
                        - **Underweight**: BMI < 18.5
                        - **Normal weight**: 18.5 ≤ BMI < 25
                        - **Overweight**: 25 ≤ BMI < 30
                        - **Obese**: BMI ≥ 30
                        """)
                        
                        if result.get('message'):
                            st.info(result['message'])
                    else:
                        st.error(f"❌ {result.get('error', 'Prediction failed')}")
    else:
        st.info("👆 Please upload an image to get started")

def show_samples_page():
    """Display sample images with true BMI values"""
    st.title("Sample Images")
    st.markdown("Browse sample images with their true BMI values")
    
    # Load samples data
    try:
        csv_path = os.path.join('samples', 'dataset.csv')
        
        if not os.path.exists(csv_path):
            st.error(f"❌ Samples CSV not found at {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if 'image_filename' not in df.columns or 'BMI' not in df.columns:
            st.error("❌ Required columns not found in CSV (image_filename, BMI)")
            return
        
        # Clean data
        columns_to_use = ['image_filename', 'BMI']
        if 'id' in df.columns:
            columns_to_use.insert(0, 'id')
        
        df_clean = df[columns_to_use].copy()
        df_clean = df_clean.dropna(subset=['image_filename', 'BMI'])
        df_clean['BMI'] = pd.to_numeric(df_clean['BMI'], errors='coerce')
        df_clean = df_clean.dropna(subset=['BMI'])
        
        # Check images directory
        images_dir = os.path.join('samples', 'front')
        
        if not os.path.exists(images_dir):
            st.error(f"❌ Images directory not found at {images_dir}")
            return
        
        # Get available images
        available_files = set(os.listdir(images_dir))
        available_files_lower = {f.lower(): f for f in available_files}
        
        # Filter samples with existing images
        samples_with_images = []
        for _, row in df_clean.iterrows():
            image_filename = row['image_filename']
            
            # Try exact match
            if image_filename in available_files:
                samples_with_images.append(row)
                continue
            
            # Try case-insensitive match
            if image_filename.lower() in available_files_lower:
                samples_with_images.append(row)
                continue
        
        if len(samples_with_images) == 0:
            st.warning("⚠️ No sample images found")
            return
        
        st.success(f"✅ Found {len(samples_with_images)} sample images")
        
        # Display samples in grid
        st.markdown("---")
        
        # Number of columns
        cols_per_row = 3
        
        for i in range(0, len(samples_with_images), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(samples_with_images):
                    sample = samples_with_images[idx]
                    
                    with cols[j]:
                        # Try to load and display image
                        image_filename = sample['image_filename']
                        image_path = os.path.join(images_dir, image_filename)
                        
                        # Try case-insensitive match if exact doesn't exist
                        if not os.path.exists(image_path):
                            image_filename_lower = image_filename.lower()
                            if image_filename_lower in available_files_lower:
                                image_filename = available_files_lower[image_filename_lower]
                                image_path = os.path.join(images_dir, image_filename)
                        
                        try:
                            if os.path.exists(image_path):
                                img = Image.open(image_path)
                                st.image(img, use_container_width=True)
                                
                                bmi = sample['BMI']
                                
                                # Determine category
                                if bmi < 18.5:
                                    category = "Underweight"
                                    category_class = "underweight"
                                elif bmi < 25:
                                    category = "Normal"
                                    category_class = "normal"
                                elif bmi < 30:
                                    category = "Overweight"
                                    category_class = "overweight"
                                else:
                                    category = "Obese"
                                    category_class = "obese"
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 0.5rem;">
                                    <div style="font-size: 1.5rem; font-weight: bold;">{bmi:.2f}</div>
                                    <div class="{category_class}" style="font-size: 1rem;">{category}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
        
    except Exception as e:
        st.error(f"❌ Error loading samples: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_privacy_policy():
    """Display privacy policy"""
    st.title("Privacy Policy & Terms")
    
    st.markdown("""
    ## Privacy Policy
    
    ### Data Collection
    This application processes images you upload to predict BMI. We want to be transparent about how we handle your data:
    
    - **Image Processing**: Images are processed in real-time and are not stored on our servers
    - **No Personal Information**: We do not collect, store, or share any personal information
    - **Temporary Processing**: Uploaded images are only held in memory during prediction and are immediately discarded
    
    ### How It Works
    1. You upload a facial image
    2. Our AI model analyzes the image
    3. The predicted BMI is returned to you
    4. The image is discarded from memory
    
    ### Disclaimer
    - This tool is for **educational and research purposes only**
    - BMI predictions from facial images are experimental and should not replace professional medical advice
    - Always consult with healthcare professionals for accurate health assessments
    - Results may vary and should not be used for medical diagnosis
    
    ### Data Security
    - All processing happens server-side with no data persistence
    - We do not use cookies or tracking technologies
    - Your privacy is our priority
    
    ### Your Rights
    - You control what images you upload
    - You can stop using the service at any time
    - No account or registration required
    
    ## Terms of Use
    
    By using this application, you agree to:
    - Use the service responsibly and legally
    - Not attempt to reverse engineer or manipulate the model
    - Understand that results are predictions, not medical diagnoses
    - Not hold the developers liable for any decisions made based on predictions
    
    ### Limitations
    - The model is trained on specific datasets and may not generalize to all populations
    - Accuracy may vary based on image quality, lighting, angle, and other factors
    - This is a research tool and should not replace professional medical assessment
    
    ## Contact
    If you have questions or concerns about this privacy policy, please contact the development team.
    
    ---
    
    **Last Updated**: January 2026
    """)

if __name__ == "__main__":
    main()
