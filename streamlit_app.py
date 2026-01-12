"""
BMI Predictor - Streamlit App
A complete Streamlit application for BMI prediction from facial images.
"""
import streamlit as st
import os
import io
from PIL import Image
import pandas as pd
from models.bmi_predictor import BMIPredictor

# Page configuration
st.set_page_config(
    page_title="BMI Predictor - AI-Powered Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bmi-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .bmi-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .bmi-category {
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bmi_predictor' not in st.session_state:
    st.session_state.bmi_predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_bmi_predictor():
    """Load BMI predictor model (cached)"""
    model_path = os.path.join('models', 'hybrid_model_v2.pth')
    
    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"
    
    try:
        predictor = BMIPredictor(model_path=model_path)
        if predictor.model_loaded:
            return predictor, None
        else:
            return None, predictor.load_error or "Model failed to load"
    except Exception as e:
        return None, str(e)

def get_bmi_category(bmi):
    """Get BMI category and color"""
    if bmi < 18.5:
        return "Underweight", "#17a2b8", "⚠️"
    elif bmi < 25:
        return "Normal weight", "#28a745", "✅"
    elif bmi < 30:
        return "Overweight", "#ffc107", "⚠️"
    else:
        return "Obese", "#dc3545", "⚠️"

def get_bmi_message(bmi, category):
    """Get personalized message based on BMI"""
    if category == "Underweight":
        return "Consider consulting a healthcare professional about healthy weight gain strategies."
    elif category == "Normal weight":
        return "You have a healthy weight! Keep up the good work with balanced nutrition and regular exercise."
    elif category == "Overweight":
        return "Consider adopting a balanced diet and regular exercise routine. Consult a healthcare professional for personalized advice."
    else:
        return "Please consult a healthcare professional for guidance on achieving a healthy weight."

# Sidebar
with st.sidebar:
    st.title("🤖 BMI Predictor")
    st.markdown("### AI-Powered Analysis")
    st.markdown("---")
    
    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["🏠 Home", "📊 Samples", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Model Status")
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Model Not Loaded")
        if st.button("🔄 Reload Model"):
            st.session_state.bmi_predictor = None
            st.rerun()

# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<div class="main-header">🤖 BMI Predictor</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Body Mass Index Calculator from Facial Images")
    
    # Load model
    if st.session_state.bmi_predictor is None:
        with st.spinner("Loading BMI prediction model..."):
            predictor, error = load_bmi_predictor()
            if predictor:
                st.session_state.bmi_predictor = predictor
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error(f"Failed to load model: {error}")
                st.session_state.model_loaded = False
    
    # File uploader
    st.markdown("---")
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        help="Upload a facial image to predict BMI"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis")
            
            if st.button("🔮 Predict BMI", type="primary", use_container_width=True):
                if st.session_state.bmi_predictor is None:
                    st.error("Model not loaded. Please reload the model from the sidebar.")
                else:
                    with st.spinner("Analyzing image and predicting BMI..."):
                        # Read image bytes
                        image_bytes = uploaded_file.read()
                        
                        # Predict BMI
                        result = st.session_state.bmi_predictor.predict(image_bytes)
                        
                        if result['success']:
                            bmi = result['bmi']
                            category = result['category']
                            message = result.get('message', '')
                            
                            # Display results
                            category_name, color, icon = get_bmi_category(bmi)
                            
                            st.markdown(f"""
                            <div class="bmi-result">
                                <div class="bmi-value">{bmi:.2f}</div>
                                <div class="bmi-category" style="color: {color};">
                                    {icon} {category_name}
                                </div>
                                <p style="margin-top: 1rem;">{message}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional info
                            with st.expander("📊 BMI Information"):
                                st.markdown(f"""
                                - **BMI Value**: {bmi:.2f}
                                - **Category**: {category_name}
                                - **Status**: {icon}
                                """)
                                
                                st.markdown("""
                                **BMI Categories:**
                                - Underweight: < 18.5
                                - Normal weight: 18.5 - 24.9
                                - Overweight: 25 - 29.9
                                - Obese: ≥ 30
                                """)
                        else:
                            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
    
    # Features section
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Image Processing</h4>
            <p>Uses a custom convolutional neural network to extract visual features from facial images, analyzing facial structure and body proportions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 ML Model Integration</h4>
            <p>Advanced hybrid model combining image features, tabular data, and landmark information for accurate BMI prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 BMI Display</h4>
            <p>Beautiful visualization of BMI results with category classification and personalized health recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Real-time Processing</h4>
            <p>Fast prediction with loading indicators and instant results display.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: This is a demonstration application. For accurate BMI assessment, please consult a healthcare professional.")

elif page == "📊 Samples":
    st.title("📊 Sample Images with True BMI Values")
    
    # Load samples
    csv_path = os.path.join('samples', 'dataset.csv')
    
    if not os.path.exists(csv_path):
        st.error(f"Samples CSV not found at {csv_path}")
    else:
        try:
            df = pd.read_csv(csv_path)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_bmi = st.number_input("Min BMI", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            with col2:
                max_bmi = st.number_input("Max BMI", min_value=0.0, max_value=50.0, value=50.0, step=0.1)
            with col3:
                num_samples = st.number_input("Number of Samples", min_value=1, max_value=100, value=10, step=1)
            
            # Filter data
            if 'BMI' in df.columns and 'image_filename' in df.columns:
                df_clean = df[['image_filename', 'BMI']].dropna()
                df_clean['BMI'] = pd.to_numeric(df_clean['BMI'], errors='coerce')
                df_clean = df_clean.dropna(subset=['BMI'])
                
                # Filter by BMI range
                filtered = df_clean[(df_clean['BMI'] >= min_bmi) & (df_clean['BMI'] <= max_bmi)]
                
                if len(filtered) > 0:
                    # Limit samples
                    if len(filtered) > num_samples:
                        filtered = filtered.sample(n=num_samples)
                    
                    st.info(f"Showing {len(filtered)} sample(s) (Total available: {len(df_clean)})")
                    
                    # Display samples in grid
                    images_dir = os.path.join('samples', 'front')
                    cols_per_row = 4
                    
                    for i in range(0, len(filtered), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, (idx, row) in enumerate(filtered.iloc[i:i+cols_per_row].iterrows()):
                            with cols[j]:
                                bmi_value = row['BMI']
                                filename = row['image_filename']
                                
                                # Try to load image
                                image_path = os.path.join(images_dir, filename)
                                if os.path.exists(image_path):
                                    img = Image.open(image_path)
                                    st.image(img, caption=f"BMI: {bmi_value:.2f}", use_container_width=True)
                                else:
                                    category, color, icon = get_bmi_category(bmi_value)
                                    st.markdown(f"""
                                    <div style="background: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                                        <h3>{icon}</h3>
                                        <h4>BMI: {bmi_value:.2f}</h4>
                                        <p>{category}</p>
                                        <small>{filename}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"No samples found in BMI range {min_bmi} - {max_bmi}")
            else:
                st.error("Required columns (BMI, image_filename) not found in CSV")
                
        except Exception as e:
            st.error(f"Error loading samples: {str(e)}")

elif page == "ℹ️ About":
    st.title("ℹ️ About BMI Predictor")
    
    st.markdown("""
    ## 🤖 BMI Predictor - AI-Powered Analysis
    
    A complete web application that predicts BMI (Body Mass Index) from uploaded facial images using machine learning.
    
    ### Features
    
    - 🖼️ **Image Upload**: Easy drag & drop or click to upload images
    - 🤖 **ML Model Integration**: Advanced hybrid model combining multiple data sources
    - 📊 **BMI Display**: Beautiful visualization of BMI results with category classification
    - 🎨 **Modern UI**: Responsive and user-friendly interface
    - ⚡ **Real-time Processing**: Fast prediction with loading indicators
    
    ### How It Works
    
    1. **Image Processing**: Uses a custom convolutional neural network to extract visual features from facial images
    2. **Feature Extraction**: Analyzes facial structure and body proportions
    3. **BMI Prediction**: Combines image features with advanced ML models
    - **Result Display**: Shows BMI value, category, and personalized recommendations
    
    ### Supported Image Formats
    
    - PNG
    - JPG/JPEG
    - GIF
    - WEBP
    
    Maximum file size: 16MB
    
    ### Important Notes
    
    ⚠️ **This is a demonstration application. For accurate BMI assessment, please consult a healthcare professional.**
    
    ### Technology Stack
    
    - **Framework**: Streamlit
    - **ML Framework**: PyTorch
    - **Image Processing**: PIL/Pillow
    - **Data Processing**: Pandas, NumPy
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>BMI Predictor - AI-Powered Body Mass Index Calculator</p>
    <p>Built with ❤️ using Streamlit and PyTorch</p>
</div>
""", unsafe_allow_html=True)
