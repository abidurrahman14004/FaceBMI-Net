import streamlit as st
import os
import sys
import traceback
from PIL import Image
import io
from models.bmi_predictor import BMIPredictor

# Page configuration
st.set_page_config(
    page_title="BMI Predictor - AI-Powered Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .bmi-value {
        font-size: 4rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
    }
    .bmi-category {
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .scale-bar {
        display: flex;
        height: 50px;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }
    .scale-segment {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 0.9rem;
    }
    .underweight { background: #3498db; }
    .normal { background: #2ecc71; }
    .overweight { background: #f39c12; }
    .obese { background: #e74c3c; }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize BMI predictor (cached to avoid reloading on every rerun)
@st.cache_resource
def load_model():
    """Load the BMI predictor model"""
    try:
        model_path = os.path.join('models', 'hybrid_model_v2.pth')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        predictor = BMIPredictor(model_path=model_path)
        if predictor.model is None:
            st.error("Model failed to load. Please check server logs.")
            return None
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Load model
bmi_predictor = load_model()

# Sidebar Navigation
st.sidebar.title("🤖 BMI Predictor")
st.sidebar.markdown("### AI-Powered Analysis")

page = st.sidebar.radio(
    "Navigate",
    ["Introduction", "How It Works", "The App"],
    index=2
)

# Introduction Page
if page == "Introduction":
    st.markdown('<div class="main-header"><h1>Welcome to BMI Predictor</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    Our AI-powered BMI (Body Mass Index) Predictor uses advanced machine learning to estimate your BMI from a single photograph. 
    This innovative tool combines computer vision, deep learning, and multi-modal data analysis to provide accurate predictions.
    """)
    
    st.markdown("### What is BMI?")
    st.markdown("""
    Body Mass Index (BMI) is a measure of body fat based on height and weight. It's calculated by dividing your weight in kilograms 
    by the square of your height in meters. BMI categories include:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Underweight:** BMI less than 18.5
        - **Normal weight:** BMI 18.5 to 24.9
        """)
    
    with col2:
        st.markdown("""
        - **Overweight:** BMI 25 to 29.9
        - **Obese:** BMI 30 or higher
        """)
    
    st.warning("""
    **Important Note:** This tool is for demonstration purposes only. For accurate health assessments, 
    please consult with a healthcare professional. BMI is a screening tool and may not accurately reflect 
    body fat percentage for all individuals.
    """)

# How It Works Page
elif page == "How It Works":
    st.markdown('<div class="main-header"><h1>How the Model Works</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### Hybrid Deep Learning Architecture")
    st.markdown("""
    Our model uses a sophisticated hybrid architecture that combines multiple data modalities:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🖼️ Image Processing</h4>
            <p>Uses ResNet18 convolutional neural network to extract visual features from facial images, 
            analyzing facial structure and body proportions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Tabular Features</h4>
            <p>Processes 36 engineered features including facial measurements, ratios, and demographic 
            information through a multi-layer neural network.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🕸️ Graph Neural Network</h4>
            <p>Analyzes 21 facial landmarks using a multi-scale Graph Convolutional Network (GCN) to 
            understand spatial relationships between facial features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔄 Fusion & Prediction</h4>
            <p>Combines all features using multi-head attention mechanism and shared layers to generate 
            accurate BMI predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Model Training")
    st.info("""
    The model was trained using 5-fold cross-validation on a comprehensive dataset, ensuring robust performance 
    and generalization. It uses multi-task learning to simultaneously predict BMI, age, sex, and BMI category.
    """)

# The App Page
elif page == "The App":
    st.markdown('<div class="main-header"><h1>The App</h1></div>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if bmi_predictor is None:
        st.error("⚠️ Model not loaded. Please check the model file and try again.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            help="Upload a clear face photo for BMI prediction"
        )
        
        if uploaded_file is not None:
            # Read image bytes once and store in session state
            if 'uploaded_image_bytes' not in st.session_state or st.session_state.get('uploaded_file_id') != id(uploaded_file):
                # Read bytes and reset file pointer
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                st.session_state['uploaded_image_bytes'] = image_bytes
                st.session_state['uploaded_file_id'] = id(uploaded_file)
            else:
                image_bytes = st.session_state['uploaded_image_bytes']
            
            # Display uploaded image
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔮 Predict BMI", type="primary", use_container_width=True):
                with st.spinner("Processing image and predicting BMI..."):
                    try:
                        # Use the stored image bytes
                        image_bytes = st.session_state['uploaded_image_bytes']
                        
                        # Make prediction
                        result = bmi_predictor.predict(image_bytes)
                        
                        if result['success']:
                            # Store results in session state
                            st.session_state['bmi'] = result['bmi']
                            st.session_state['category'] = result['category']
                            st.session_state['message'] = result.get('message', '')
                            st.session_state['predicted'] = True
                            st.rerun()
                        else:
                            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.error(traceback.format_exc())
    
    with col2:
        st.subheader("📊 Your Results")
        
        if 'predicted' in st.session_state and st.session_state['predicted']:
            bmi = st.session_state['bmi']
            category = st.session_state['category']
            message = st.session_state['message']
            
            # Display BMI value
            st.markdown(f'<div class="bmi-value">{bmi}</div>', unsafe_allow_html=True)
            
            # Display category with color
            category_colors = {
                'Underweight': '#3498db',
                'Normal weight': '#2ecc71',
                'Overweight': '#f39c12',
                'Obese': '#e74c3c'
            }
            color = category_colors.get(category, '#666')
            st.markdown(
                f'<div class="bmi-category" style="color: {color}; font-weight: bold;">{category}</div>',
                unsafe_allow_html=True
            )
            
            # Display message
            st.info(message)
            
            # BMI Scale
            st.markdown("### BMI Scale")
            
            # Calculate position on scale
            if bmi < 18.5:
                position = (bmi / 18.5) * 25
            elif bmi < 25:
                position = 25 + ((bmi - 18.5) / (25 - 18.5)) * 25
            elif bmi < 30:
                position = 50 + ((bmi - 25) / (30 - 25)) * 25
            else:
                position = min(75 + ((bmi - 30) / 10) * 25, 100)
            
            # Create scale visualization
            scale_html = f"""
            <div class="scale-bar">
                <div class="scale-segment underweight">Underweight</div>
                <div class="scale-segment normal">Normal</div>
                <div class="scale-segment overweight">Overweight</div>
                <div class="scale-segment obese">Obese</div>
            </div>
            <div style="position: relative; margin-top: -50px; height: 50px;">
                <div style="position: absolute; left: {position}%; width: 4px; height: 60px; background: #000; 
                            transform: translateX(-50%); border-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
            </div>
            """
            st.markdown(scale_html, unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Upload Another Image", use_container_width=True, key="reset_button"):
                # Clear session state
                for key in ['bmi', 'category', 'message', 'predicted', 'uploaded_image_bytes', 'uploaded_file_id']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            st.info("👆 Upload an image and click 'Predict BMI' to see your results here")
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #999;">
                <p style="font-size: 4rem; margin: 0;">📈</p>
                <p>Your results will appear here</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About:**  
This application uses advanced deep learning to predict BMI from facial images.

**Disclaimer:**  
For demonstration purposes only. Consult healthcare professionals for accurate health assessments.
""")
