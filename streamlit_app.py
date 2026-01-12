"""
BMI Predictor - Complete Streamlit Application
Rebuilt from scratch - Clean, organized, and deployment-ready.
"""
import streamlit as st
import os
import io
from PIL import Image
import pandas as pd
import random
from models.bmi_predictor import BMIPredictor

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BMI Predictor - AI-Powered Body Mass Index Calculator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Matching Flask App Design
# ============================================================================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --sidebar-bg: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        --card-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: var(--primary-gradient) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: var(--sidebar-bg) !important;
        padding: 1rem;
    }
    
    .custom-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    .bmi-value-large {
        font-size: 5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        text-align: center;
        margin: 1rem 0;
    }
    
    .bmi-category-large {
        font-weight: 600;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    
    .bmi-scale-container {
        display: flex;
        border-radius: 10px;
        overflow: hidden;
        margin: 2rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        height: 50px;
    }
    
    .scale-seg {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.85rem;
    }
    
    .scale-underweight { background: #3498db; }
    .scale-normal { background: #2ecc71; }
    .scale-overweight { background: #f39c12; }
    .scale-obese { background: #e74c3c; }
    
    .sample-card-wrapper {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .sample-card-wrapper:hover {
        transform: translateY(-5px);
    }
    
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'bmi_predictor' not in st.session_state:
    st.session_state.bmi_predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'uploaded_image_bytes' not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if 'samples_data' not in st.session_state:
    st.session_state.samples_data = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_bmi_predictor():
    """Load BMI predictor model (cached for performance)"""
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
        import traceback
        traceback.print_exc()
        return None, str(e)

def get_bmi_category(bmi):
    """Get BMI category, color, and icon"""
    if bmi < 18.5:
        return "Underweight", "#17a2b8", "⚠️"
    elif bmi < 25:
        return "Normal weight", "#28a745", "✅"
    elif bmi < 30:
        return "Overweight", "#ffc107", "⚠️"
    else:
        return "Obese", "#dc3545", "⚠️"

@st.cache_data
def load_samples_data():
    """Load and filter samples data (cached)"""
    csv_path = os.path.join('samples', 'dataset.csv')
    
    if not os.path.exists(csv_path):
        return None, "CSV file not found"
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'image_filename' not in df.columns or 'BMI' not in df.columns:
            return None, "Required columns not found in CSV"
        
        columns_to_return = ['image_filename', 'BMI']
        if 'id' in df.columns:
            columns_to_return.insert(0, 'id')
        
        df_clean = df[columns_to_return].copy()
        df_clean = df_clean.dropna(subset=['image_filename', 'BMI'])
        df_clean['BMI'] = pd.to_numeric(df_clean['BMI'], errors='coerce')
        df_clean = df_clean.dropna(subset=['BMI'])
        
        # Filter samples with existing images
        images_dir = os.path.join('samples', 'front')
        samples_with_images = []
        
        if os.path.exists(images_dir):
            available_files_lower = {}
            for file in os.listdir(images_dir):
                file_lower = file.lower()
                if file_lower not in available_files_lower:
                    available_files_lower[file_lower] = file
            
            for _, row in df_clean.iterrows():
                image_filename = row['image_filename']
                image_path = os.path.join(images_dir, image_filename)
                
                # Try exact match
                if os.path.exists(image_path):
                    samples_with_images.append(row.to_dict())
                    continue
                
                # Try case-insensitive match
                filename_lower = image_filename.lower()
                if filename_lower in available_files_lower:
                    samples_with_images.append(row.to_dict())
                    continue
                
                # Try different extensions
                if filename_lower.endswith(('.jpg', '.jpeg')):
                    base_name = os.path.splitext(image_filename)[0]
                    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                        test_filename = base_name + ext
                        if test_filename.lower() in available_files_lower:
                            samples_with_images.append(row.to_dict())
                            break
        
        return samples_with_images, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

def find_image_file(filename, images_dir):
    """Find image file with case-insensitive matching"""
    if not os.path.exists(images_dir):
        return None
    
    # Try exact match
    image_path = os.path.join(images_dir, filename)
    if os.path.exists(image_path):
        return image_path
    
    # Try case-insensitive
    filename_lower = filename.lower()
    for file in os.listdir(images_dir):
        if file.lower() == filename_lower:
            return os.path.join(images_dir, file)
    
    # Try different extensions
    if filename_lower.endswith(('.jpg', '.jpeg')):
        base_name = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            test_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(test_path):
                return test_path
    
    return None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.25); margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 1.5rem; margin-bottom: 0.5rem;">
            <i class="bi bi-robot"></i> BMI Predictor
        </h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">AI-Powered Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Samples", "ℹ️ Privacy Policy"],
        label_visibility="collapsed",
        key="nav"
    )
    
    st.markdown("---")
    
    st.markdown("### Model Status")
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Model Not Loaded")
        if st.button("🔄 Reload Model", use_container_width=True):
            st.session_state.bmi_predictor = None
            st.cache_resource.clear()
            st.rerun()

# ============================================================================
# MAIN CONTENT - HOME PAGE
# ============================================================================
if page == "🏠 Home":
    # Load model
    if st.session_state.bmi_predictor is None:
        with st.spinner("Loading BMI prediction model..."):
            predictor, error = load_bmi_predictor()
            if predictor:
                st.session_state.bmi_predictor = predictor
                st.session_state.model_loaded = True
            else:
                st.session_state.model_loaded = False
                if error:
                    st.error(f"Failed to load model: {error}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📖 Introduction", "⚙️ How It Works", "📱 The App"])
    
    # Introduction Tab
    with tab1:
        st.markdown('<h2 style="color: white; font-weight: bold; margin-bottom: 2rem;">Introduction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #333; margin-bottom: 1.5rem;">Welcome to BMI Predictor</h3>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; line-height: 1.6;">
                Our AI-powered BMI (Body Mass Index) Predictor uses advanced machine learning to estimate your BMI from a single photograph. 
                This innovative tool combines computer vision, deep learning, and multi-modal data analysis to provide accurate predictions.
            </p>
            
            <div style="background: #e7f3ff; padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; text-align: center;">
                <p style="color: #667eea; font-weight: bold; font-size: 1.1rem; margin: 0;">
                    <i class="bi bi-university"></i>
                    Developed by the Qatar University research team, led by Dr. Amith Khandakar
                </p>
            </div>
            
            <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">What is BMI?</h4>
            <p style="color: #666; margin-bottom: 1rem; line-height: 1.6;">
                Body Mass Index (BMI) is a measure of body fat based on height and weight. It's calculated by dividing your weight in kilograms 
                by the square of your height in meters. BMI categories include:
            </p>
            <ul style="color: #666; line-height: 1.8;">
                <li><strong>Underweight:</strong> BMI less than 18.5</li>
                <li><strong>Normal weight:</strong> BMI 18.5 to 24.9</li>
                <li><strong>Overweight:</strong> BMI 25 to 29.9</li>
                <li><strong>Obese:</strong> BMI 30 or higher</li>
            </ul>
            
            <div style="background: #fff3cd; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
                <p style="color: #856404; margin: 0; line-height: 1.6;">
                    <i class="bi bi-exclamation-triangle-fill"></i>
                    <strong>Important Note:</strong> This tool is for demonstration purposes only. For accurate health assessments, 
                    please consult with a healthcare professional. BMI is a screening tool and may not accurately reflect body fat 
                    percentage for all individuals.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works Tab
    with tab2:
        st.markdown('<h2 style="color: white; font-weight: bold; margin-bottom: 2rem;">How the Model Works</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #333; margin-bottom: 1.5rem;">Hybrid Deep Learning Architecture</h3>
            <p style="color: #666; margin-bottom: 2rem; line-height: 1.6;">
                Our model uses a sophisticated hybrid architecture that combines multiple data modalities:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2.5rem; color: #667eea; margin-bottom: 1rem;">
                    <i class="bi bi-image"></i>
                </div>
                <h5 style="color: #333; font-weight: bold; margin-bottom: 1rem;">Image Processing</h5>
                <p style="color: #666; margin: 0; line-height: 1.6;">
                    Uses a custom convolutional neural network to extract visual features from facial images, 
                    analyzing facial structure and body proportions.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2.5rem; color: #28a745; margin-bottom: 1rem;">
                    <i class="bi bi-diagram-3"></i>
                </div>
                <h5 style="color: #333; font-weight: bold; margin-bottom: 1rem;">Graph Neural Network</h5>
                <p style="color: #666; margin: 0; line-height: 1.6;">
                    Analyzes 21 facial landmarks using a multi-scale Graph Convolutional Network (GCN) to understand 
                    spatial relationships between facial features.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2.5rem; color: #17a2b8; margin-bottom: 1rem;">
                    <i class="bi bi-bar-chart"></i>
                </div>
                <h5 style="color: #333; font-weight: bold; margin-bottom: 1rem;">Tabular Features</h5>
                <p style="color: #666; margin: 0; line-height: 1.6;">
                    Processes 36 engineered features including facial measurements, ratios, and demographic information 
                    through a multi-layer neural network.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2.5rem; color: #ffc107; margin-bottom: 1rem;">
                    <i class="bi bi-arrow-left-right"></i>
                </div>
                <h5 style="color: #333; font-weight: bold; margin-bottom: 1rem;">Fusion & Prediction</h5>
                <p style="color: #666; margin: 0; line-height: 1.6;">
                    Combines all features using multi-head attention mechanism and shared layers to generate 
                    accurate BMI predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #333; margin-bottom: 1rem;">Model Training</h4>
            <p style="color: #666; margin: 0; line-height: 1.6;">
                The model was trained using 5-fold cross-validation on a comprehensive dataset, ensuring robust performance 
                and generalization. It uses multi-task learning to simultaneously predict BMI, age, sex, and BMI category.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # The App Tab
    with tab3:
        st.markdown('<h2 style="color: white; font-weight: bold; margin-bottom: 2rem;">The App</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #333; margin-bottom: 1.5rem;">
                    <i class="bi bi-upload" style="color: #667eea;"></i> Upload Your Image
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                help="Upload a facial image to predict BMI",
                key="main_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    # Read image bytes immediately and store
                    uploaded_file.seek(0)  # Reset file pointer
                    image_bytes = uploaded_file.read()
                    
                    # Validate image bytes
                    if len(image_bytes) == 0:
                        st.error("Error: Empty file uploaded")
                        st.session_state.uploaded_image_bytes = None
                    else:
                        # Verify it's a valid image and convert to bytes if needed
                        try:
                            # Ensure we have raw bytes (not BytesIO)
                            if isinstance(image_bytes, io.BytesIO):
                                image_bytes = image_bytes.read()
                            
                            # Verify image can be opened
                            img_io = io.BytesIO(image_bytes)
                            test_image = Image.open(img_io)
                            # Convert to RGB for consistency
                            if test_image.mode != 'RGB':
                                test_image = test_image.convert('RGB')
                            
                            # Store raw bytes for prediction
                            st.session_state.uploaded_image_bytes = image_bytes
                            
                            # Display image
                            st.image(test_image, caption="Uploaded Image", use_container_width=True)
                        except Exception as img_error:
                            st.error(f"Invalid image file: {str(img_error)}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.session_state.uploaded_image_bytes = None
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    st.session_state.uploaded_image_bytes = None
            else:
                st.session_state.uploaded_image_bytes = None
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #333; margin-bottom: 1.5rem;">
                    <i class="bi bi-graph-up-arrow" style="color: #28a745;"></i> Your Results
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.uploaded_image_bytes is None:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #999;">
                    <i class="bi bi-graph-up" style="font-size: 4rem; opacity: 0.3;"></i>
                    <p style="margin-top: 1rem; color: #999;">Upload an image and click "Predict BMI" to see your results here</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("🔮 Predict BMI", type="primary", use_container_width=True):
                    if st.session_state.bmi_predictor is None:
                        st.error("Model not loaded. Please reload the model from the sidebar.")
                    else:
                        with st.spinner("Analyzing image and predicting BMI..."):
                            try:
                                # Ensure we have raw bytes
                                image_bytes = st.session_state.uploaded_image_bytes
                                if isinstance(image_bytes, io.BytesIO):
                                    image_bytes = image_bytes.read()
                                
                                # Validate bytes before prediction
                                if not isinstance(image_bytes, bytes):
                                    st.error("Error: Invalid image data format")
                                    st.session_state.prediction_result = None
                                elif len(image_bytes) == 0:
                                    st.error("Error: Empty image data")
                                    st.session_state.prediction_result = None
                                else:
                                    # Predict BMI
                                    result = st.session_state.bmi_predictor.predict(image_bytes)
                                    
                                    if result['success']:
                                        st.session_state.prediction_result = result
                                        st.success("✅ Prediction completed!")
                                    else:
                                        st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                                        st.session_state.prediction_result = None
                            except Exception as e:
                                st.error(f"❌ Error during prediction: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                                st.session_state.prediction_result = None
                
                # Display results
                if st.session_state.prediction_result:
                    result = st.session_state.prediction_result
                    bmi = result['bmi']
                    category = result['category']
                    message = result.get('message', '')
                    
                    category_name, color, icon = get_bmi_category(bmi)
                    
                    st.markdown(f'<div class="bmi-value-large">{bmi:.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bmi-category-large" style="color: {color};">{icon} {category_name}</div>', unsafe_allow_html=True)
                    
                    st.info(message)
                    
                    st.markdown("""
                    <div class="bmi-scale-container">
                        <div class="scale-seg scale-underweight">Underweight</div>
                        <div class="scale-seg scale-normal">Normal</div>
                        <div class="scale-seg scale-overweight">Overweight</div>
                        <div class="scale-seg scale-obese">Obese</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                    
                    if st.button("🔄 Upload Another Image", use_container_width=True):
                        st.session_state.uploaded_image_bytes = None
                        st.session_state.prediction_result = None
                        st.rerun()

# ============================================================================
# SAMPLES PAGE
# ============================================================================
elif page == "📊 Samples":
    st.markdown('<h2 style="color: white; font-weight: bold; margin-bottom: 2rem;">Sample Images with True BMI Values</h2>', unsafe_allow_html=True)
    
    # Load samples data
    if st.session_state.samples_data is None:
        with st.spinner("Loading samples data..."):
            samples_data, error = load_samples_data()
            if error:
                st.error(f"❌ Error loading samples: {error}")
                st.session_state.samples_data = []
            else:
                st.session_state.samples_data = samples_data or []
    
    # Reload button
    if st.button("🔄 Reload Samples", use_container_width=True):
        st.cache_data.clear()
        st.session_state.samples_data = None
        st.rerun()
    
    if len(st.session_state.samples_data) == 0:
        st.warning("⚠️ No samples with images found.")
        images_dir = os.path.join('samples', 'front')
        csv_path = os.path.join('samples', 'dataset.csv')
        
        with st.expander("Debug Information"):
            st.write(f"**CSV Path:** {csv_path}")
            st.write(f"**CSV Exists:** {os.path.exists(csv_path)}")
            st.write(f"**Images Directory:** {images_dir}")
            st.write(f"**Images Directory Exists:** {os.path.exists(images_dir)}")
            if os.path.exists(images_dir):
                files = os.listdir(images_dir)
                st.write(f"**Files in directory:** {len(files)}")
                if len(files) > 0:
                    st.write(f"**First 5 files:** {files[:5]}")
    else:
        st.markdown("""
        <div class="custom-card">
        """, unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_bmi = st.number_input("Min BMI", min_value=0.0, max_value=50.0, value=0.0, step=0.1, key="min_bmi")
        with col2:
            max_bmi = st.number_input("Max BMI", min_value=0.0, max_value=50.0, value=50.0, step=0.1, key="max_bmi")
        with col3:
            num_samples = st.number_input("Number of Samples", min_value=1, max_value=100, value=10, step=1, key="num_samples")
        
        # Filter samples
        filtered = [s for s in st.session_state.samples_data if min_bmi <= float(s['BMI']) <= max_bmi]
        
        if len(filtered) == 0:
            st.warning(f"⚠️ No samples found in BMI range {min_bmi} - {max_bmi}")
        else:
            # Random sample if needed
            if len(filtered) > num_samples:
                filtered = random.sample(filtered, num_samples)
            
            st.info(f"ℹ️ Showing {len(filtered)} sample(s) (Total available with images: {len(st.session_state.samples_data)})")
            
            # Display samples in grid
            images_dir = os.path.join('samples', 'front')
            cols_per_row = 4
            
            for i in range(0, len(filtered), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, sample in enumerate(filtered[i:i+cols_per_row]):
                    with cols[j]:
                        bmi_value = float(sample['BMI'])
                        filename = sample['image_filename']
                        sample_id = sample.get('id', filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', ''))
                        
                        category, color, icon = get_bmi_category(bmi_value)
                        
                        # Find and display image
                        image_path = find_image_file(filename, images_dir)
                        
                        if image_path and os.path.exists(image_path):
                            try:
                                img = Image.open(image_path)
                                st.image(img, caption=f"BMI: {bmi_value:.2f}", use_container_width=True)
                            except Exception as e:
                                st.markdown(f"""
                                <div style="background: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                    <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                                    <h4 style="color: white; margin: 0.5rem 0;">BMI: {bmi_value:.2f}</h4>
                                    <p style="color: white; margin: 0.5rem 0;">{category}</p>
                                    <small style="color: rgba(255,255,255,0.8);">{sample_id}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                                <h4 style="color: white; margin: 0.5rem 0;">BMI: {bmi_value:.2f}</h4>
                                <p style="color: white; margin: 0.5rem 0;">{category}</p>
                                <small style="color: rgba(255,255,255,0.8);">{sample_id}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Card footer
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: white; border-radius: 0 0 10px 10px; margin-top: -5px;">
                            <h6 style="color: {color}; font-weight: bold; margin: 0.5rem 0; font-size: 1.1rem;">BMI: {bmi_value:.2f}</h6>
                            <small style="color: #666; display: block; margin: 0.5rem 0;">{category}</small>
                            <small style="color: #999; font-size: 0.7rem;">ID: {sample_id}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PRIVACY POLICY PAGE
# ============================================================================
elif page == "ℹ️ Privacy Policy":
    st.markdown('<h2 style="color: white; font-weight: bold; margin-bottom: 2rem;">Privacy Policy</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #333; margin-bottom: 1.5rem;">Terms of Use and Restrictions</h3>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Non-Commercial Use Only</h4>
        <p style="color: #666; margin-bottom: 1.5rem; line-height: 1.6;">
            This BMI Predictor application is provided for <strong>research and educational purposes only</strong>. 
            The use of this application for commercial purposes is <strong>strictly prohibited</strong> without explicit 
            written permission from Qatar University.
        </p>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Data Privacy</h4>
        <ul style="color: #666; margin-bottom: 1.5rem; line-height: 1.8;">
            <li><strong>No Data Storage:</strong> Images uploaded to this application are processed in real-time and are 
                <strong>not stored</strong> on our servers.</li>
            <li><strong>No Personal Information Collection:</strong> We do not collect, store, or transmit any personal 
                information about users.</li>
            <li><strong>Temporary Processing:</strong> Images are processed temporarily in memory and are immediately 
                discarded after prediction.</li>
        </ul>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Restrictions</h4>
        <ol style="color: #666; margin-bottom: 1.5rem; line-height: 1.8;">
            <li style="margin-bottom: 1rem;">
                <strong>Commercial Use Prohibited:</strong> This application may not be used for any commercial purposes, 
                including but not limited to:
                <ul style="margin-top: 0.5rem;">
                    <li>Commercial health assessments</li>
                    <li>Paid services</li>
                    <li>Integration into commercial products</li>
                    <li>Resale or redistribution</li>
                </ul>
            </li>
            <li style="margin-bottom: 1rem;">
                <strong>Medical Disclaimer:</strong>
                <ul style="margin-top: 0.5rem;">
                    <li>This tool is <strong>NOT</strong> a medical device</li>
                    <li>Results are <strong>NOT</strong> a substitute for professional medical advice</li>
                    <li><strong>DO NOT</strong> use this tool for medical diagnosis or treatment decisions</li>
                    <li>Always consult qualified healthcare professionals for health assessments</li>
                </ul>
            </li>
            <li style="margin-bottom: 1rem;">
                <strong>Research Use:</strong>
                <ul style="margin-top: 0.5rem;">
                    <li>This application is intended for research and educational purposes</li>
                    <li>Users may use it for academic research with proper attribution</li>
                    <li>Any research publications using this tool should acknowledge Qatar University</li>
                </ul>
            </li>
        </ol>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Intellectual Property</h4>
        <ul style="color: #666; margin-bottom: 1.5rem; line-height: 1.8;">
            <li>The model, algorithms, and application are the intellectual property of Qatar University</li>
            <li>Unauthorized reproduction, distribution, or modification is prohibited</li>
            <li>For licensing inquiries, please contact Qatar University</li>
        </ul>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Limitation of Liability</h4>
        <p style="color: #666; margin-bottom: 1rem; line-height: 1.6;">
            Qatar University and its researchers are not liable for:
        </p>
        <ul style="color: #666; margin-bottom: 1.5rem; line-height: 1.8;">
            <li>Any decisions made based on predictions from this application</li>
            <li>Any inaccuracies in BMI predictions</li>
            <li>Any consequences arising from the use or misuse of this application</li>
        </ul>
        
        <h4 style="color: #333; margin-top: 2rem; margin-bottom: 1rem;">Contact</h4>
        <p style="color: #666; margin-bottom: 1rem; line-height: 1.6;">
            For questions, licensing inquiries, or permission requests, please contact:
        </p>
        <ul style="color: #666; margin-bottom: 1.5rem; line-height: 1.8;">
            <li><strong>Qatar University Research Team</strong></li>
            <li><strong>Led by Dr. Amith Khandakar</strong></li>
        </ul>
        
        <hr style="margin: 2rem 0; border-color: #ddd;">
        
        <p style="color: #999; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <strong>Last Updated:</strong> January 2025
        </p>
        <p style="color: #999; font-size: 0.9rem; margin-top: 0.5rem;">
            <strong>By using this application, you agree to these terms and restrictions.</strong>
        </p>
        
        <div style="background: #fff3cd; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
            <p style="color: #856404; margin: 0; line-height: 1.6;">
                <i class="bi bi-exclamation-triangle-fill"></i>
                <strong>Important:</strong> This application is for research and educational purposes only. 
                Commercial use is strictly prohibited. Always consult healthcare professionals for medical advice.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); padding: 2rem;">
    <p style="margin: 0.5rem 0; font-size: 1rem;">BMI Predictor - AI-Powered Body Mass Index Calculator</p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">Built with ❤️ using Streamlit and PyTorch</p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem; color: rgba(255,255,255,0.7);">Developed by Qatar University Research Team</p>
</div>
""", unsafe_allow_html=True)
