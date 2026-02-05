import streamlit as st
from pathlib import Path
import os
import tempfile
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional
import requests

# Get config - priority: Streamlit secrets > config.py > environment variables
def get_config():
    """Get configuration from various sources."""
    api_key = ""
    model = "gemini-2.5-flash"
    model_url = ""
    
    # Try Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        model = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
        model_url = st.secrets.get("YOLO_MODEL_URL", "")
    except:
        pass
    
    # Fall back to config.py
    if not api_key:
        try:
            from config import GEMINI_API_KEY, GEMINI_MODEL
            api_key = GEMINI_API_KEY
            model = GEMINI_MODEL
        except ImportError:
            pass
    
    # Fall back to environment variables
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        model_url = os.environ.get("YOLO_MODEL_URL", "")
    
    return api_key, model, model_url

GEMINI_API_KEY, GEMINI_MODEL, YOLO_MODEL_URL = get_config()

# Page configuration
st.set_page_config(
    page_title="Car Damage Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pickup-label {
        background-color: #4CAF50;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .drop-label {
        background-color: #FF9800;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .category-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        padding: 8px 15px;
        background: linear-gradient(90deg, #e3f2fd, #ffffff);
        border-left: 4px solid #1E88E5;
        border-radius: 0 5px 5px 0;
    }
    .gemini-result {
        background-color: #f0f7ff;
        border: 1px solid #b3d4fc;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


def download_model(url: str, save_path: Path) -> bool:
    """Download YOLO model from URL."""
    try:
        with st.spinner("Downloading YOLO model... This may take a minute."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False


@st.cache_resource
def load_yolo_model():
    """Load the YOLO model for scratch detection."""
    try:
        from ultralytics import YOLO
        model_path = Path(__file__).parent / "yolo26-seg.pt"
        
        # If model doesn't exist locally, try to download from URL
        if not model_path.exists():
            if YOLO_MODEL_URL:
                if not download_model(YOLO_MODEL_URL, model_path):
                    return None
            else:
                st.error(f"YOLO model not found. Please upload yolo26-seg.pt or set YOLO_MODEL_URL in secrets.")
                return None
        
        return YOLO(str(model_path))
    except ImportError:
        st.error("Ultralytics not installed. Run: `pip install ultralytics`")
        return None
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


def process_image_with_yolo(model, image_path: str, output_path: str) -> Optional[str]:
    """Process an image with YOLO model and return the output path."""
    try:
        results = model.predict(source=image_path, verbose=False)
        if results:
            results[0].save(filename=output_path)
            return output_path
        return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def extract_category_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the category and type (pickup/drop) from filename."""
    filename_lower = filename.lower()
    
    # Check for pickup or drop prefix
    if filename_lower.startswith("pickup_"):
        img_type = "pickup"
        remainder = filename_lower[7:]  # Remove "pickup_"
    elif filename_lower.startswith("drop_"):
        img_type = "drop"
        remainder = filename_lower[5:]  # Remove "drop_"
    else:
        return None, None
    
    # Remove file extension
    remainder = Path(remainder).stem
    
    return img_type, remainder


def pair_images(uploaded_files: List) -> Dict[str, Dict[str, any]]:
    """Pair uploaded images based on their naming convention."""
    pairs = {}
    
    for uploaded_file in uploaded_files:
        img_type, category = extract_category_from_filename(uploaded_file.name)
        
        if img_type is None or category is None:
            continue
            
        if category not in pairs:
            pairs[category] = {"pickup": None, "drop": None}
        
        pairs[category][img_type] = uploaded_file
    
    return pairs


def analyze_with_gemini(pickup_image: Image.Image, drop_image: Image.Image, category: str) -> str:
    """Send images to Gemini for scratch detection analysis."""
    api_key = GEMINI_API_KEY
    
    if not api_key or api_key == "your-gemini-api-key-here":
        return "❌ Please set your Gemini API key in `config.py`"
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""You are an expert car damage assessor for a car rental company. 

I am providing you with two images of the same part of a car: "{category.replace('_', ' ')}".

**Image 1**: PICKUP image (when the car was rented out)
**Image 2**: DROP-OFF image (when the car was returned)

Please analyze both images carefully and provide:

1. **New Damage Found**: List any NEW scratches, dents, or damage that appeared between pickup and drop-off.

2. **Damage Details** (if any):
   - Location on the car part
   - Type (scratch, dent, scuff, crack, etc.)
   - Severity (minor/moderate/severe)

3. **Condition Summary**: Brief comparison of before vs after condition.

4. **Recommendation**: Whether repair is needed and urgency level.

Be concise and professional."""

        response = model.generate_content([prompt, pickup_image, drop_image])
        return response.text
        
    except ImportError:
        return "❌ Install google-generativeai: `pip install google-generativeai`"
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"


def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Rental Damage Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload pickup & drop-off photos → Auto-pair → YOLO Detection → AI Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload** 10-20 car photos
        2. Images auto-pair by name
        3. Click **Process with YOLO**
        4. Optionally use **Gemini AI** analysis
        """)
        
        st.markdown("---")
        
        st.header("📝 Naming Format")
        st.code("pickup_car_front.jpg\ndrop_car_front.jpg")
        
        st.markdown("---")
        
        st.header("⚙️ Config Status")
        if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
            st.success(f"✅ Gemini API configured ({GEMINI_MODEL})")
        else:
            st.warning("⚠️ Set API key in config.py")
    
    # File uploader
    st.header("📤 Upload Car Images")
    uploaded_files = st.file_uploader(
        "Upload all pickup and drop-off images (10-20 photos)",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        help="Name format: pickup_[part].jpg and drop_[part].jpg"
    )
    
    if uploaded_files:
        # Pair the images
        pairs = pair_images(uploaded_files)
        
        # Separate complete and incomplete pairs
        complete_pairs = {k: v for k, v in pairs.items() if v['pickup'] and v['drop']}
        incomplete = {k: v for k, v in pairs.items() if not (v['pickup'] and v['drop'])}
        unpaired = [f.name for f in uploaded_files if extract_category_from_filename(f.name)[0] is None]
        
        # Statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📷 Uploaded", len(uploaded_files))
        with col2:
            st.metric("✅ Paired", len(complete_pairs))
        with col3:
            st.metric("⚠️ Incomplete", len(incomplete))
        with col4:
            st.metric("❓ Unpaired", len(unpaired))
        
        # Show warnings for incomplete/unpaired
        if incomplete or unpaired:
            with st.expander("⚠️ View Issues"):
                if incomplete:
                    st.markdown("**Incomplete pairs (missing pickup or drop):**")
                    for cat, pair in incomplete.items():
                        missing = "drop" if pair['pickup'] else "pickup"
                        st.write(f"- `{cat}`: missing **{missing}** image")
                if unpaired:
                    st.markdown("**Unpaired files (invalid naming):**")
                    for name in unpaired:
                        st.write(f"- `{name}`")
        
        st.markdown("---")
        
        if complete_pairs:
            # Process buttons
            col1, col2 = st.columns(2)
            with col1:
                process_btn = st.button("🔍 Process All with YOLO", type="primary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    if 'processed' in st.session_state:
                        del st.session_state['processed']
                    st.rerun()
            
            # YOLO Processing
            if process_btn:
                model = load_yolo_model()
                if model:
                    st.session_state['processed'] = {}
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        progress = st.progress(0)
                        status = st.empty()
                        
                        total = len(complete_pairs) * 2
                        done = 0
                        
                        for category, pair in complete_pairs.items():
                            st.session_state['processed'][category] = {'pickup': None, 'drop': None}
                            
                            for img_type in ['pickup', 'drop']:
                                status.text(f"Processing {category} - {img_type}...")
                                
                                file_obj = pair[img_type]
                                file_obj.seek(0)
                                img_data = file_obj.read()
                                file_obj.seek(0)
                                
                                input_path = os.path.join(temp_dir, f"{img_type}_{category}.jpg")
                                output_path = os.path.join(temp_dir, f"{img_type}_{category}_out.jpg")
                                
                                with open(input_path, 'wb') as f:
                                    f.write(img_data)
                                
                                result = process_image_with_yolo(model, input_path, output_path)
                                
                                if result and os.path.exists(result):
                                    st.session_state['processed'][category][img_type] = Image.open(result).copy()
                                
                                done += 1
                                progress.progress(done / total)
                        
                        progress.empty()
                        status.empty()
                    
                    st.success("✅ YOLO processing complete!")
            
            # Display pairs
            st.header("🔄 Image Pairs & Comparison")
            
            for category, pair in complete_pairs.items():
                category_display = category.replace("_", " ").title()
                
                st.markdown(f'<div class="category-title">📸 {category_display}</div>', unsafe_allow_html=True)
                
                # Load images
                pair['pickup'].seek(0)
                pair['drop'].seek(0)
                pickup_img = Image.open(pair['pickup'])
                drop_img = Image.open(pair['drop'])
                
                # Check if processed
                has_processed = ('processed' in st.session_state and 
                                category in st.session_state['processed'] and
                                st.session_state['processed'][category]['pickup'] and
                                st.session_state['processed'][category]['drop'])
                
                # Tabs for views
                tab1, tab2, tab3 = st.tabs(["📷 Original", "🔍 YOLO Result", "🤖 Gemini AI"])
                
                with tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<span class="pickup-label">📥 PICKUP</span>', unsafe_allow_html=True)
                        st.image(pickup_img, width="stretch")
                    with c2:
                        st.markdown('<span class="drop-label">📤 DROP-OFF</span>', unsafe_allow_html=True)
                        st.image(drop_img, width="stretch")
                
                with tab2:
                    if has_processed:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown('<span class="pickup-label">📥 PICKUP (YOLO)</span>', unsafe_allow_html=True)
                            st.image(st.session_state['processed'][category]['pickup'], width="stretch")
                        with c2:
                            st.markdown('<span class="drop-label">📤 DROP-OFF (YOLO)</span>', unsafe_allow_html=True)
                            st.image(st.session_state['processed'][category]['drop'], width="stretch")
                    else:
                        st.info("👆 Click 'Process All with YOLO' to detect scratches")
                
                with tab3:
                    if st.button(f"🤖 Analyze with Gemini", key=f"gem_{category}"):
                        with st.spinner("Analyzing with Gemini AI..."):
                            result = analyze_with_gemini(pickup_img, drop_img, category)
                        st.markdown(result)
                
                st.markdown("---")
        
        else:
            st.warning("No complete pairs found. Ensure images have matching pickup_* and drop_* names.")
    
    else:
        st.info("👆 Upload your car images above to get started")
        
        with st.expander("📋 Example Naming Convention"):
            st.markdown("""
            | Pickup | Drop-off |
            |--------|----------|
            | `pickup_car_front.jpg` | `drop_car_front.jpg` |
            | `pickup_car_rear.jpg` | `drop_car_rear.jpg` |
            | `pickup_car_bonnet.jpg` | `drop_car_bonnet.jpg` |
            | `pickup_car_left_front_door.jpg` | `drop_car_left_front_door.jpg` |
            | ... | ... |
            """)
    
    # Footer
    st.markdown("---")
    st.caption("🚗 Car Rental Damage Detection | YOLO + Gemini AI")


if __name__ == "__main__":
    main()
