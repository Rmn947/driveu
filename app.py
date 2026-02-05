"""
DriveU - Car Rental Image Comparison System
"""

import streamlit as st
from pathlib import Path
import os
import tempfile
from PIL import Image
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="DriveU",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .processed-label {
        background-color: #9C27B0;
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the segmentation model."""
    try:
        from ultralytics import YOLO
        model_path = Path(__file__).parent / "yolo26-seg.pt"
        if model_path.exists():
            return YOLO(str(model_path))
        else:
            st.error("Model file not found.")
            return None
    except ImportError:
        st.error("Required package not installed. Run: `pip install ultralytics`")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def process_image(model, image_path: str, output_path: str) -> Optional[str]:
    """Process an image and return the output path."""
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
    
    if filename_lower.startswith("pickup_"):
        img_type = "pickup"
        remainder = filename_lower[7:]
    elif filename_lower.startswith("drop_"):
        img_type = "drop"
        remainder = filename_lower[5:]
    else:
        return None, None
    
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


def main():
    # Header
    st.markdown('<h1 class="main-header">DriveU</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload pickup & drop-off photos for comparison</p>', unsafe_allow_html=True)
    
    # File uploader
    st.header("📤 Upload Car Images")
    uploaded_files = st.file_uploader(
        "Upload all pickup and drop-off images",
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
            # Process button
            col1, col2 = st.columns(2)
            with col1:
                process_btn = st.button("🔍 Process", type="primary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    if 'processed' in st.session_state:
                        del st.session_state['processed']
                    st.rerun()
            
            # Processing
            if process_btn:
                model = load_model()
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
                                
                                result = process_image(model, input_path, output_path)
                                
                                if result and os.path.exists(result):
                                    st.session_state['processed'][category][img_type] = Image.open(result).copy()
                                
                                done += 1
                                progress.progress(done / total)
                        
                        progress.empty()
                        status.empty()
                    
                    st.success("✅ Processing complete!")
            
            # Display pairs
            st.header("🔄 Image Comparison")
            
            for category, pair in complete_pairs.items():
                category_display = category.replace("_", " ").title()
                
                st.markdown(f'<div class="category-title">📸 {category_display}</div>', unsafe_allow_html=True)
                
                # Load original images
                pair['pickup'].seek(0)
                pair['drop'].seek(0)
                pickup_img = Image.open(pair['pickup'])
                drop_img = Image.open(pair['drop'])
                
                # Check if processed
                has_processed = ('processed' in st.session_state and 
                                category in st.session_state['processed'] and
                                st.session_state['processed'][category]['pickup'] and
                                st.session_state['processed'][category]['drop'])
                
                # Original images row
                st.markdown("**Original Images:**")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<span class="pickup-label">📥 PICKUP</span>', unsafe_allow_html=True)
                    st.image(pickup_img, use_container_width=True)
                with c2:
                    st.markdown('<span class="drop-label">📤 DROP-OFF</span>', unsafe_allow_html=True)
                    st.image(drop_img, use_container_width=True)
                
                # Processed images row (if available)
                if has_processed:
                    st.markdown("**Processed Images:**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<span class="processed-label">📥 PICKUP (Processed)</span>', unsafe_allow_html=True)
                        st.image(st.session_state['processed'][category]['pickup'], use_container_width=True)
                    with c2:
                        st.markdown('<span class="processed-label">📤 DROP-OFF (Processed)</span>', unsafe_allow_html=True)
                        st.image(st.session_state['processed'][category]['drop'], use_container_width=True)
                
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


if __name__ == "__main__":
    main()
