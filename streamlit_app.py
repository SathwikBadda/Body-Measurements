"""
Streamlit GUI for Body Measurement Assistant
Web-based interface for easier interaction
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import time

from config import *
from pose_detector import PoseDetector
from calibration import Calibrator
from measurement_calculator import MeasurementCalculator
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Body Measurement Assistant",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'pose_detector' not in st.session_state:
    st.session_state.pose_detector = PoseDetector(use_mediapipe=True, use_movenet=False)
if 'calibrator' not in st.session_state:
    st.session_state.calibrator = Calibrator()
if 'measurement_calculator' not in st.session_state:
    st.session_state.measurement_calculator = MeasurementCalculator(st.session_state.calibrator)
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'measurements_history' not in st.session_state:
    st.session_state.measurements_history = []

def process_image(image, user_height=None):
    """Process uploaded image"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Detect pose
    keypoints = st.session_state.pose_detector.detect(frame)
    
    # Process
    processed_frame = frame.copy()
    measurements = None
    
    if keypoints and st.session_state.pose_detector.is_valid_pose(keypoints):
        # Auto-calibrate if height provided and not calibrated
        if user_height and not st.session_state.calibrator.is_calibrated:
            st.session_state.calibrator.calibrate(keypoints, user_height)
            st.success(f"✅ Calibrated with height: {user_height} cm")
        
        # Draw visualizations
        processed_frame = st.session_state.visualizer.draw_skeleton(processed_frame, keypoints)
        processed_frame = st.session_state.visualizer.draw_keypoints(processed_frame, keypoints)
        
        # Calculate measurements if calibrated
        if st.session_state.calibrator.is_calibrated:
            measurements = st.session_state.measurement_calculator.get_averaged_measurements(keypoints)
            processed_frame = st.session_state.visualizer.draw_measurements(
                processed_frame, keypoints, measurements
            )
    else:
        st.warning("⚠️ No valid pose detected. Please ensure full body is visible.")
    
    # Convert back to RGB for display
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_frame, measurements, keypoints

def main():
    st.title("📏 Body Measurement Assistant")
    st.markdown("### AI-Powered Body Measurement using MediaPipe Pose Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        user_height = st.number_input(
            "Your Height (cm)",
            min_value=100,
            max_value=250,
            value=170,
            help="Enter your actual height for calibration"
        )
        
        st.markdown("---")
        st.markdown(GUI_SIDEBAR_INFO)
        
        st.markdown("---")
        
        # Calibration status
        if st.session_state.calibrator.is_calibrated:
            st.success("✅ System Calibrated")
            st.info(f"Height: {st.session_state.calibrator.user_height_cm} cm")
            if st.button("Reset Calibration"):
                st.session_state.calibrator = Calibrator()
                st.session_state.measurement_calculator = MeasurementCalculator(st.session_state.calibrator)
                st.rerun()
        else:
            st.warning("⚠️ Not Calibrated")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Measurement", "📊 History", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload your image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a full-body image in upright pose"
            )
            
            # Webcam option
            use_webcam = st.checkbox("Use Webcam", value=False)
            
            if use_webcam:
                img_file = st.camera_input("Take a picture")
                if img_file:
                    uploaded_file = img_file
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.subheader("Results")
            
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    processed_img, measurements, keypoints = process_image(image, user_height)
                    
                    st.image(processed_img, caption="Processed Image", use_column_width=True)
                    
                    if measurements:
                        st.success("✅ Measurements Calculated!")
                        
                        # Display measurements
                        st.markdown("### 📐 Body Measurements")
                        
                        # Create metrics
                        cols = st.columns(3)
                        idx = 0
                        for key, value in measurements.items():
                            if value is not None:
                                with cols[idx % 3]:
                                    st.metric(
                                        MEASUREMENTS[key]['name'],
                                        f"{value:.1f} cm"
                                    )
                                idx += 1
                        
                        # Validation
                        validation = st.session_state.measurement_calculator.validate_measurements(measurements)
                        if not validation['is_valid']:
                            st.warning("⚠️ Validation Warnings:")
                            for warning in validation['warnings']:
                                st.write(f"- {warning}")
                        
                        # Save button
                        if st.button("💾 Save Measurements"):
                            measurement_record = {
                                'timestamp': pd.Timestamp.now(),
                                'height_cm': user_height,
                                **measurements
                            }
                            st.session_state.measurements_history.append(measurement_record)
                            st.success("✅ Measurements saved to history!")
                    
                    else:
                        st.error("❌ Could not calculate measurements. Please calibrate first or ensure full body is visible.")
    
    with tab2:
        st.subheader("📊 Measurement History")
        
        if st.session_state.measurements_history:
            df = pd.DataFrame(st.session_state.measurements_history)
            
            # Display dataframe
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"measurements_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Visualizations
            if len(df) > 1:
                st.markdown("### 📈 Measurement Trends")
                
                # Select measurement to plot
                measurement_cols = [col for col in df.columns if col not in ['timestamp', 'height_cm']]
                selected_measurement = st.selectbox("Select measurement", measurement_cols)
                
                if selected_measurement:
                    st.line_chart(df.set_index('timestamp')[selected_measurement])
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.measurements_history = []
                st.rerun()
        else:
            st.info("No measurements recorded yet. Take measurements in the 'Measurement' tab.")
    
    with tab3:
        st.subheader("ℹ️ About This Application")
        
        st.markdown("""
        ### Body Measurement Assistant
        
        This application uses advanced AI and computer vision to estimate body measurements from images.
        
        #### 🔬 Technology Stack:
        - **MediaPipe Pose**: Google's pose detection model
        - **OpenCV**: Image processing
        - **Python**: Core implementation
        - **Streamlit**: Web interface
        
        #### 📏 Measurements Provided:
        - Shoulder Width
        - Sleeve Length (Left & Right)
        - Pant Length (Left & Right)
        - Torso Length
        - Chest Width (estimated)
        
        #### 🎯 How It Works:
        1. **Pose Detection**: Detects 33 body keypoints using MediaPipe
        2. **Calibration**: Uses your actual height to convert pixels to centimeters
        3. **Measurement**: Calculates distances between keypoints
        4. **Smoothing**: Averages multiple measurements for accuracy
        
        #### 💡 Tips for Best Results:
        - Stand 2-3 meters from camera
        - Wear fitted clothing
        - Ensure good lighting
        - Stand in T-pose or upright position
        - Make sure full body is visible
        - Use a plain background if possible
        
        #### ⚠️ Accuracy Notes:
        - Measurements are estimates and may vary ±2-5 cm
        - For professional tailoring, confirm with manual measurements
        - Works best with frontal, full-body images
        
        #### 📝 References:
        Based on research in human pose estimation and body measurement techniques
        using MediaPipe and MoveNet architectures.
        """)

if __name__ == "__main__":
    main()