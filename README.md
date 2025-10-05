# Body-Measurements
Created a body measurement project where using media pipeline


🚀 Step-by-Step Execution Guide
Complete Setup and Execution Instructions
Step 1: System Preparation
1.1 Check Python Version
bashpython --version
# Should show Python 3.8, 3.9, or 3.10
If you don't have Python:

Windows: Download from python.org
Mac: brew install python@3.10
Linux: sudo apt-get install python3.10

1.2 Create Project Directory
bashmkdir body_measurement_assistant
cd body_measurement_assistant
Step 2: Save All Code Files
Create each file with the exact content provided:

config.py - Configuration settings
pose_detector.py - Pose detection module
calibration.py - Calibration logic
measurement_calculator.py - Measurement calculations
visualization.py - Visualization functions
main.py - Main application
streamlit_app.py - GUI application (optional)
requirements.txt - Dependencies

Quick File Creation (Windows PowerShell):
powershellNew-Item -ItemType File -Name "config.py"
New-Item -ItemType File -Name "pose_detector.py"
New-Item -ItemType File -Name "calibration.py"
New-Item -ItemType File -Name "measurement_calculator.py"
New-Item -ItemType File -Name "visualization.py"
New-Item -ItemType File -Name "main.py"
New-Item -ItemType File -Name "streamlit_app.py"
New-Item -ItemType File -Name "requirements.txt"
Quick File Creation (Mac/Linux):
bashtouch config.py pose_detector.py calibration.py measurement_calculator.py visualization.py main.py streamlit_app.py requirements.txt
Step 3: Virtual Environment Setup
3.1 Create Virtual Environment
Windows:
bashpython -m venv venv
Mac/Linux:
bashpython3 -m venv venv
3.2 Activate Virtual Environment
Windows (Command Prompt):
bashvenv\Scripts\activate
Windows (PowerShell):
bashvenv\Scripts\Activate.ps1
Mac/Linux:
bashsource venv/bin/activate
You should see (venv) prefix in your terminal.
Step 4: Install Dependencies
4.1 Upgrade pip
bashpython -m pip install --upgrade pip
4.2 Install All Requirements
bashpip install -r requirements.txt
This will install:

opencv-python (4.8.1.78)
mediapipe (0.10.7)
tensorflow (2.13.0)
numpy (1.24.3)
pandas (2.0.3)
streamlit (1.28.0)
Pillow (10.1.0)
matplotlib (3.8.0)

Note: Installation may take 5-10 minutes depending on your internet speed.
4.3 Verify Installation
bashpython -c "import cv2, mediapipe, tensorflow; print('All packages installed successfully!')"
Step 5: First Run (CLI Mode)
5.1 Basic Test Run
bashpython main.py
Expected Output:
==================================================
BODY MEASUREMENT ASSISTANT
==================================================

Instructions:
1. Stand 2-3 meters away from camera
2. Ensure full body is visible in frame
3. Press 'c' to calibrate (required before measurements)
4. Press 's' to save measurements
5. Press 'p' to take snapshot
6. Press 'q' to quit

Starting camera...
5.2 Camera Window Should Open

You should see yourself in the camera
Keypoints may not be visible yet (need calibration)

5.3 Calibration Process

Stand in T-pose (arms extended horizontally)
Press 'c' key on keyboard
Enter your height when prompted in terminal:

   Enter your height in centimeters: 175

Wait for confirmation:

   === CALIBRATION MODE ===
   Please stand upright with full body visible in frame...
   Calibration successful!
   User height: 175 cm
   Pixel height: 523.45 px
   Scale factor: 0.3342 cm/px
   ✓ Calibration successful!
Step 6: Taking Measurements
6.1 Stand Properly

Full body visible
Face camera directly
Arms slightly away from body
Legs shoulder-width apart

6.2 View Real-time Measurements

Green skeleton overlay appears
Red dots show keypoints
Magenta lines show measurements
Text displays measurement values

6.3 Save Measurements

Press 's' to save to CSV
Check outputs/measurements.csv

6.4 Take Snapshot

Press 'p' to save current frame
Check outputs/snapshots/ folder

Step 7: GUI Mode (Streamlit)
7.1 Launch Streamlit App
bashstreamlit run streamlit_app.py
7.2 Browser Opens Automatically

URL: http://localhost:8501
If not, manually open the URL shown in terminal

7.3 Using the GUI

Set Your Height (sidebar):

Enter height in cm (e.g., 175)


Upload Image:

Click "Browse files"
Select a full-body photo
Or use "Use Webcam" checkbox


View Results:

Processed image shows on right
Measurements display as metrics below


Save Measurements:

Click "💾 Save Measurements"
View in "History" tab
Download CSV