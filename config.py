"""
Configuration file for Body Measurement Assistant
Contains all settings, constants, and parameters
"""

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# Pose Detection Settings
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0, 1, or 2 (higher = more accurate but slower)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# MoveNet Settings
MOVENET_MODEL_NAME = 'movenet_thunder'  # 'movenet_lightning' or 'movenet_thunder'
MOVENET_INPUT_SIZE = 256  # 192 for lightning, 256 for thunder

# Measurement Settings
SMOOTHING_WINDOW = 10  # Number of frames to average for smoothing
MIN_CONFIDENCE_THRESHOLD = 0.3  # Minimum keypoint confidence to use
DEFAULT_HEIGHT_CM = 170  # Default height if not provided

# Keypoint Indices (MediaPipe Pose)
class KeypointIndex:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Measurement Definitions
MEASUREMENTS = {
    'shoulder_width': {
        'points': [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER],
        'name': 'Shoulder Width'
    },
    'left_sleeve_length': {
        'points': [KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST],
        'name': 'Left Sleeve Length'
    },
    'right_sleeve_length': {
        'points': [KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST],
        'name': 'Right Sleeve Length'
    },
    'left_pant_length': {
        'points': [KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE],
        'name': 'Left Pant Length'
    },
    'right_pant_length': {
        'points': [KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE],
        'name': 'Right Pant Length'
    },
    'chest_width': {
        'points': [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER],
        'name': 'Chest Width',
        'multiplier': 0.7  # Approximate chest width from shoulder points
    },
    'torso_length': {
        'points': [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER, 
                   KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP],
        'name': 'Torso Length',
        'type': 'vertical'  # Measure vertical distance
    }
}

# Visualization Settings
SKELETON_COLOR = (0, 255, 0)  # Green
KEYPOINT_COLOR = (0, 0, 255)  # Red
MEASUREMENT_LINE_COLOR = (255, 0, 255)  # Magenta
TEXT_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (0, 0, 0)  # Black
KEYPOINT_RADIUS = 5
LINE_THICKNESS = 2
TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# Skeleton connections for visualization
POSE_CONNECTIONS = [
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
    (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR)
]

# Output Settings
OUTPUT_DIR = 'outputs'
MEASUREMENTS_CSV = 'outputs/measurements.csv'
SNAPSHOT_DIR = 'outputs/snapshots'

# GUI Settings (for Streamlit)
GUI_TITLE = "Body Measurement Assistant"
GUI_SIDEBAR_INFO = """
### Instructions:
1. Stand 2-3 meters from camera
2. Ensure full body is visible
3. Stand in T-pose or upright
4. Enter your actual height
5. Click 'Start Measurement'
"""