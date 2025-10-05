"""
Pose Detection Module
Handles pose detection using MediaPipe Pose and MoveNet
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from config import *

class PoseDetector:
    def __init__(self, use_mediapipe=True, use_movenet=True):
        """
        Initialize pose detector with MediaPipe and/or MoveNet
        
        Args:
            use_mediapipe (bool): Use MediaPipe Pose
            use_movenet (bool): Use MoveNet
        """
        self.use_mediapipe = use_mediapipe
        self.use_movenet = use_movenet
        
        # Initialize MediaPipe
        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
            )
        
        # Initialize MoveNet
        if self.use_movenet:
            try:
                self.movenet_model = self._load_movenet_model()
            except Exception as e:
                print(f"Warning: Could not load MoveNet model: {e}")
                self.use_movenet = False
    
    def _load_movenet_model(self):
        """Load pre-trained MoveNet model from TensorFlow Hub"""
        model_url = f"https://tfhub.dev/google/movenet/singlepose/{MOVENET_MODEL_NAME}/4"
        model = tf.saved_model.load(model_url)
        return model
    
    def detect_mediapipe(self, frame):
        """
        Detect pose using MediaPipe
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            keypoints: List of (x, y, confidence) tuples
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        keypoints = []
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                confidence = landmark.visibility
                keypoints.append((x, y, confidence))
        
        return keypoints if keypoints else None
    
    def detect_movenet(self, frame):
        """
        Detect pose using MoveNet
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            keypoints: List of (x, y, confidence) tuples
        """
        # Preprocess image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
        img = tf.cast(img, dtype=tf.int32)
        img = tf.expand_dims(img, axis=0)
        
        # Run inference
        outputs = self.movenet_model.signatures['serving_default'](img)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        keypoints = []
        
        # MoveNet returns 17 keypoints, we need to map to MediaPipe's 33
        movenet_to_mediapipe = {
            0: 0,   # nose
            1: 2,   # left_eye
            2: 5,   # right_eye
            3: 7,   # left_ear
            4: 8,   # right_ear
            5: 11,  # left_shoulder
            6: 12,  # right_shoulder
            7: 13,  # left_elbow
            8: 14,  # right_elbow
            9: 15,  # left_wrist
            10: 16, # right_wrist
            11: 23, # left_hip
            12: 24, # right_hip
            13: 25, # left_knee
            14: 26, # right_knee
            15: 27, # left_ankle
            16: 28  # right_ankle
        }
        
        # Initialize with None for all 33 MediaPipe keypoints
        keypoints = [None] * 33
        
        for i, (y_norm, x_norm, conf) in enumerate(keypoints_with_scores):
            if i in movenet_to_mediapipe:
                mp_idx = movenet_to_mediapipe[i]
                x = int(x_norm * w)
                y = int(y_norm * h)
                keypoints[mp_idx] = (x, y, conf)
        
        return keypoints
    
    def detect(self, frame):
        """
        Detect pose using available methods and combine results
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            keypoints: List of (x, y, confidence) tuples (33 points for MediaPipe format)
        """
        keypoints = None
        
        # Try MediaPipe first (more accurate and complete)
        if self.use_mediapipe:
            keypoints = self.detect_mediapipe(frame)
        
        # If MediaPipe fails or not used, try MoveNet
        if keypoints is None and self.use_movenet:
            keypoints = self.detect_movenet(frame)
        
        # If both methods available, can implement fusion logic here
        # For now, we prioritize MediaPipe
        
        return keypoints
    
    def is_valid_pose(self, keypoints):
        """
        Check if detected pose is valid for measurement
        
        Args:
            keypoints: List of keypoint tuples
            
        Returns:
            bool: True if pose is valid
        """
        if keypoints is None:
            return False
        
        # Check if key points are detected with sufficient confidence
        critical_points = [
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.LEFT_HIP,
            KeypointIndex.RIGHT_HIP
        ]
        
        for idx in critical_points:
            if idx >= len(keypoints) or keypoints[idx] is None:
                return False
            if keypoints[idx][2] < MIN_CONFIDENCE_THRESHOLD:
                return False
        
        return True
    
    def get_keypoint(self, keypoints, index):
        """
        Safely get a keypoint from the list
        
        Args:
            keypoints: List of keypoint tuples
            index: Keypoint index
            
        Returns:
            tuple: (x, y, confidence) or None if not available
        """
        if keypoints is None or index >= len(keypoints):
            return None
        return keypoints[index]
    
    def close(self):
        """Release resources"""
        if self.use_mediapipe:
            self.pose.close()