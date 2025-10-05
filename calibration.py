"""
Calibration Module
Handles pixel-to-centimeter calibration using user height
"""

import numpy as np
from config import KeypointIndex

class Calibrator:
    def __init__(self):
        """Initialize calibrator"""
        self.scale_factor = None  # pixels per cm
        self.is_calibrated = False
        self.user_height_cm = None
    
    def calibrate(self, keypoints, user_height_cm):
        """
        Calibrate pixel-to-cm ratio using user's actual height
        
        Args:
            keypoints: List of detected keypoints
            user_height_cm: User's actual height in centimeters
            
        Returns:
            bool: True if calibration successful
        """
        if keypoints is None:
            print("Error: No keypoints detected for calibration")
            return False
        
        # Calculate pixel height from head to ankle
        pixel_height = self._calculate_body_height_pixels(keypoints)
        
        if pixel_height is None or pixel_height == 0:
            print("Error: Could not calculate pixel height")
            return False
        
        # Calculate scale factor (cm per pixel)
        self.scale_factor = user_height_cm / pixel_height
        self.user_height_cm = user_height_cm
        self.is_calibrated = True
        
        print(f"Calibration successful!")
        print(f"User height: {user_height_cm} cm")
        print(f"Pixel height: {pixel_height:.2f} px")
        print(f"Scale factor: {self.scale_factor:.4f} cm/px")
        
        return True
    
    def _calculate_body_height_pixels(self, keypoints):
        """
        Calculate body height in pixels from keypoints
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            float: Body height in pixels or None if cannot calculate
        """
        # Method 1: Top of head to ankle (most accurate)
        top_point = self._get_top_point(keypoints)
        bottom_point = self._get_bottom_point(keypoints)
        
        if top_point and bottom_point:
            height = self._euclidean_distance(top_point[:2], bottom_point[:2])
            return height
        
        # Method 2: Shoulder to ankle (fallback)
        left_shoulder = keypoints[KeypointIndex.LEFT_SHOULDER] if len(keypoints) > KeypointIndex.LEFT_SHOULDER else None
        right_shoulder = keypoints[KeypointIndex.RIGHT_SHOULDER] if len(keypoints) > KeypointIndex.RIGHT_SHOULDER else None
        left_ankle = keypoints[KeypointIndex.LEFT_ANKLE] if len(keypoints) > KeypointIndex.LEFT_ANKLE else None
        right_ankle = keypoints[KeypointIndex.RIGHT_ANKLE] if len(keypoints) > KeypointIndex.RIGHT_ANKLE else None
        
        if left_shoulder and left_ankle:
            shoulder_to_ankle = self._euclidean_distance(left_shoulder[:2], left_ankle[:2])
            # Approximate total height (shoulder to ankle is ~75% of total height)
            return shoulder_to_ankle / 0.75
        
        if right_shoulder and right_ankle:
            shoulder_to_ankle = self._euclidean_distance(right_shoulder[:2], right_ankle[:2])
            return shoulder_to_ankle / 0.75
        
        return None
    
    def _get_top_point(self, keypoints):
        """Get the topmost point of the body (head)"""
        candidates = []
        
        # Check face points
        face_indices = [
            KeypointIndex.NOSE,
            KeypointIndex.LEFT_EYE,
            KeypointIndex.RIGHT_EYE,
            KeypointIndex.LEFT_EAR,
            KeypointIndex.RIGHT_EAR
        ]
        
        for idx in face_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                if keypoints[idx][2] > 0.3:  # Confidence threshold
                    candidates.append(keypoints[idx])
        
        if candidates:
            # Return the highest point (minimum y value)
            return min(candidates, key=lambda p: p[1])
        
        return None
    
    def _get_bottom_point(self, keypoints):
        """Get the bottommost point of the body (feet)"""
        candidates = []
        
        # Check foot/ankle points
        foot_indices = [
            KeypointIndex.LEFT_ANKLE,
            KeypointIndex.RIGHT_ANKLE,
            KeypointIndex.LEFT_HEEL,
            KeypointIndex.RIGHT_HEEL,
            KeypointIndex.LEFT_FOOT_INDEX,
            KeypointIndex.RIGHT_FOOT_INDEX
        ]
        
        for idx in foot_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                if keypoints[idx][2] > 0.3:  # Confidence threshold
                    candidates.append(keypoints[idx])
        
        if candidates:
            # Return the lowest point (maximum y value)
            return max(candidates, key=lambda p: p[1])
        
        return None
    
    def _euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def pixels_to_cm(self, pixel_distance):
        """
        Convert pixel distance to centimeters
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            float: Distance in centimeters or None if not calibrated
        """
        if not self.is_calibrated:
            return None
        
        return pixel_distance * self.scale_factor
    
    def auto_calibrate_from_reference(self, keypoints, reference_measurement_cm, 
                                     reference_type='shoulder_width'):
        """
        Auto-calibrate using a known measurement (alternative calibration method)
        
        Args:
            keypoints: List of detected keypoints
            reference_measurement_cm: Known measurement in cm
            reference_type: Type of measurement ('shoulder_width', etc.)
            
        Returns:
            bool: True if calibration successful
        """
        if reference_type == 'shoulder_width':
            left_shoulder = keypoints[KeypointIndex.LEFT_SHOULDER]
            right_shoulder = keypoints[KeypointIndex.RIGHT_SHOULDER]
            
            if left_shoulder and right_shoulder:
                pixel_distance = self._euclidean_distance(
                    left_shoulder[:2], right_shoulder[:2]
                )
                self.scale_factor = reference_measurement_cm / pixel_distance
                self.is_calibrated = True
                return True
        
        return False
    
    def get_calibration_info(self):
        """Get calibration information"""
        return {
            'is_calibrated': self.is_calibrated,
            'scale_factor': self.scale_factor,
            'user_height_cm': self.user_height_cm
        }