"""
Measurement Calculator Module
Computes body measurements from keypoints
"""

import numpy as np
from collections import deque
from config import MEASUREMENTS, SMOOTHING_WINDOW, KeypointIndex

class MeasurementCalculator:
    def __init__(self, calibrator):
        """
        Initialize measurement calculator
        
        Args:
            calibrator: Calibrator instance for pixel-to-cm conversion
        """
        self.calibrator = calibrator
        self.measurement_history = {key: deque(maxlen=SMOOTHING_WINDOW) 
                                   for key in MEASUREMENTS.keys()}
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: (x, y, confidence) tuple
            point2: (x, y, confidence) tuple
            
        Returns:
            float: Distance in pixels or None if points invalid
        """
        if point1 is None or point2 is None:
            return None
        
        if point1[2] < 0.3 or point2[2] < 0.3:  # Confidence check
            return None
        
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_path_length(self, points):
        """
        Calculate total path length through multiple points
        
        Args:
            points: List of (x, y, confidence) tuples
            
        Returns:
            float: Total path length in pixels or None if invalid
        """
        if not points or len(points) < 2:
            return None
        
        total_distance = 0
        for i in range(len(points) - 1):
            dist = self.calculate_distance(points[i], points[i + 1])
            if dist is None:
                return None
            total_distance += dist
        
        return total_distance
    
    def calculate_single_measurement(self, keypoints, measurement_name):
        """
        Calculate a single measurement
        
        Args:
            keypoints: List of detected keypoints
            measurement_name: Name of measurement from MEASUREMENTS dict
            
        Returns:
            float: Measurement in cm or None if cannot calculate
        """
        if measurement_name not in MEASUREMENTS:
            return None
        
        measurement_config = MEASUREMENTS[measurement_name]
        point_indices = measurement_config['points']
        
        # Get the actual keypoints
        points = []
        for idx in point_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                points.append(keypoints[idx])
            else:
                return None
        
        # Calculate pixel distance based on measurement type
        if measurement_config.get('type') == 'vertical':
            # For torso length: average of shoulder points to average of hip points
            if len(point_indices) == 4:
                shoulder_mid_x = (points[0][0] + points[1][0]) / 2
                shoulder_mid_y = (points[0][1] + points[1][1]) / 2
                hip_mid_x = (points[2][0] + points[3][0]) / 2
                hip_mid_y = (points[2][1] + points[3][1]) / 2
                
                pixel_distance = self.calculate_distance(
                    (shoulder_mid_x, shoulder_mid_y, 1.0),
                    (hip_mid_x, hip_mid_y, 1.0)
                )
        elif len(points) == 2:
            # Simple distance between two points
            pixel_distance = self.calculate_distance(points[0], points[1])
        else:
            # Path length through multiple points
            pixel_distance = self.calculate_path_length(points)
        
        if pixel_distance is None:
            return None
        
        # Apply multiplier if specified
        if 'multiplier' in measurement_config:
            pixel_distance *= measurement_config['multiplier']
        
        # Convert to cm
        measurement_cm = self.calibrator.pixels_to_cm(pixel_distance)
        
        return measurement_cm
    
    def calculate_all_measurements(self, keypoints):
        """
        Calculate all defined measurements
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            dict: Dictionary of measurements in cm
        """
        measurements = {}
        
        for name in MEASUREMENTS:
            value = self.calculate_single_measurement(keypoints, name)
            if value is not None:
                # Add to history for smoothing
                self.measurement_history[name].append(value)
                # Use smoothed value
                measurements[name] = np.mean(list(self.measurement_history[name]))
            else:
                measurements[name] = None
        
        return measurements
    
    def get_averaged_measurements(self, keypoints):
        """
        Get measurements with multi-frame averaging for stability
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            dict: Smoothed measurements
        """
        return self.calculate_all_measurements(keypoints)
    
    def calculate_body_proportions(self, measurements):
        """
        Calculate body proportions and ratios
        
        Args:
            measurements: Dict of measurements in cm
            
        Returns:
            dict: Body proportion information
        """
        proportions = {}
        
        # Shoulder to hip ratio
        if measurements.get('shoulder_width') and measurements.get('torso_length'):
            proportions['shoulder_hip_ratio'] = (
                measurements['shoulder_width'] / measurements['torso_length']
            )
        
        # Arm symmetry
        left_sleeve = measurements.get('left_sleeve_length')
        right_sleeve = measurements.get('right_sleeve_length')
        if left_sleeve and right_sleeve:
            proportions['arm_symmetry'] = min(left_sleeve, right_sleeve) / max(left_sleeve, right_sleeve)
            proportions['arm_difference_cm'] = abs(left_sleeve - right_sleeve)
        
        # Leg symmetry
        left_pant = measurements.get('left_pant_length')
        right_pant = measurements.get('right_pant_length')
        if left_pant and right_pant:
            proportions['leg_symmetry'] = min(left_pant, right_pant) / max(left_pant, right_pant)
            proportions['leg_difference_cm'] = abs(left_pant - right_pant)
        
        return proportions
    
    def validate_measurements(self, measurements):
        """
        Validate measurements for realistic ranges
        
        Args:
            measurements: Dict of measurements
            
        Returns:
            dict: Validation results with warnings
        """
        validation = {'is_valid': True, 'warnings': []}
        
        # Define realistic ranges (in cm)
        ranges = {
            'shoulder_width': (30, 60),
            'left_sleeve_length': (50, 90),
            'right_sleeve_length': (50, 90),
            'left_pant_length': (70, 130),
            'right_pant_length': (70, 130),
            'torso_length': (40, 80)
        }
        
        for name, (min_val, max_val) in ranges.items():
            if name in measurements and measurements[name] is not None:
                value = measurements[name]
                if value < min_val or value > max_val:
                    validation['warnings'].append(
                        f"{MEASUREMENTS[name]['name']}: {value:.1f}cm seems unusual "
                        f"(expected {min_val}-{max_val}cm)"
                    )
        
        if validation['warnings']:
            validation['is_valid'] = False
        
        return validation
    
    def format_measurements(self, measurements):
        """
        Format measurements for display
        
        Args:
            measurements: Dict of measurements
            
        Returns:
            str: Formatted string
        """
        lines = ["\n=== BODY MEASUREMENTS ==="]
        
        for key, value in measurements.items():
            if value is not None:
                display_name = MEASUREMENTS[key]['name']
                lines.append(f"{display_name}: {value:.1f} cm")
        
        return "\n".join(lines)