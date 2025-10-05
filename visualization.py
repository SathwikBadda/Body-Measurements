"""
Visualization Module
Handles drawing skeleton, keypoints, and measurements on frames
"""

import cv2
import numpy as np
from config import *

class Visualizer:
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def draw_keypoints(self, frame, keypoints):
        """
        Draw keypoints on frame
        
        Args:
            frame: Input image
            keypoints: List of (x, y, confidence) tuples
            
        Returns:
            frame: Frame with keypoints drawn
        """
        if keypoints is None:
            return frame
        
        for i, kp in enumerate(keypoints):
            if kp is not None and kp[2] > MIN_CONFIDENCE_THRESHOLD:
                x, y = int(kp[0]), int(kp[1])
                # Color based on confidence
                confidence = kp[2]
                color = self._confidence_to_color(confidence)
                cv2.circle(frame, (x, y), KEYPOINT_RADIUS, color, -1)
                cv2.circle(frame, (x, y), KEYPOINT_RADIUS + 2, (255, 255, 255), 1)
        
        return frame
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw skeleton connections between keypoints
        
        Args:
            frame: Input image
            keypoints: List of keypoints
            
        Returns:
            frame: Frame with skeleton drawn
        """
        if keypoints is None:
            return frame
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx] is not None and keypoints[end_idx] is not None):
                
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                if start_point[2] > MIN_CONFIDENCE_THRESHOLD and end_point[2] > MIN_CONFIDENCE_THRESHOLD:
                    pt1 = (int(start_point[0]), int(start_point[1]))
                    pt2 = (int(end_point[0]), int(end_point[1]))
                    cv2.line(frame, pt1, pt2, SKELETON_COLOR, LINE_THICKNESS)
        
        return frame
    
    def draw_measurements(self, frame, keypoints, measurements):
        """
        Draw measurement lines and values on frame
        
        Args:
            frame: Input image
            keypoints: List of keypoints
            measurements: Dict of measurements in cm
            
        Returns:
            frame: Frame with measurements drawn
        """
        if keypoints is None or measurements is None:
            return frame
        
        y_offset = 30
        
        # Draw measurement lines
        for name, value in measurements.items():
            if value is not None and name in MEASUREMENTS:
                config = MEASUREMENTS[name]
                point_indices = config['points']
                
                # Draw line for 2-point measurements
                if len(point_indices) == 2:
                    idx1, idx2 = point_indices
                    if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                        keypoints[idx1] is not None and keypoints[idx2] is not None):
                        
                        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                        
                        # Draw measurement line
                        cv2.line(frame, pt1, pt2, MEASUREMENT_LINE_COLOR, LINE_THICKNESS)
                        
                        # Draw measurement value at midpoint
                        mid_x = (pt1[0] + pt2[0]) // 2
                        mid_y = (pt1[1] + pt2[1]) // 2
                        text = f"{value:.1f}cm"
                        cv2.putText(frame, text, (mid_x - 30, mid_y - 10),
                                  TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        
        # Draw measurement summary on left side
        for name, value in measurements.items():
            if value is not None and name in MEASUREMENTS:
                display_name = MEASUREMENTS[name]['name']
                text = f"{display_name}: {value:.1f} cm"
                cv2.putText(frame, text, (10, y_offset),
                          TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
                y_offset += 25
        
        return frame
    
    def draw_info_panel(self, frame, info_dict):
        """
        Draw information panel on frame
        
        Args:
            frame: Input image
            info_dict: Dictionary of information to display
            
        Returns:
            frame: Frame with info panel
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        panel_height = len(info_dict) * 30 + 40
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        y_offset = 30
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset),
                       TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            y_offset += 30
        
        return frame
    
    def draw_calibration_status(self, frame, calibrator):
        """
        Draw calibration status on frame
        
        Args:
            frame: Input image
            calibrator: Calibrator instance
            
        Returns:
            frame: Frame with calibration status
        """
        status_color = (0, 255, 0) if calibrator.is_calibrated else (0, 0, 255)
        status_text = "CALIBRATED" if calibrator.is_calibrated else "NOT CALIBRATED"
        
        cv2.putText(frame, status_text, (frame.shape[1] - 200, 30),
                   TEXT_FONT, TEXT_SCALE, status_color, TEXT_THICKNESS)
        
        if calibrator.is_calibrated:
            info_text = f"Height: {calibrator.user_height_cm}cm"
            cv2.putText(frame, info_text, (frame.shape[1] - 200, 60),
                       TEXT_FONT, TEXT_SCALE - 0.1, TEXT_COLOR, 1)
        
        return frame
    
    def draw_instructions(self, frame, instructions):
        """
        Draw instructions on frame
        
        Args:
            frame: Input image
            instructions: List of instruction strings
            
        Returns:
            frame: Frame with instructions
        """
        y_offset = frame.shape[0] - 100
        
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       TEXT_FONT, TEXT_SCALE - 0.1, (255, 255, 0), 1)
            y_offset += 25
        
        return frame
    
    def draw_fps(self, frame, fps):
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input image
            fps: Frames per second value
            
        Returns:
            frame: Frame with FPS counter
        """
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (frame.shape[1] - 120, frame.shape[0] - 20),
                   TEXT_FONT, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)
        return frame
    
    def _confidence_to_color(self, confidence):
        """
        Convert confidence value to color
        
        Args:
            confidence: Confidence value (0-1)
            
        Returns:
            tuple: BGR color
        """
        # Red (low) to Green (high)
        if confidence > 0.7:
            return (0, 255, 0)  # Green
        elif confidence > 0.4:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def create_side_by_side_view(self, original, processed):
        """
        Create side-by-side view of original and processed frames
        
        Args:
            original: Original frame
            processed: Processed frame
            
        Returns:
            Combined frame
        """
        # Resize if needed
        h1, w1 = original.shape[:2]
        h2, w2 = processed.shape[:2]
        
        if h1 != h2:
            original = cv2.resize(original, (w2, h2))
        
        # Concatenate horizontally
        combined = np.hstack([original, processed])
        
        # Add labels
        cv2.putText(combined, "ORIGINAL", (10, 30),
                   TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        cv2.putText(combined, "PROCESSED", (w2 + 10, 30),
                   TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        
        return combined