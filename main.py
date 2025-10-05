"""
Main Application - Body Measurement Assistant
Real-time body measurement using pose detection
"""

import cv2
import numpy as np
import argparse
import time
import os
import pandas as pd
from datetime import datetime

from config import *
from pose_detector import PoseDetector
from calibration import Calibrator
from measurement_calculator import MeasurementCalculator
from visualization import Visualizer

class BodyMeasurementApp:
    def __init__(self, camera_index=CAMERA_INDEX, video_path=None):
        """
        Initialize the Body Measurement Application
        
        Args:
            camera_index: Index of camera device
            video_path: Path to video file (if not using camera)
        """
        # Initialize components
        self.pose_detector = PoseDetector(use_mediapipe=True, use_movenet=False)
        self.calibrator = Calibrator()
        self.measurement_calculator = MeasurementCalculator(self.calibrator)
        self.visualizer = Visualizer()
        
        # Video capture
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # State variables
        self.is_running = False
        self.measurements_data = []
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    
    def calibrate_system(self, frame, user_height_cm):
        """
        Calibrate the measurement system
        
        Args:
            frame: Current frame
            user_height_cm: User's actual height in cm
            
        Returns:
            bool: Success status
        """
        print("\n=== CALIBRATION MODE ===")
        print("Please stand upright with full body visible in frame...")
        
        # Detect pose
        keypoints = self.pose_detector.detect(frame)
        
        if keypoints is None:
            print("Error: No pose detected. Please ensure full body is visible.")
            return False
        
        # Check if pose is valid
        if not self.pose_detector.is_valid_pose(keypoints):
            print("Error: Invalid pose detected. Please stand upright.")
            return False
        
        # Calibrate
        success = self.calibrator.calibrate(keypoints, user_height_cm)
        
        if success:
            print("✓ Calibration successful!")
            # Visualize calibration
            cal_frame = frame.copy()
            cal_frame = self.visualizer.draw_skeleton(cal_frame, keypoints)
            cal_frame = self.visualizer.draw_keypoints(cal_frame, keypoints)
            cv2.imshow('Calibration - Press any key', cal_frame)
            cv2.waitKey(2000)
        
        return success
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            processed_frame: Frame with visualizations
            measurements: Dict of measurements
        """
        # Detect pose
        keypoints = self.pose_detector.detect(frame)
        
        # Initialize processed frame
        processed_frame = frame.copy()
        measurements = None
        
        if keypoints and self.pose_detector.is_valid_pose(keypoints):
            # Draw skeleton and keypoints
            processed_frame = self.visualizer.draw_skeleton(processed_frame, keypoints)
            processed_frame = self.visualizer.draw_keypoints(processed_frame, keypoints)
            
            # Calculate measurements if calibrated
            if self.calibrator.is_calibrated:
                measurements = self.measurement_calculator.get_averaged_measurements(keypoints)
                processed_frame = self.visualizer.draw_measurements(
                    processed_frame, keypoints, measurements
                )
        
        # Draw calibration status
        processed_frame = self.visualizer.draw_calibration_status(processed_frame, self.calibrator)
        
        # Draw FPS
        processed_frame = self.visualizer.draw_fps(processed_frame, self.fps)
        
        # Draw instructions
        instructions = [
            "Press 'c' - Calibrate",
            "Press 's' - Save measurements",
            "Press 'p' - Take snapshot",
            "Press 'q' - Quit"
        ]
        processed_frame = self.visualizer.draw_instructions(processed_frame, instructions)
        
        return processed_frame, measurements
    
    def save_measurements(self, measurements):
        """
        Save measurements to CSV
        
        Args:
            measurements: Dict of measurements
        """
        if measurements is None:
            print("No measurements to save")
            return
        
        # Prepare data
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_height_cm': self.calibrator.user_height_cm
        }
        data.update(measurements)
        
        # Append to list
        self.measurements_data.append(data)
        
        # Save to CSV
        df = pd.DataFrame(self.measurements_data)
        df.to_csv(MEASUREMENTS_CSV, index=False)
        print(f"✓ Measurements saved to {MEASUREMENTS_CSV}")
        
        # Print summary
        print(self.measurement_calculator.format_measurements(measurements))
    
    def take_snapshot(self, frame):
        """
        Save current frame as snapshot
        
        Args:
            frame: Current frame
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{SNAPSHOT_DIR}/snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Snapshot saved: {filename}")
    
    def run(self, user_height_cm=None):
        """
        Run the main application loop
        
        Args:
            user_height_cm: User's height in cm (for auto-calibration)
        """
        self.is_running = True
        
        print("\n" + "="*50)
        print("BODY MEASUREMENT ASSISTANT")
        print("="*50)
        print("\nInstructions:")
        print("1. Stand 2-3 meters away from camera")
        print("2. Ensure full body is visible in frame")
        print("3. Press 'c' to calibrate (required before measurements)")
        print("4. Press 's' to save measurements")
        print("5. Press 'p' to take snapshot")
        print("6. Press 'q' to quit")
        print("\nStarting camera...")
        
        # Auto-calibrate if height provided
        calibration_done = False
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = 30 / elapsed
                self.start_time = time.time()
            
            # Process frame
            processed_frame, measurements = self.process_frame(frame)
            
            # Display
            cv2.imshow('Body Measurement Assistant', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                self.is_running = False
                
            elif key == ord('c'):
                # Calibrate
                height = user_height_cm
                if height is None:
                    print("\nEnter your height in centimeters: ", end='')
                    try:
                        height = float(input())
                    except ValueError:
                        print("Invalid height. Using default 170 cm")
                        height = 170
                
                self.calibrate_system(frame, height)
                calibration_done = True
                
            elif key == ord('s'):
                # Save measurements
                if not self.calibrator.is_calibrated:
                    print("Please calibrate first (press 'c')")
                elif measurements:
                    self.save_measurements(measurements)
                else:
                    print("No valid measurements to save")
                    
            elif key == ord('p'):
                # Take snapshot
                self.take_snapshot(processed_frame)
            
            # Auto-calibrate on first valid frame if height provided
            if not calibration_done and user_height_cm is not None:
                keypoints = self.pose_detector.detect(frame)
                if keypoints and self.pose_detector.is_valid_pose(keypoints):
                    if self.calibrate_system(frame, user_height_cm):
                        calibration_done = True
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Release resources and cleanup"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.close()
        print("✓ Application closed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Body Measurement Assistant')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                       help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (optional)')
    parser.add_argument('--height', type=float, default=None,
                       help='Your height in cm (for auto-calibration)')
    
    args = parser.parse_args()
    
    # Create and run application
    app = BodyMeasurementApp(
        camera_index=args.camera,
        video_path=args.video
    )
    
    app.run(user_height_cm=args.height)

if __name__ == "__main__":
    main()