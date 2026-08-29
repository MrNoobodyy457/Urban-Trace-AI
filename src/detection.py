"""
PHASE 1: Vehicle Detection using YOLOv8
Detects vehicles in CCTV frames
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class VehicleDetector:
    """Detect vehicles in frames using YOLOv8"""
    
    def __init__(self, model_size='m', confidence=0.5):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
            confidence: Detection confidence threshold (0-1)
        """
        print(f"Loading YOLOv8-{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.confidence = confidence
        
        # COCO vehicle classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        print("✓ Model loaded successfully")
    
    def detect_frame(self, frame):
        """
        Detect vehicles in a single frame
        
        Args:
            frame: OpenCV image (BGR)
        
        Returns:
            List of detections: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': float,
                    'class_name': str
                },
                ...
            ]
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                
                # Only keep vehicle classes
                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Calculate center
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': (cx, cy),
                        'confidence': conf,
                        'class_name': self.vehicle_classes[class_id]
                    })
        
        return detections
    
    def visualize_detections(self, frame, detections, show=True):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: OpenCV image
            detections: Output from detect_frame()
            show: Display the image
        
        Returns:
            Frame with drawn detections
        """
        frame_vis = frame.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bbox
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame_vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if show:
            cv2.imshow('Detections', frame_vis)
        
        return frame_vis
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process entire video and optionally save output
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            max_frames: Limit frames to process (for testing)
        
        Returns:
            List of detections per frame
        """
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"📝 Saving to: {output_path}")
        
        all_detections = []
        frame_idx = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Detect
            detections = self.detect_frame(frame)
            all_detections.append({
                'frame': frame_idx,
                'detections': detections
            })
            
            # Visualize
            frame_vis = self.visualize_detections(frame, detections, show=False)
            
            # Save if requested
            if output_path:
                out.write(frame_vis)
            
            # Progress
            if (frame_idx + 1) % 30 == 0:
                print(f"  ✓ Processed {frame_idx + 1}/{total_frames} frames ({len(detections)} vehicles)")
            
            # Limit frames if specified
            if max_frames and frame_idx >= max_frames - 1:
                break
            
            frame_idx += 1
        
        video.release()
        if output_path:
            out.release()
            print(f"✅ Video saved: {output_path}")
        
        print(f"✅ Detection complete: {len(all_detections)} frames processed")
        return all_detections


# ============ TESTING ============
if __name__ == "__main__":
    # Test on a single frame (if you have test data)
    print("\n=== Vehicle Detection Test ===\n")
    
    detector = VehicleDetector(model_size='m', confidence=0.5)
    
    # Create a dummy frame for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Running detection on dummy frame...")
    detections = detector.detect_frame(dummy_frame)
    print(f"Detections: {len(detections)} vehicles found")
    
    print("\n✅ Detection module working!")
    print("\n📌 Next: Add a real test video to data/test_videos/ and run:")
    print("   detector.process_video('data/test_videos/your_video.mp4', 'data/sample_output/output.mp4')")
