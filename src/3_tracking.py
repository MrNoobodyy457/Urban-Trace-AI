"""
PHASE 1: TRACKING - PLACE IN: src/3_tracking.py
Single-camera vehicle tracking using centroid-based matching
"""

import numpy as np
from collections import defaultdict


class SimpleTracker:
    """
    Lightweight centroid-based tracker for single camera
    Tracks vehicles within one camera before cross-camera association
    """
    
    def __init__(self, max_age=30, distance_threshold=50):
        """
        Args:
            max_age: Frames to keep track alive without detection
            distance_threshold: Pixels for centroid matching
        """
        self.tracks = {}  # track_id -> track_info
        self.next_id = 0
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        
        print(f"✓ Tracker initialized (max_age={max_age}, dist_threshold={distance_threshold})")
    
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of {
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'confidence': float,
                'plate': str or None,
                'plate_confidence': float
            }
        
        Returns:
            Dictionary of {track_id: detection_with_id}
        """
        matched_tracks = {}
        
        # Track centroids we've already matched
        matched_detection_indices = set()
        
        # Try to match each detection to existing track
        for det_idx, detection in enumerate(detections):
            cx, cy = detection['center']
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.tracks.items():
                if track_data['age'] > 0:  # Only match active tracks
                    tcx, tcy = track_data['center']
                    distance = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                    
                    # Match if within threshold and closest
                    if distance < self.distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
            
            # Update matched track or create new one
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['center'] = (cx, cy)
                self.tracks[best_track_id]['bbox'] = detection['bbox']
                self.tracks[best_track_id]['confidence'] = detection['confidence']
                self.tracks[best_track_id]['age'] += 1
                
                # Update plate if better
                if detection.get('plate'):
                    self.tracks[best_track_id]['plate'] = detection['plate']
                    self.tracks[best_track_id]['plate_confidence'] = detection.get('plate_confidence', 0)
                
                matched_tracks[best_track_id] = detection
                matched_detection_indices.add(det_idx)
            else:
                # New track
                self.next_id += 1
                new_track_id = self.next_id
                
                self.tracks[new_track_id] = {
                    'center': (cx, cy),
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'plate': detection.get('plate'),
                    'plate_confidence': detection.get('plate_confidence', 0),
                    'age': 1,
                    'frames_since_update': 0
                }
                
                matched_tracks[new_track_id] = detection
                matched_detection_indices.add(det_idx)
        
        # Age out old tracks
        dead_tracks = []
        for track_id, track_data in self.tracks.items():
            track_data['frames_since_update'] += 1
            
            if track_data['frames_since_update'] > self.max_age:
                dead_tracks.append(track_id)
        
        # Remove dead tracks
        for track_id in dead_tracks:
            del self.tracks[track_id]
        
        return matched_tracks
    
    def get_active_tracks(self):
        """Get all currently active tracks"""
        return {tid: tdata for tid, tdata in self.tracks.items() 
                if tdata['frames_since_update'] == 0}
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 0


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack (more advanced multi-object tracking)
    Placeholder for future implementation
    """
    
    def __init__(self):
        print("⚠️  ByteTrack not yet implemented. Use SimpleTracker for now.")
    
    def update(self, detections):
        raise NotImplementedError("Use SimpleTracker instead")


# ============ TESTING ============
if __name__ == "__main__":
    print("\n=== Tracking Module Test ===\n")
    
    tracker = SimpleTracker(max_age=30, distance_threshold=50)
    
    # Simulate some detections
    detections_frame1 = [
        {
            'bbox': [100, 100, 150, 180],
            'center': (125, 140),
            'confidence': 0.9,
            'plate': 'MH02AB1234',
            'plate_confidence': 0.85
        },
        {
            'bbox': [300, 200, 350, 280],
            'center': (325, 240),
            'confidence': 0.88,
            'plate': None,
            'plate_confidence': 0
        }
    ]
    
    tracks1 = tracker.update(detections_frame1)
    print(f"Frame 1: {len(tracks1)} tracks")
    for tid, det in tracks1.items():
        print(f"  Track {tid}: plate={det.get('plate')}")
    
    # Move vehicles slightly
    detections_frame2 = [
        {
            'bbox': [105, 105, 155, 185],
            'center': (130, 145),
            'confidence': 0.9,
            'plate': 'MH02AB1234',  # Same plate
            'plate_confidence': 0.86
        },
        {
            'bbox': [305, 210, 355, 290],
            'center': (330, 250),
            'confidence': 0.87,
            'plate': 'KA01XY5678',
            'plate_confidence': 0.80
        }
    ]
    
    tracks2 = tracker.update(detections_frame2)
    print(f"Frame 2: {len(tracks2)} tracks")
    for tid, det in tracks2.items():
        print(f"  Track {tid}: plate={det.get('plate')}")
    
    print("\n✅ Tracking module ready!")
