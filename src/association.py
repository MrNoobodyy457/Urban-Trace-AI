"""
PHASE 2: CROSS-CAMERA ASSOCIATION - PLACE IN: src/4_association.py
Match same vehicle across multiple cameras
"""

from datetime import datetime, timedelta
from difflib import SequenceMatcher
import numpy as np


class CrossCameraAssociator:
    """Match vehicles across camera network"""
    
    def __init__(self, camera_graph):
        """
        Args:
            camera_graph: Dict describing camera connections
            Example: {
                'cam_north_1': {'neighbors': ['cam_north_2'], 'travel_time': 8},
                'cam_north_2': {'neighbors': ['cam_north_1', 'cam_center'], 'travel_time': 6},
            }
        """
        self.camera_graph = camera_graph
        self.observation_history = {}  # plate -> list of observations
        self.vehicle_global_map = {}  # (camera, local_id) -> global_id
        
        print("✓ Cross-camera associator initialized")
    
    def add_observation(self, local_track_id, camera_id, plate, timestamp, 
                       direction, confidence, bbox=None):
        """
        Register a vehicle observation
        
        Args:
            local_track_id: Track ID within single camera
            camera_id: Camera identifier
            plate: License plate string
            timestamp: datetime object
            direction: Direction (e.g., 'north', 'south')
            confidence: Plate confidence (0-1)
            bbox: [x1, y1, x2, y2]
        """
        if plate is None:
            return  # Skip observations without plate
        
        # Clean plate
        plate_clean = plate.upper().replace(' ', '')
        
        if plate_clean not in self.observation_history:
            self.observation_history[plate_clean] = []
        
        observation = {
            'camera': camera_id,
            'local_id': local_track_id,
            'timestamp': timestamp,
            'direction': direction,
            'confidence': confidence,
            'bbox': bbox
        }
        
        self.observation_history[plate_clean].append(observation)
    
    def find_trajectory(self, plate):
        """
        Reconstruct vehicle path across cameras
        
        Args:
            plate: License plate
        
        Returns:
            List of cameras in order: ['cam_A', 'cam_C', 'cam_F', ...]
        """
        plate_clean = plate.upper().replace(' ', '')
        
        if plate_clean not in self.observation_history:
            return []
        
        obs = sorted(self.observation_history[plate_clean], 
                     key=lambda x: x['timestamp'])
        
        trajectory = []
        
        for i, ob in enumerate(obs):
            current_cam = ob['camera']
            
            if i == 0:
                # First observation
                trajectory.append(current_cam)
            else:
                prev_cam = trajectory[-1]
                
                # Check if plausible path
                if self._is_plausible_transition(
                    prev_cam, current_cam, 
                    obs[i-1]['timestamp'], ob['timestamp']
                ):
                    trajectory.append(current_cam)
        
        return trajectory
    
    def _is_plausible_transition(self, from_cam, to_cam, t1, t2):
        """
        Check if transition between cameras is plausible
        
        Args:
            from_cam, to_cam: Camera IDs
            t1, t2: Timestamps
        
        Returns:
            True if transition is possible
        """
        time_diff = (t2 - t1).total_seconds()
        
        # Must be positive and within reasonable window
        if time_diff <= 0 or time_diff > 600:  # Max 10 mins
            return False
        
        # Check if cameras are neighbors
        if from_cam not in self.camera_graph:
            return False
        
        neighbors = self.camera_graph[from_cam].get('neighbors', [])
        if to_cam not in neighbors:
            return False
        
        # Check travel time constraint
        expected_travel = self.camera_graph[from_cam].get('travel_time', 60)
        max_travel = expected_travel * 3  # Allow 3x buffer
        
        if time_diff > max_travel:
            return False
        
        return True
    
    def plate_similarity(self, plate1, plate2):
        """
        Fuzzy match plates (handle OCR errors)
        
        Args:
            plate1, plate2: License plates
        
        Returns:
            Similarity score (0-1)
        """
        p1 = plate1.upper().replace(' ', '')
        p2 = plate2.upper().replace(' ', '')
        
        # Common OCR confusions
        p1_norm = self._normalize_plate(p1)
        p2_norm = self._normalize_plate(p2)
        
        # Use sequence matching
        ratio = SequenceMatcher(None, p1_norm, p2_norm).ratio()
        return ratio
    
    def _normalize_plate(self, plate):
        """
        Normalize plate for comparison
        Handles: 0↔O, 1↔I, 8↔B, 5↔S
        """
        replacements = {
            '0': 'O',
            '1': 'I',
            '8': 'B',
            '5': 'S'
        }
        normalized = plate
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized
    
    def get_all_trajectories(self):
        """Get trajectories for all observed plates"""
        trajectories = {}
        
        for plate in self.observation_history:
            traj = self.find_trajectory(plate)
            if len(traj) > 1:  # Only multi-camera trajectories
                trajectories[plate] = traj
        
        return trajectories
    
    def get_observations_for_plate(self, plate):
        """Get all observations for a specific plate"""
        plate_clean = plate.upper().replace(' ', '')
        return self.observation_history.get(plate_clean, [])


# ============ TESTING ============
if __name__ == "__main__":
    print("\n=== Cross-Camera Association Test ===\n")
    
    camera_graph = {
        'cam_north_1': {'neighbors': ['cam_north_2'], 'travel_time': 8},
        'cam_north_2': {'neighbors': ['cam_north_1', 'cam_center'], 'travel_time': 6},
        'cam_center': {'neighbors': ['cam_north_2', 'cam_south_1'], 'travel_time': 10},
    }
    
    assoc = CrossCameraAssociator(camera_graph)
    
    # Simulate observations
    now = datetime.now()
    
    assoc.add_observation(
        local_track_id=1,
        camera_id='cam_north_1',
        plate='MH02AB1234',
        timestamp=now,
        direction='south',
        confidence=0.9
    )
    
    assoc.add_observation(
        local_track_id=2,
        camera_id='cam_north_2',
        plate='MH02AB1234',
        timestamp=now + timedelta(seconds=7),  # 7 seconds later
        direction='south',
        confidence=0.88
    )
    
    assoc.add_observation(
        local_track_id=3,
        camera_id='cam_center',
        plate='MH02AB1234',
        timestamp=now + timedelta(seconds=13),  # 13 seconds later
        direction='south',
        confidence=0.87
    )
    
    trajectory = assoc.find_trajectory('MH02AB1234')
    print(f"Trajectory for MH02AB1234: {' → '.join(trajectory)}")
    
    print("\n✅ Association module ready!")
