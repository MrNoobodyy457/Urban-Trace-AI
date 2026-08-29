"""
PHASE 3: TRAJECTORY BUILDING - PLACE IN: src/5_trajectory.py
Build complete vehicle journeys from observations
"""

from datetime import datetime
from sqlalchemy.orm import Session


class TrajectoryEngine:
    """Build and manage vehicle trajectories"""
    
    def __init__(self, db_session):
        """
        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
    
    def build_trajectory(self, plate, observations):
        """
        Build complete trajectory from observations
        
        Args:
            plate: License plate
            observations: List of observation dicts from associator
        
        Returns:
            Trajectory dict
        """
        if not observations:
            return None
        
        # Sort by time
        sorted_obs = sorted(observations, key=lambda x: x['timestamp'])
        
        start_time = sorted_obs[0]['timestamp']
        end_time = sorted_obs[-1]['timestamp']
        duration = (end_time - start_time).total_seconds()
        
        path = [o['camera'] for o in sorted_obs]
        
        trajectory = {
            'plate': plate,
            'path': path,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'num_cameras': len(set(path)),
            'observations': [
                {
                    'camera': o['camera'],
                    'timestamp': o['timestamp'].isoformat(),
                    'direction': o['direction'],
                    'confidence': o['confidence']
                }
                for o in sorted_obs
            ]
        }
        
        return trajectory
    
    def get_segment_travel_time(self, trajectory, camera_from, camera_to):
        """
        Extract travel time between two cameras
        
        Args:
            trajectory: Trajectory dict
            camera_from, camera_to: Camera IDs
        
        Returns:
            Travel time in seconds or None
        """
        obs = trajectory['observations']
        
        from_time = None
        to_time = None
        
        for o in obs:
            if o['camera'] == camera_from and from_time is None:
                from_time = datetime.fromisoformat(o['timestamp'])
            elif o['camera'] == camera_to and from_time is not None:
                to_time = datetime.fromisoformat(o['timestamp'])
                break
        
        if from_time and to_time:
            return (to_time - from_time).total_seconds()
        
        return None
    
    def get_trajectory_statistics(self, trajectory):
        """
        Calculate statistics from trajectory
        
        Args:
            trajectory: Trajectory dict
        
        Returns:
            Stats dict
        """
        obs = trajectory['observations']
        
        # Calculate inter-camera times
        inter_times = []
        for i in range(1, len(obs)):
            t1 = datetime.fromisoformat(obs[i-1]['timestamp'])
            t2 = datetime.fromisoformat(obs[i]['timestamp'])
            inter_times.append((t2 - t1).total_seconds())
        
        stats = {
            'total_duration': trajectory['duration_seconds'],
            'num_observations': len(obs),
            'num_unique_cameras': len(set(o['camera'] for o in obs)),
            'avg_inter_camera_time': sum(inter_times) / len(inter_times) if inter_times else 0,
            'min_inter_camera_time': min(inter_times) if inter_times else 0,
            'max_inter_camera_time': max(inter_times) if inter_times else 0,
            'path': trajectory['path']
        }
        
        return stats


# ============ TESTING ============
if __name__ == "__main__":
    print("\n=== Trajectory Module Test ===\n")
    
    engine = TrajectoryEngine(None)  # No DB needed for test
    
    # Simulate observations
    from datetime import datetime, timedelta
    
    now = datetime.now()
    observations = [
        {
            'camera': 'cam_north_1',
            'timestamp': now,
            'direction': 'south',
            'confidence': 0.9
        },
        {
            'camera': 'cam_north_2',
            'timestamp': now + timedelta(seconds=8),
            'direction': 'south',
            'confidence': 0.88
        },
        {
            'camera': 'cam_center',
            'timestamp': now + timedelta(seconds=14),
            'direction': 'south',
            'confidence': 0.87
        },
    ]
    
    # Build trajectory
    traj = engine.build_trajectory('MH02AB1234', observations)
    
    print(f"Trajectory:")
    print(f"  Plate: {traj['plate']}")
    print(f"  Path: {' → '.join(traj['path'])}")
    print(f"  Duration: {traj['duration_seconds']:.1f} seconds")
    print(f"  Cameras: {traj['num_cameras']}")
    
    # Get stats
    stats = engine.get_trajectory_statistics(traj)
    print(f"\nStats:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n✅ Trajectory module ready!")
