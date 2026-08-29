"""
PHASE 3: ANALYTICS - PLACE IN: src/6_analytics.py
Generate traffic statistics and anomaly detection
"""

from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class TrafficAnalytics:
    """Analyze traffic patterns from vehicle trajectories"""
    
    def __init__(self):
        self.trajectories = []  # Store processed trajectories
        print("✓ Analytics engine initialized")
    
    def add_trajectory(self, trajectory):
        """Add a trajectory to analysis"""
        self.trajectories.append(trajectory)
    
    def congestion_by_segment(self, camera_from, camera_to, time_window_min=5):
        """
        Analyze traffic on a specific segment
        
        Args:
            camera_from, camera_to: Camera IDs
            time_window_min: Look at last N minutes
        
        Returns:
            Congestion stats dict
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_min)
        
        travel_times = []
        vehicle_count = 0
        
        for traj in self.trajectories:
            # Check if this trajectory uses the segment
            obs = traj['observations']
            
            from_obs = None
            to_obs = None
            
            for o in obs:
                if o['camera'] == camera_from:
                    from_obs = o
                elif o['camera'] == camera_to and from_obs is not None:
                    to_obs = o
                    break
            
            if from_obs and to_obs:
                from_time = datetime.fromisoformat(from_obs['timestamp'])
                to_time = datetime.fromisoformat(to_obs['timestamp'])
                
                # Only count recent observations
                if to_time > cutoff_time:
                    travel_time = (to_time - from_time).total_seconds()
                    travel_times.append(travel_time)
                    vehicle_count += 1
        
        # Calculate statistics
        if travel_times:
            avg_travel = np.mean(travel_times)
            std_travel = np.std(travel_times)
        else:
            avg_travel = 0
            std_travel = 0
        
        # Determine congestion level
        if avg_travel < 10:
            level = 'light'
        elif avg_travel < 20:
            level = 'moderate'
        elif avg_travel < 40:
            level = 'heavy'
        else:
            level = 'severe'
        
        return {
            'segment': f'{camera_from} → {camera_to}',
            'avg_travel_time': avg_travel,
            'std_travel_time': std_travel,
            'vehicle_count': vehicle_count,
            'congestion_level': level,
            'time_window_min': time_window_min
        }
    
    def detect_anomalies(self):
        """
        Detect unusual traffic patterns
        
        Returns:
            List of anomaly dicts
        """
        anomalies = []
        
        # Check for direction reversals
        for traj in self.trajectories:
            obs = traj['observations']
            
            # If vehicle appears in same camera twice
            cameras = [o['camera'] for o in obs]
            camera_counts = defaultdict(int)
            
            for cam in cameras:
                camera_counts[cam] += 1
            
            for cam, count in camera_counts.items():
                if count > 1:
                    anomalies.append({
                        'type': 'camera_revisit',
                        'plate': traj['plate'],
                        'camera': cam,
                        'count': count,
                        'severity': 'medium'
                    })
            
            # Check for unusually fast travel (teleportation)
            for i in range(1, len(obs)):
                t1 = datetime.fromisoformat(obs[i-1]['timestamp'])
                t2 = datetime.fromisoformat(obs[i]['timestamp'])
                time_diff = (t2 - t1).total_seconds()
                
                if time_diff < 1:  # Less than 1 second between cameras
                    anomalies.append({
                        'type': 'impossibly_fast_travel',
                        'plate': traj['plate'],
                        'from': obs[i-1]['camera'],
                        'to': obs[i]['camera'],
                        'time_diff': time_diff,
                        'severity': 'high'
                    })
        
        return anomalies
    
    def speed_analysis(self, camera_graph):
        """
        Estimate vehicle speeds between cameras
        
        Args:
            camera_graph: Dict with camera locations and distances
            Example: {
                'cam_a': {'lat': 19.0, 'lon': 72.0, 'neighbors': {...}},
                'cam_b': {'lat': 19.01, 'lon': 72.01, 'neighbors': {...}},
            }
        
        Returns:
            Speed statistics
        """
        # This requires camera coordinates (GPS)
        # Placeholder for now
        return {
            'note': 'Speed analysis requires camera GPS coordinates'
        }
    
    def peak_hours(self, num_bins=24):
        """
        Analyze traffic by time of day
        
        Args:
            num_bins: Number of time bins (24 for hourly)
        
        Returns:
            Traffic per hour dict
        """
        traffic_per_hour = defaultdict(int)
        
        for traj in self.trajectories:
            start_time = datetime.fromisoformat(traj['observations'][0]['timestamp'])
            hour = start_time.hour
            traffic_per_hour[hour] += 1
        
        # Format as list
        hours = [traffic_per_hour.get(h, 0) for h in range(num_bins)]
        
        return {
            'traffic_by_hour': hours,
            'peak_hour': hours.index(max(hours)) if hours else None,
            'peak_traffic': max(hours) if hours else 0
        }
    
    def generate_report(self, camera_graph, time_window_min=30):
        """
        Generate comprehensive traffic report
        
        Args:
            camera_graph: Camera connectivity dict
            time_window_min: Report time window
        
        Returns:
            Report dict
        """
        segments = []
        for cam_from in camera_graph:
            for cam_to in camera_graph[cam_from].get('neighbors', []):
                stats = self.congestion_by_segment(cam_from, cam_to, time_window_min)
                segments.append(stats)
        
        anomalies = self.detect_anomalies()
        peak_data = self.peak_hours()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_window_min': time_window_min,
            'total_trajectories': len(self.trajectories),
            'segments': segments,
            'anomalies': anomalies,
            'peak_hours': peak_data
        }
        
        return report


# ============ TESTING ============
if __name__ == "__main__":
    print("\n=== Analytics Module Test ===\n")
    
    analytics = TrafficAnalytics()
    
    # Add sample trajectories
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    traj1 = {
        'plate': 'MH02AB1234',
        'path': ['cam_north_1', 'cam_north_2', 'cam_center'],
        'start_time': now,
        'end_time': now + timedelta(seconds=15),
        'duration_seconds': 15,
        'observations': [
            {
                'camera': 'cam_north_1',
                'timestamp': now.isoformat(),
                'direction': 'south',
                'confidence': 0.9
            },
            {
                'camera': 'cam_north_2',
                'timestamp': (now + timedelta(seconds=8)).isoformat(),
                'direction': 'south',
                'confidence': 0.88
            },
            {
                'camera': 'cam_center',
                'timestamp': (now + timedelta(seconds=15)).isoformat(),
                'direction': 'south',
                'confidence': 0.87
            }
        ]
    }
    
    analytics.add_trajectory(traj1)
    
    # Test segment analysis
    stats = analytics.congestion_by_segment('cam_north_1', 'cam_north_2')
    print(f"Congestion: {stats['segment']}")
    print(f"  Avg travel time: {stats['avg_travel_time']:.1f}s")
    print(f"  Level: {stats['congestion_level']}")
    
    # Test anomalies
    anomalies = analytics.detect_anomalies()
    print(f"\nAnomalies: {len(anomalies)} found")
    
    print("\n✅ Analytics module ready!")
