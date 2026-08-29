"""
ANPR Configuration Presets for Different Scenarios

Use these configurations to quickly adapt your ANPR system to different 
traffic scenarios, lighting conditions, and vehicle types.
"""

class ANPRConfig:
    """Base configuration class"""
    
    def __init__(self):
        # Detection parameters
        self.yolo_conf_threshold = 0.25
        self.yolo_iou_threshold = 0.45
        self.nms_iou_thresh = 0.35
        
        # Plate dimensions
        self.min_plate_aspect_ratio = 2.0
        self.max_plate_aspect_ratio = 6.5
        self.min_plate_area = 500
        self.max_plate_area = 80000
        
        # OCR/Text processing
        self.min_ocr_confidence = 0.45
        self.ocr_early_exit_conf = 0.65
        self.ocr_strategies = ['default', 'high_contrast', 'low_light', 'glare']
        
        # Tracking
        self.tracker_iou_threshold = 0.3
        self.tracker_max_age = 15
        
        # HSV ranges for plate detection
        self.hsv_yellow_lower = (12, 40, 50)
        self.hsv_yellow_upper = (35, 255, 255)
        self.hsv_white_lower = (0, 0, 140)
        self.hsv_white_upper = (180, 50, 255)
        
        self.name = "Default"
        self.description = "Balanced configuration for general use"


class HighwayConfig(ANPRConfig):
    """
    Configuration for highway/motorway scenarios
    - Vehicles at higher speeds
    - Generally better lighting
    - Plates at consistent angles
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.30  # Stricter - at highway speeds
        self.yolo_iou_threshold = 0.50
        self.min_ocr_confidence = 0.50   # Higher quality required
        self.ocr_strategies = ['default', 'glare']  # Fast processing
        self.tracker_max_age = 10  # Shorter persistence (faster moving)
        self.name = "Highway"
        self.description = "High-speed vehicles, good lighting, consistent angles"


class UrbanTrafficConfig(ANPRConfig):
    """
    Configuration for urban traffic scenarios
    - Mixed vehicle speeds
    - Variable lighting (buildings, trees)
    - Complex angles (intersections)
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.22  # Slightly relaxed
        self.min_ocr_confidence = 0.42
        self.tracker_max_age = 20  # Longer persistence (stop-and-go)
        self.hsv_yellow_lower = (10, 35, 45)  # Slightly broader
        self.hsv_yellow_upper = (38, 255, 255)
        self.name = "Urban Traffic"
        self.description = "Mixed speeds, variable lighting, complex angles"


class NightConfig(ANPRConfig):
    """
    Configuration for night-time/low-light scenarios
    - Headlights and streetlights
    - Lower contrast plates
    - More motion blur
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.20  # More relaxed
        self.min_ocr_confidence = 0.38   # Accept lower confidence
        self.ocr_early_exit_conf = 0.55
        self.ocr_strategies = ['low_light', 'high_contrast', 'default', 'glare']
        self.tracker_max_age = 25  # Much longer persistence (headlights trail)
        self.hsv_yellow_lower = (10, 25, 30)  # Broader for dimmer yellows
        self.hsv_yellow_upper = (40, 255, 255)
        self.name = "Night Mode"
        self.description = "Low-light conditions, headlights, reduced contrast"


class ParkedVehiclesConfig(ANPRConfig):
    """
    Configuration for detecting parked/stationary vehicles
    - Usually better lighting (daytime)
    - More variation in angles
    - Clearer plate images (no motion blur)
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.28  # Stricter (stationary = clear detection)
        self.min_ocr_confidence = 0.55   # High confidence for static plates
        self.min_plate_area = 800        # Larger plates (clearer from distance)
        self.ocr_strategies = ['high_contrast', 'default']  # Quality over speed
        self.tracker_max_age = 5  # Very short (no motion expected)
        self.name = "Parked Vehicles"
        self.description = "Stationary vehicles, varied angles, good clarity"


class AutoRickshawConfig(ANPRConfig):
    """
    Configuration specialized for auto-rickshaws (tuk-tuks)
    - Yellow plates standard
    - Often in congested traffic
    - Side-view angles common
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.23
        self.min_plate_aspect_ratio = 2.5  # Autos have specific ratio
        self.max_plate_aspect_ratio = 5.5
        self.hsv_yellow_lower = (12, 45, 55)  # Very tuned for yellow
        self.hsv_yellow_upper = (34, 255, 255)
        self.min_ocr_confidence = 0.40
        self.tracker_max_age = 18
        self.name = "Auto-Rickshaws"
        self.description = "Yellow plates, urban congestion, side angles"


class CommercialVehicleConfig(ANPRConfig):
    """
    Configuration for commercial vehicles (trucks, buses)
    - Larger bounding boxes
    - Sometimes reflective or worn plates
    - Higher mounted plates
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.25
        self.min_plate_area = 1000         # Larger vehicles = larger plates
        self.max_plate_area = 150000
        self.min_plate_aspect_ratio = 2.8
        self.max_plate_aspect_ratio = 7.0  # Can be wider
        self.min_ocr_confidence = 0.40
        self.tracker_max_age = 12
        self.name = "Commercial Vehicles"
        self.description = "Large vehicles, worn/reflective plates, high mounting"


class BrightSunConfig(ANPRConfig):
    """
    Configuration for very bright/sunny conditions
    - Glare and reflections common
    - Overexposed regions
    - Harsh shadows
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.25
        self.min_ocr_confidence = 0.42
        self.ocr_strategies = ['glare', 'high_contrast', 'default', 'low_light']
        self.ocr_early_exit_conf = 0.70  # Need very clear reads
        self.hsv_white_lower = (0, 0, 160)  # Higher brightness threshold
        self.name = "Bright Sun"
        self.description = "Glare, reflections, harsh shadows, overexposed"


class CongestedTrafficConfig(ANPRConfig):
    """
    Configuration for heavy congestion/traffic jams
    - Many overlapping vehicles
    - Stationary/slow moving
    - Close proximity plates
    """
    def __init__(self):
        super().__init__()
        self.yolo_conf_threshold = 0.26  # Slightly stricter (overlaps)
        self.nms_iou_thresh = 0.30       # Aggressive NMS for overlap
        self.min_ocr_confidence = 0.48
        self.tracker_max_age = 30        # Very long (stationary)
        self.min_plate_area = 700
        self.name = "Heavy Congestion"
        self.description = "Many vehicles, overlapping plates, slow motion"


# Configuration preset dictionary
CONFIG_PRESETS = {
    'default': ANPRConfig,
    'highway': HighwayConfig,
    'urban': UrbanTrafficConfig,
    'night': NightConfig,
    'parked': ParkedVehiclesConfig,
    'autorickshaw': AutoRickshawConfig,
    'commercial': CommercialVehicleConfig,
    'bright_sun': BrightSunConfig,
    'congestion': CongestedTrafficConfig,
}


def get_config(config_name: str = 'default') -> ANPRConfig:
    """Get a configuration preset by name"""
    config_name = config_name.lower()
    if config_name not in CONFIG_PRESETS:
        print(f"Warning: Config '{config_name}' not found. Using 'default'")
        return ANPRConfig()
    return CONFIG_PRESETS[config_name]()


def print_all_configs():
    """Print all available configurations"""
    print("\n" + "="*60)
    print("ANPR CONFIGURATION PRESETS")
    print("="*60 + "\n")
    for name, config_class in CONFIG_PRESETS.items():
        config = config_class()
        print(f"• {name.upper()}: {config.description}")
        print(f"  - YOLO Confidence: {config.yolo_conf_threshold}")
        print(f"  - Min OCR Confidence: {config.min_ocr_confidence}")
        print(f"  - Tracker Max Age: {config.tracker_max_age}")
        print(f"  - OCR Strategies: {', '.join(config.ocr_strategies)}")
        print()


# Example usage
if __name__ == "__main__":
    print_all_configs()
    
    # Get specific config
    urban_config = get_config('urban')
    print(f"Selected config: {urban_config.name}")
    print(f"YOLO Threshold: {urban_config.yolo_conf_threshold}")