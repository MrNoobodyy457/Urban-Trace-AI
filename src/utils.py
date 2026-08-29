"""
UTILITIES - PLACE IN: src/utils.py
Helper functions for UrbanTrace AI
"""

import cv2
import numpy as np
from datetime import datetime


def draw_bbox_on_frame(frame, bbox, label='', color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on frame
    
    Args:
        frame: OpenCV image
        bbox: [x1, y1, x2, y2]
        label: Text label
        color: RGB color tuple
        thickness: Line thickness
    
    Returns:
        Frame with drawn bbox
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame


def get_bbox_center(bbox):
    """
    Get center coordinates of bounding box
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1, p2: (x, y) tuples
    
    Returns:
        Distance in pixels
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def format_time_delta(seconds):
    """
    Format seconds as readable duration
    
    Args:
        seconds: Time duration in seconds
    
    Returns:
        Formatted string: "1m 23s"
    """
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    
    if mins == 0:
        return f"{secs}s"
    else:
        return f"{mins}m {secs}s"


def get_current_timestamp():
    """Get current timestamp as ISO format string"""
    return datetime.now().isoformat()


def is_plate_valid(plate):
    """
    Check if plate format looks valid
    
    Args:
        plate: License plate string
    
    Returns:
        Boolean
    """
    import re
    
    plate_clean = plate.upper().replace(' ', '')
    
    # Indian plate: XX00XX0000 (2 letters, 2 digits, 2 letters, 4 digits)
    pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$')
    
    return bool(pattern.match(plate_clean))


def parse_bbox(det_dict):
    """
    Extract bbox from detection dictionary
    
    Args:
        det_dict: Detection dict with 'bbox' key
    
    Returns:
        [x1, y1, x2, y2]
    """
    if 'bbox' in det_dict:
        return det_dict['bbox']
    elif 'xyxy' in det_dict:
        return det_dict['xyxy']
    else:
        raise ValueError("No bbox found in detection")


def resize_frame(frame, width=None, height=None):
    """
    Resize frame maintaining aspect ratio
    
    Args:
        frame: OpenCV image
        width: Target width (if None, calc from height)
        height: Target height (if None, calc from width)
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if width is None:
        # Calculate width from height
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        # Calculate height from width
        ratio = width / w
        height = int(h * ratio)
    
    return cv2.resize(frame, (width, height))


def overlay_text(frame, text, position=(10, 30), font_scale=0.7, 
                 thickness=2, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """
    Draw text with background on frame
    
    Args:
        frame: OpenCV image
        text: Text to draw
        position: (x, y) starting position
        font_scale: Font size
        thickness: Font thickness
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
    
    Returns:
        Frame with text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                 (x - 5, y - text_height - 5),
                 (x + text_width + 5, y + baseline + 5),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    
    return frame


def save_frame(frame, filepath):
    """
    Save frame to disk
    
    Args:
        frame: OpenCV image
        filepath: Output file path
    
    Returns:
        Boolean success
    """
    try:
        cv2.imwrite(filepath, frame)
        return True
    except Exception as e:
        print(f"❌ Error saving frame: {e}")
        return False


# ============ TESTING ============
if __name__ == "__main__":
    print("\n=== Utils Module Test ===\n")
    
    # Test distance
    dist = calculate_distance((0, 0), (3, 4))
    print(f"Distance (0,0) → (3,4): {dist:.2f}")
    assert abs(dist - 5.0) < 0.1
    
    # Test time formatting
    duration = format_time_delta(83)
    print(f"83 seconds: {duration}")
    assert duration == "1m 23s"
    
    # Test plate validation
    valid = is_plate_valid("MH02AB1234")
    print(f"Plate 'MH02AB1234' valid: {valid}")
    assert valid == True
    
    invalid = is_plate_valid("INVALID")
    print(f"Plate 'INVALID' valid: {invalid}")
    assert invalid == False
    
    print("\n✅ Utils module ready!")
