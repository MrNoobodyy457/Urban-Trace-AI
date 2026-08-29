"""
Improved test_detection.py with configuration presets and better error handling

Usage:
    python test_detection_improved.py --config urban
    python test_detection_improved.py --config night
    python test_detection_improved.py --config default
"""

import cv2
import os
import torch
import argparse
from src.anpr import PlateReader
from src.anpr_config import get_config


def find_input_video(candidates=None):
    """Find input video with multiple fallback paths"""
    if candidates is None:
        candidates = [
            r"data\test_videos\sample.mp4",
            "sample.mp4",
            r"data\sample.mp4",
            "test_video.mp4",
            r"videos\test.mp4"
        ]
    
    for path in candidates:
        if os.path.exists(path):
            print(f"✓ Found video: {path}")
            return path
    
    print("✗ Error: Could not locate sample video")
    print(f"  Searched in: {candidates}")
    return None


def process_video(output_dir=r"data\sample_output", 
                 output_filename="output_annotated.mp4",
                 config_name='default',
                 max_frames=None,
                 debug=False):
    """
    Process video with improved ANPR system
    
    Args:
        output_dir: Directory for output video
        output_filename: Name of output file
        config_name: Configuration preset ('default', 'urban', 'night', etc.)
        max_frames: Maximum frames to process (for testing)
        debug: Print debug information
    """
    
    # Load configuration
    config = get_config(config_name)
    print(f"\n📋 Using configuration: {config.name}")
    print(f"   {config.description}")
    
    # Find input video
    input_path = find_input_video()
    if not input_path:
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Initialize ANPR system
    print("\n⚙️  Initializing ANPR system...")
    gpu_available = torch.cuda.is_available()
    print(f"   GPU available: {'Yes' if gpu_available else 'No'}")
    reader = PlateReader(gpu=gpu_available)
    
    # Open video
    print(f"\n▶️  Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"   ✗ Error opening video at {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("   ✗ Error creating output video writer")
        cap.release()
        return

    # Processing loop
    frame_count = 0
    total_plates_detected = 0
    high_confidence_plates = 0
    
    print(f"\n🎬 Processing video...")
    print(f"   Max frames: {max_frames if max_frames else 'All'}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Stop at max_frames if specified
        if max_frames and frame_count > max_frames:
            break

        # Process frame
        plates = reader.extract_plate_from_frame(frame, frame_idx=frame_count)

        # Draw results
        for text, conf, (x1, y1, x2, y2) in plates:
            total_plates_detected += 1
            
            # Color based on confidence
            if conf >= 0.65:
                color = (0, 255, 0)  # Green - high confidence
                high_confidence_plates += 1
            elif conf >= 0.50:
                color = (0, 165, 255)  # Orange - medium confidence
            else:
                color = (0, 0, 255)  # Red - lower confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{text} ({conf:.2f})"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            
            # Background rectangle
            cv2.rectangle(
                frame, 
                (x1, y1 - label_h - 12), 
                (x1 + label_w + 8, y1 + 2),
                color, 
                cv2.FILLED
            )
            
            # Text
            cv2.putText(
                frame, 
                label, 
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.65, 
                (0, 0, 0), 
                2
            )
            
            # Debug output
            if debug:
                print(f"  Frame {frame_count}: {text} (conf: {conf:.3f})")

        # Write frame
        out.write(frame)

        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   ✓ Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

    # Cleanup
    cap.release()
    out.release()
    
    # Summary statistics
    print(f"\n✅ Complete! Output saved to: {os.path.abspath(output_path)}")
    print(f"\n📊 Summary Statistics:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total plates detected: {total_plates_detected}")
    print(f"   High confidence (>0.65): {high_confidence_plates}")
    if total_plates_detected > 0:
        avg_plates_per_frame = total_plates_detected / frame_count
        print(f"   Avg plates/frame: {avg_plates_per_frame:.2f}")
        print(f"   High confidence rate: {(high_confidence_plates/total_plates_detected)*100:.1f}%")


def process_single_frame(image_path, config_name='default', debug=True):
    """Process a single image for testing"""
    
    print(f"\n📋 Using configuration: {config_name}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"▶️  Loading image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("✗ Error loading image")
        return
    
    print("⚙️  Initializing ANPR system...")
    reader = PlateReader(gpu=torch.cuda.is_available())
    
    print("🔍 Processing frame...")
    plates = reader.extract_plate_from_frame(frame)
    
    print(f"\n✅ Found {len(plates)} plate(s):\n")
    
    for text, conf, (x1, y1, x2, y2) in plates:
        print(f"   • {text}")
        print(f"     Confidence: {conf:.3f}")
        print(f"     Bounding box: ({x1}, {y1}, {x2}, {y2})")
        print()
    
    # Draw and display
    output_image = frame.copy()
    for text, conf, (x1, y1, x2, y2) in plates:
        color = (0, 255, 0) if conf >= 0.65 else (0, 165, 255)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        label = f"{text} ({conf:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.rectangle(
            output_image,
            (x1, y1 - label_h - 12),
            (x1 + label_w + 8, y1 + 2),
            color,
            cv2.FILLED
        )
        cv2.putText(
            output_image,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )
    
    # Save result
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "detected_plates.jpg")
    cv2.imwrite(output_path, output_image)
    print(f"📸 Saved output to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improved ANPR video processing with configuration presets"
    )
    parser.add_argument(
        '--mode', 
        choices=['video', 'image'], 
        default='video',
        help='Processing mode: video or single image'
    )
    parser.add_argument(
        '--config',
        default='default',
        choices=['default', 'highway', 'urban', 'night', 'parked', 'autorickshaw', 
                'commercial', 'bright_sun', 'congestion'],
        help='Configuration preset to use'
    )
    parser.add_argument(
        '--input',
        help='Input video/image path (optional, will auto-detect video)'
    )
    parser.add_argument(
        '--output',
        default=r"data\sample_output",
        help='Output directory for processed video'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to process (for testing)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        process_video(
            output_dir=args.output,
            config_name=args.config,
            max_frames=args.max_frames,
            debug=args.debug
        )
    else:  # image mode
        if not args.input:
            print("Error: --input required for image mode")
            exit(1)
        process_single_frame(args.input, config_name=args.config, debug=args.debug)