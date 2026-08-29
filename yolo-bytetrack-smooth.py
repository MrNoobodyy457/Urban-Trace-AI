import os
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- 1. CLI ARGUMENTS & CONFIG -----------------
parser = argparse.ArgumentParser(description="Urban Trace AI - Vehicle Trajectory Tracker")
parser.add_argument(
    "--input", 
    type=str, 
    default="input_vids/traffic3.mp4", 
    help="Path to input video (e.g. input_vids/cam1.mp4)"
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    default="output_runs", 
    help="Directory to save output video"
)
parser.add_argument(
    "--smooth_factor", 
    type=float, 
    default=0.65, 
    help="EMA smoothing alpha (0.1 = very smooth/sluggish, 1.0 = raw/jittery)"
)
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_VIDEO_PATH.exists():
    raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH.resolve()}")

output_filename = f"tracked_{INPUT_VIDEO_PATH.stem}.mp4"
output_path = str(OUTPUT_DIR / output_filename)

# ----------------- 2. MODEL & VIDEO SETUP -----------------
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt")

# COCO Vehicle Classes: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if fps <= 0 or np.isnan(fps) else fps

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    output_path = str(OUTPUT_DIR / f"tracked_{INPUT_VIDEO_PATH.stem}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ----------------- 3. TRACKING & SMOOTHING BUFFERS -----------------
# Stores smoothed points for drawing: {track_id: [(x1, y1), (x2, y2), ...]}
track_history = defaultdict(list)
# Stores latest smoothed (x, y) float coordinate: {track_id: (x, y)}
smoothed_positions = {}

def get_id_color(track_id):
    np.random.seed(int(track_id) * 97)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))

def smooth_point(track_id, raw_point, alpha=0.65):
    """Applies Exponential Moving Average (EMA) to smooth out trajectory jitter."""
    if track_id not in smoothed_positions:
        smoothed_positions[track_id] = raw_point
        return (int(raw_point[0]), int(raw_point[1]))
    
    prev_x, prev_y = smoothed_positions[track_id]
    curr_x, curr_y = raw_point
    
    # EMA formula
    new_x = alpha * curr_x + (1 - alpha) * prev_x
    new_y = alpha * curr_y + (1 - alpha) * prev_y
    
    smoothed_positions[track_id] = (new_x, new_y)
    return (int(round(new_x)), int(round(new_y)))

print(f"Tracking: {INPUT_VIDEO_PATH.name} on {device}")
print(f"Output Destination: {Path(output_path).resolve()}")

# ----------------- 4. PROCESSING LOOP -----------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        device=device,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        names = results[0].names

        # Step 1: Smooth points and draw trajectories
        for (x, y, w, h), track_id in zip(boxes_xywh, track_ids):
            color = get_id_color(track_id)
            raw_contact = (float(x), float(y + h / 2.0))
            
            # Smooth coordinate before appending
            clean_pt = smooth_point(track_id, raw_contact, alpha=args.smooth_factor)
            track_history[track_id].append(clean_pt)

            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

        # Step 2: Draw crisp bounding boxes & contrast badges
        for (x1, y1, x2, y2), track_id, cls_id in zip(boxes_xyxy, track_ids, class_ids):
            color = get_id_color(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"#{track_id} {names[cls_id]}"
            font_scale, font_thickness = 0.5, 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            badge_y1, badge_y2 = max(0, y1 - text_h - 8), y1
            cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, badge_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + 3, badge_y2), color, -1)

            cv2.putText(
                frame,
                label,
                (x1 + 6, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

    out.write(frame)

cap.release()
out.release()
print(f"\nDone! Video saved to: {Path(output_path).resolve()}")
