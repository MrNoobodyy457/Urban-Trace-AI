import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- 1. CONFIGURATION & PATHS -----------------
INPUT_VIDEO_PATH = "traffic.mp4"

# Set your custom output directory and filename here
OUTPUT_DIR = Path("./output_runs")
OUTPUT_FILENAME = "traffic_full_trajectory.mp4"
# -----------------------------------------------------------

# Automatically create destination folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUTPUT_DIR / OUTPUT_FILENAME)

# Verify input video exists
if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Input video not found at: {os.path.abspath(INPUT_VIDEO_PATH)}")

# Setup Model & CUDA
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt")

# COCO Vehicle Classes: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# ----------------- 2. VIDEO CAPTURE & WRITER -----------------
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video file: {INPUT_VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if fps <= 0 or np.isnan(fps) else fps

# Codec initialization with fallback
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    # Fallback to AVI / XVID if MP4 container fails
    output_path = str(OUTPUT_DIR / (Path(OUTPUT_FILENAME).stem + ".avi"))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Warning: Standard MP4 codec unavailable. Falling back to: {output_path}")

print(f"Processing started...")
print(f"Saving output to: {os.path.abspath(output_path)}")

# ----------------- 3. TRACKING & ANNOTATION -----------------
track_history = defaultdict(lambda: [])

def get_id_color(track_id):
    np.random.seed(int(track_id) * 97)
    color = np.random.randint(60, 255, size=3)
    return tuple(int(c) for c in color)

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Run ByteTrack on GPU
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

        # Step 1: Draw persistent full trajectory paths
        for (x, y, w, h), track_id in zip(boxes_xywh, track_ids):
            color = get_id_color(track_id)
            bottom_center = (int(x), int(y + h / 2))
            track_history[track_id].append(bottom_center)

            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

        # Step 2: Draw crisp bounding boxes and high-contrast labels
        for (x1, y1, x2, y2), track_id, cls_id in zip(boxes_xyxy, track_ids, class_ids):
            color = get_id_color(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Solid 2px Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"#{track_id} {names[cls_id]}"
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            cv2.rectangle(
                frame,
                (x1, badge_y1),
                (x1 + text_w + 6, badge_y2),
                (20, 20, 20),
                -1
            )
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

print(f"\nProcessing complete! ({frame_count} frames)")
print(f"Video file successfully written to: {os.path.abspath(output_path)}")
