import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="input_vids/traffic3.mp4")
parser.add_argument("--output_dir", type=str, default="output_runs")
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUTPUT_DIR / f"clean_tracked_{INPUT_VIDEO_PATH.stem}.mp4")

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11x.pt")
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not out.isOpened():
    out = cv2.VideoWriter(str(OUTPUT_DIR / f"clean_tracked_{INPUT_VIDEO_PATH.stem}.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# ----------------- 1. ACTIVE ROAD ROI POLYGON -----------------
# Adjust these 4 points to cover the active roadway and exclude the parked bike bay
ROAD_POLYGON = np.array([
    [int(width * 0.35), int(height * 0.30)],  # Top-Left (near horizon)
    [int(width * 0.85), int(height * 0.30)],  # Top-Right
    [int(width * 1.00), int(height * 1.00)],  # Bottom-Right
    [int(width * 0.25), int(height * 1.00)]   # Bottom-Left
], dtype=np.int32)

track_history = defaultdict(list)
track_start_positions = {}

def get_id_color(track_id):
    np.random.seed(int(track_id) * 101)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))

def is_inside_roi(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

print(f"Running Production Traffic Pipeline with Road ROI...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference directly
    results = model.track(
        source=frame,
        persist=True,
        tracker="custom_tracker.yaml",
        classes=VEHICLE_CLASSES,
        imgsz=1280,
        conf=0.35,
        iou=0.60,
        device=device,
        verbose=False
    )

    # Optional: Draw faint ROI polygon boundary on the frame
    cv2.polylines(frame, [ROAD_POLYGON], isClosed=True, color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)

    if results[0].boxes.id is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        names = results[0].names

        for (x1, y1, x2, y2), (x, y, w, h), track_id, cls_id in zip(boxes_xyxy, boxes_xywh, track_ids, class_ids):
            contact_point = (int(x), int(y + h / 2.0))

            # Filter 1: Check if vehicle base is inside the active road corridor
            if not is_inside_roi(contact_point, ROAD_POLYGON):
                continue

            # Filter 2: Motion Displacement Gate (Filters out static parked cars inside ROI)
            if track_id not in track_start_positions:
                track_start_positions[track_id] = contact_point
            
            start_x, start_y = track_start_positions[track_id]
            total_displacement = np.hypot(contact_point[0] - start_x, contact_point[1] - start_y)
            
            track_history[track_id].append(contact_point)
            color = get_id_color(track_id)

            # Draw trajectory path only if vehicle has actively moved > 30 pixels[cite: 2]
            if total_displacement > 30 and len(track_history[track_id]) > 2:
                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Draw Bounding Box & Label
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"#{track_id} {names[cls_id]}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            badge_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, y1), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + 3, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"Successfully processed! Output saved to: {output_path}")
