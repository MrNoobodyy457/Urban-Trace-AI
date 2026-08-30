import argparse
import tempfile
from pathlib import Path
from collections import defaultdict, deque
import sys
import os

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Add the src folder to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from anpr import PlateReader

def get_id_color(track_id):
    np.random.seed(int(track_id) * 137)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))

class StationaryIdentityRegistry:
    def __init__(self, radius_px, cls_agnostic=False):
        self.radius_px = radius_px
        self.cls_agnostic = cls_agnostic
        self.anchors = {}  # display_id -> {"center": (x,y), "cls": int, "last_seen": frame_idx}
        self.next_display_id = 1
        self.raw_to_display = {}  # raw_track_id -> display_id, cached per tracker ID for stability

    def resolve(self, raw_track_id, center, cls_id, frame_idx):
        if raw_track_id in self.raw_to_display:
            display_id = self.raw_to_display[raw_track_id]
            self.anchors[display_id].update(center=center, cls=cls_id, last_seen=frame_idx)
            return display_id

        best_id, best_dist = None, float("inf")
        for display_id, anchor in self.anchors.items():
            if not self.cls_agnostic and anchor["cls"] != cls_id:
                continue
            dist = np.hypot(center[0] - anchor["center"][0], center[1] - anchor["center"][1])
            if dist < self.radius_px and dist < best_dist:
                best_dist, best_id = dist, display_id

        if best_id is not None:
            self.raw_to_display[raw_track_id] = best_id
            self.anchors[best_id].update(center=center, cls=cls_id, last_seen=frame_idx)
            return best_id

        display_id = self.next_display_id
        self.next_display_id += 1
        self.raw_to_display[raw_track_id] = display_id
        self.anchors[display_id] = {"center": center, "cls": cls_id, "last_seen": frame_idx}
        return display_id

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def is_inside(inner_box, outer_box):
    """Check if inner_box is completely or mostly inside outer_box"""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Calculate area of inner box
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    if inner_area == 0:
        return False
        
    # Calculate intersection
    xA = max(ix1, ox1)
    yA = max(iy1, oy1)
    xB = min(ix2, ox2)
    yB = min(iy2, oy2)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # If more than 50% of the plate box is inside the vehicle box, it belongs to it
    return (interArea / float(inner_area)) > 0.5

def main():
    parser = argparse.ArgumentParser(description="Integrated Vehicle Tracker and ANPR Pipeline")
    parser.add_argument("--input", type=str, default="input_vids/traffic1.mp4")
    parser.add_argument("--output_dir", type=str, default="output_runs")
    parser.add_argument("--static_disp_px", type=float, default=25.0)
    parser.add_argument("--static_window_s", type=float, default=1.5)
    parser.add_argument("--max_gap_seconds", type=float, default=2.0)
    parser.add_argument("--min_confirm_frames", type=int, default=5)
    parser.add_argument("--track_buffer_seconds", type=float, default=3.0)
    parser.add_argument("--anchor_radius_px", type=float, default=None)
    args = parser.parse_args()

    INPUT_VIDEO_PATH = Path(args.input)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"integrated_{INPUT_VIDEO_PATH.stem}.mp4"

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Loading YOLO vehicle detector...")
    detector = YOLO("yolo11l.pt")
    VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

    print("Loading ANPR plate reader...")
    plate_reader = PlateReader(model_path="license_plate_detector.pt", gpu=torch.cuda.is_available())

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"Failed to open video {INPUT_VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS reported as: {fps:.2f}")

    anchor_radius_px = args.anchor_radius_px or (np.hypot(width, height) * 0.045)

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        output_path = output_path.with_suffix(".avi")
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    tracker_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": int(fps * args.track_buffer_seconds),
        "match_thresh": 0.8,
        "fuse_score": True,
    }
    tmp_tracker_yaml = Path(tempfile.gettempdir()) / f"bytetrack_runtime_{INPUT_VIDEO_PATH.stem}.yaml"
    with open(tmp_tracker_yaml, "w") as f:
        yaml.safe_dump(tracker_cfg, f)

    stationary_registry = StationaryIdentityRegistry(radius_px=anchor_radius_px)

    track_history = defaultdict(lambda: deque(maxlen=int(fps * 6)))
    track_first_seen = {}
    
    # Track the best plate read for each display ID
    # mapping: display_id -> {"text": string, "conf": float}
    vehicle_plates = {}

    frame_idx = 0
    last_frame_time = None
    discontinuity_count = 0

    print("Tracking and reading plates...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if last_frame_time is not None and (current_time - last_frame_time) > args.max_gap_seconds:
            discontinuity_count += 1
            track_history.clear()
            track_first_seen.clear()
        last_frame_time = current_time

        # 1. Run Vehicle Tracking
        results = detector.track(
            source=frame,
            classes=VEHICLE_CLASSES,
            imgsz=1280,
            conf=0.28,
            iou=0.65,
            tracker=str(tmp_tracker_yaml),
            persist=True,
            device=device,
            verbose=False,
        )[0]

        if results.boxes.id is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            raw_ids = results.boxes.id.int().cpu().numpy()
            cls_ids = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
        else:
            xyxy, raw_ids, cls_ids, confs = [], [], [], []

        vehicle_bboxes = []
        vehicle_display_ids = []

        # Process vehicle tracks
        for box, raw_id, cls_id, conf in zip(xyxy, raw_ids, cls_ids, confs):
            raw_id = int(raw_id)
            cls_id = int(cls_id)
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if raw_id not in track_first_seen:
                track_first_seen[raw_id] = frame_idx
            if frame_idx - track_first_seen[raw_id] < args.min_confirm_frames:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            track_history[raw_id].append((cx, cy, frame_idx))

            window_s = args.static_window_s
            hist = track_history[raw_id]
            pts_in_window = [p for p in hist if (frame_idx - p[2]) <= fps * window_s]
            is_static = False
            if len(pts_in_window) >= max(3, int(fps * window_s * 0.5)):
                xs = [p[0] for p in pts_in_window]
                ys = [p[1] for p in pts_in_window]
                disp = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
                is_static = disp < args.static_disp_px

            if is_static:
                display_id = stationary_registry.resolve(raw_id, (cx, cy), cls_id, frame_idx)
            else:
                display_id = raw_id

            vehicle_bboxes.append((x1, y1, x2, y2))
            vehicle_display_ids.append((display_id, cls_id, is_static, hist))

        # 2. Run ANPR
        # We process every 2nd frame to save some computation, plates persist in vehicle_plates anyway.
        if frame_idx % 2 == 1:
            plates = plate_reader.extract_plate_from_frame(frame, frame_idx=frame_idx)
            
            # Associate plates with vehicles
            for p_text, p_conf, p_bbox in plates:
                best_vehicle_id = None
                
                for (vx1, vy1, vx2, vy2), (v_id, _, _, _) in zip(vehicle_bboxes, vehicle_display_ids):
                    if is_inside(p_bbox, (vx1, vy1, vx2, vy2)):
                        best_vehicle_id = v_id
                        break
                
                if best_vehicle_id is not None:
                    # Update plate if it's the first time, or if confidence is higher
                    if best_vehicle_id not in vehicle_plates or p_conf > vehicle_plates[best_vehicle_id]["conf"]:
                        vehicle_plates[best_vehicle_id] = {"text": p_text, "conf": p_conf}
                
                # Draw plate bounding box with OCR reading on the frame directly as well
                px1, py1, px2, py2 = p_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

        # 3. Draw Vehicles and associated plates
        for (x1, y1, x2, y2), (display_id, cls_id, is_static, hist) in zip(vehicle_bboxes, vehicle_display_ids):
            color = get_id_color(display_id)

            if not is_static and len(hist) > 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in hist], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            class_name = detector.names[cls_id]
            status = " [P]" if is_static else ""
            
            # Incorporate plate text if known
            plate_info = ""
            if display_id in vehicle_plates:
                plate_info = f" | Plate: {vehicle_plates[display_id]['text']} ({vehicle_plates[display_id]['conf']:.2f})"
            
            label = f"#{display_id} {class_name}{status}{plate_info}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            if y1 - text_h - 8 < 0:
                badge_y1 = y1
                badge_y2 = y1 + text_h + 8
                text_y = y1 + text_h + 4
            else:
                badge_y1 = y1 - text_h - 8
                badge_y2 = y1
                text_y = y1 - 4

            cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, badge_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + 3, badge_y2), color, -1)
            cv2.putText(frame, label, (x1 + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        
        # Simple progress
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}")

    cap.release()
    out.release()
    print(f"Integration tracking complete: {output_path.resolve()}")
    print(f"Discontinuities detected: {discontinuity_count}")
    print(f"Distinct stable static identities registered: {len(stationary_registry.anchors)}")
    print(f"Total distinct license plates logged: {len(vehicle_plates)}")

if __name__ == "__main__":
    main()
