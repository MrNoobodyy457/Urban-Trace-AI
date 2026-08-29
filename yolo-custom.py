import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
parser = argparse.ArgumentParser(description="Deterministic Spatial-Anchor CCTV Tracker")
parser.add_argument("--input", type=str, default="input_vids/traffic3.mp4", help="Path to input video")
parser.add_argument("--output_dir", type=str, default="output_runs", help="Output directory")
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUTPUT_DIR / f"anchor_tracked_{INPUT_VIDEO_PATH.stem}.mp4")

device = 0 if torch.cuda.is_available() else "cpu"
detector = YOLO("yolo11x.pt")
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not out.isOpened():
    out = cv2.VideoWriter(str(OUTPUT_DIR / f"anchor_tracked_{INPUT_VIDEO_PATH.stem}.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))


def get_id_color(track_id):
    np.random.seed(int(track_id) * 137)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


class SpatialAnchorTracker:
    def __init__(self, fps=30.0, frame_size=(1920, 1080)):
        self.fps = fps
        self.width, self.height = frame_size
        self.next_id = 1
        
        # Active tracks: {id: {"box": [x1,y1,x2,y2], "center": (x,y), "cls": cls, "last_seen": frame_idx, "hits": int, "is_static": bool, "history": []}}
        self.tracks = {}
        # Permanent spatial anchors for parked objects: {id: {"center": (x,y), "cls": cls, "box": []}}
        self.static_anchors = {}
        
        self.anchor_radius = np.hypot(self.width, self.height) * 0.045  # ~50-80px spatial tolerance
        self.max_lost_frames = int(fps * 20.0)  # Retain active tracks for 20 seconds

    def update(self, detections, frame_idx):
        assigned_track_ids = set()
        unmatched_dets = []

        # --- PHASE 1: Match against Static Anchors (Parked Memory) ---
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            matched_anchor_id = None
            min_dist = float("inf")
            for anchor_id, anchor in self.static_anchors.items():
                if anchor_id in assigned_track_ids:
                    continue
                if anchor["cls"] != cls_id:
                    continue
                
                dist = np.hypot(center[0] - anchor["center"][0], center[1] - anchor["center"][1])
                if dist < self.anchor_radius and dist < min_dist:
                    min_dist = dist
                    matched_anchor_id = anchor_id

            if matched_anchor_id is not None:
                assigned_track_ids.add(matched_anchor_id)
                self.tracks[matched_anchor_id]["box"] = [x1, y1, x2, y2]
                self.tracks[matched_anchor_id]["center"] = center
                self.tracks[matched_anchor_id]["last_seen"] = frame_idx
                self.tracks[matched_anchor_id]["hits"] += 1
                self.tracks[matched_anchor_id]["history"].append((int(center[0]), int(y2)))
            else:
                unmatched_dets.append(det)

        # --- PHASE 2: Match Remaining Detections to Active Moving Tracks via IoU ---
        still_unmatched = []
        for det in unmatched_dets:
            x1, y1, x2, y2, conf, cls_id = det
            best_iou = 0.20
            best_track_id = None

            for trk_id, trk in self.tracks.items():
                if trk_id in assigned_track_ids:
                    continue
                if trk["cls"] != cls_id:
                    continue
                if frame_idx - trk["last_seen"] > self.max_lost_frames:
                    continue

                iou = compute_iou([x1, y1, x2, y2], trk["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = trk_id

            if best_track_id is not None:
                assigned_track_ids.add(best_track_id)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                self.tracks[best_track_id]["box"] = [x1, y1, x2, y2]
                self.tracks[best_track_id]["center"] = center
                self.tracks[best_track_id]["last_seen"] = frame_idx
                self.tracks[best_track_id]["hits"] += 1
                self.tracks[best_track_id]["history"].append((int(center[0]), int(y2)))
            else:
                still_unmatched.append(det)

        # --- PHASE 3: Register New Candidates ---
        for det in still_unmatched:
            x1, y1, x2, y2, conf, cls_id = det
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            new_id = self.next_id
            self.next_id += 1

            self.tracks[new_id] = {
                "box": [x1, y1, x2, y2],
                "center": center,
                "cls": cls_id,
                "last_seen": frame_idx,
                "hits": 1,
                "is_static": False,
                "history": [(int(center[0]), int(y2))]
            }

        # --- PHASE 4: Update Stationarity Anchors ---
        for trk_id, trk in list(self.tracks.items()):
            if trk["hits"] >= int(self.fps * 1.5):  # Visible for >1.5s
                hist = trk["history"]
                if len(hist) > 10:
                    start_pt = hist[0]
                    curr_pt = hist[-1]
                    displacement = np.hypot(curr_pt[0] - start_pt[0], curr_pt[1] - start_pt[1])
                    
                    if displacement < 25:  # Has moved less than 25px total
                        trk["is_static"] = True
                        self.static_anchors[trk_id] = {
                            "center": trk["center"],
                            "cls": trk["cls"],
                            "box": trk["box"]
                        }

        # Return active tracks visible in current frame with >=3 hits (filters 1-frame ghost boxes)
        output_tracks = []
        for trk_id, trk in self.tracks.items():
            if trk["last_seen"] == frame_idx and trk["hits"] >= 3:
                output_tracks.append((trk["box"], trk_id, trk["cls"], trk["is_static"], trk["history"]))

        return output_tracks


tracker = SpatialAnchorTracker(fps=fps, frame_size=(width, height))
frame_idx = 0

print(f"Tracking using Spatial-Anchor Memory on {device}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    # High-resolution raw detection
    results = detector(
        source=frame,
        classes=VEHICLE_CLASSES,
        imgsz=1280,
        conf=0.28,
        iou=0.65,
        device=device,
        verbose=False
    )[0]

    detections = []
    if len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.int().cpu().numpy()
        for b, c, cl in zip(xyxy, conf, cls):
            detections.append([b[0], b[1], b[2], b[3], c, cl])

    active_tracks = tracker.update(detections, frame_idx)

    for box, track_id, cls_id, is_static, history in active_tracks:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = get_id_color(track_id)

        # Draw trajectory only for moving objects
        if not is_static and len(history) > 2:
            pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label badge
        class_name = detector.names[cls_id]
        status = " [P]" if is_static else ""
        label = f"#{track_id} {class_name}{status}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        badge_y1 = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, y1), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + 3, y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"Tracking complete: {Path(output_path).resolve()}")
