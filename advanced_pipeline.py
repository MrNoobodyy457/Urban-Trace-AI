import argparse
import tempfile
from pathlib import Path
from collections import defaultdict, deque
import sys
import os
import re

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Add the src folder to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from anpr import PlateReader

VALID_STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA", "GJ", "HR", "HP", "JH", "JK", 
    "KA", "KL", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", 
    "TN", "TR", "TS", "UP", "UK", "WB"
]

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_fix_plate(text):
    """
    Cleans up character confusions, matches closest state code, 
    and checks if it broadly looks like an Indian plate.
    """
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Common OCR confusions
    dict_char_to_num = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4', 'G': '6'}
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '4': 'A', '6': 'G'}
    
    if len(clean_text) < 7 or len(clean_text) > 11:
        return None
        
    arr = list(clean_text)
    
    # First 2 should be letters
    for i in (0, 1):
        if arr[i] in dict_num_to_char:
            arr[i] = dict_num_to_char[arr[i]]
            
    # Try to match to a valid state code if it's off by 1 char
    state_str = "".join(arr[:2])
    if state_str not in VALID_STATE_CODES:
        best_code = state_str
        min_dist = 999
        for code in VALID_STATE_CODES:
            dist = levenshtein_distance(state_str, code)
            if dist < min_dist:
                min_dist = dist
                best_code = code
        # If it's close enough (1 typo), snap it to the valid state code
        if min_dist == 1:
            arr[0] = best_code[0]
            arr[1] = best_code[1]
            
    # Next 1-2 should be numbers
    for i in range(2, min(4, len(arr))):
        if arr[i].isdigit():
            pass
        elif arr[i] in dict_char_to_num:
            arr[i] = dict_char_to_num[arr[i]]
            
    # Last 3-4 should be numbers
    for i in range(max(2, len(arr) - 4), len(arr)):
        if arr[i] in dict_char_to_num:
            arr[i] = dict_char_to_num[arr[i]]
            
    fixed_text = "".join(arr)
    
    # Lenient regex: State Code (2) + District (1-2) + Optional Letters (0-3) + Serial (3-4)
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$'
    if re.match(pattern, fixed_text):
        return fixed_text
    return None

def get_best_plate(reads):
    """
    Given a list of (text, conf), groups by text and returns the one 
    with the highest cumulative confidence (voting).
    """
    if not reads:
        return None, 0.0
        
    votes = defaultdict(float)
    for text, conf in reads:
        votes[text] += conf
        
    best_text = max(votes.keys(), key=lambda k: votes[k])
    return best_text, votes[best_text]


def get_id_color(track_id):
    np.random.seed(int(track_id) * 137)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))

class StationaryIdentityRegistry:
    def __init__(self, radius_px, cls_agnostic=False):
        self.radius_px = radius_px
        self.cls_agnostic = cls_agnostic
        self.anchors = {}
        self.next_display_id = 1
        self.raw_to_display = {}

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

def is_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    if inner_area == 0:
        return False
        
    xA = max(ix1, ox1)
    yA = max(iy1, oy1)
    xB = min(ix2, ox2)
    yB = min(iy2, oy2)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return (interArea / float(inner_area)) > 0.5

def main():
    parser = argparse.ArgumentParser(description="Advanced Pipeline with OCR Voting and Static Filtering")
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
    output_path = OUTPUT_DIR / f"advanced_{INPUT_VIDEO_PATH.stem}.mp4"

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
    tmp_tracker_yaml = Path(tempfile.gettempdir()) / f"bytetrack_advanced_runtime.yaml"
    with open(tmp_tracker_yaml, "w") as f:
        yaml.safe_dump(tracker_cfg, f)

    stationary_registry = StationaryIdentityRegistry(radius_px=anchor_radius_px)

    track_history = defaultdict(lambda: deque(maxlen=int(fps * 6)))
    track_first_seen = {}
    
    # Lifetime displacement to filter out the barrier
    track_start_pos = {}
    track_max_disp = defaultdict(float)
    
    # Track all plate reads for each display ID for voting
    # mapping: display_id -> list of (text, confidence)
    vehicle_reads = defaultdict(list)

    frame_idx = 0
    last_frame_time = None
    discontinuity_count = 0

    print("Tracking and reading plates (Advanced Pipeline)...")

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
            track_start_pos.clear()
            track_max_disp.clear()
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
            
            # Lifetime displacement tracking
            if raw_id not in track_start_pos:
                track_start_pos[raw_id] = (cx, cy)
            
            start_cx, start_cy = track_start_pos[raw_id]
            disp_from_start = np.hypot(cx - start_cx, cy - start_cy)
            if disp_from_start > track_max_disp[raw_id]:
                track_max_disp[raw_id] = disp_from_start
                
            # If object has lived for over 1.5 seconds and NEVER moved more than static_disp_px, it's a barrier/false-positive
            frames_alive = frame_idx - track_first_seen[raw_id]
            if frames_alive > (fps * 1.5) and track_max_disp[raw_id] < args.static_disp_px:
                continue # Ignore this entirely, don't assign display ID, don't draw!

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

        # 2. Run ANPR (Global pass)
        if frame_idx % 2 == 1:
            plates = plate_reader.extract_plate_from_frame(frame, frame_idx=frame_idx)
            
            for p_text, p_conf, p_bbox in plates:
                # Use our new lenient logic on the raw text found by original plate reader if it passed their strict check
                # Note: original plate_reader already filters by strict regex. 
                # We will just accept it as it passed their strict check, and add to voting.
                best_vehicle_id = None
                for (vx1, vy1, vx2, vy2), (v_id, _, _, _) in zip(vehicle_bboxes, vehicle_display_ids):
                    if is_inside(p_bbox, (vx1, vy1, vx2, vy2)):
                        best_vehicle_id = v_id
                        break
                
                if best_vehicle_id is not None:
                    vehicle_reads[best_vehicle_id].append((p_text, p_conf))
                
                px1, py1, px2, py2 = p_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

        # 3. Targeted Crop for Vehicles with NO plate yet or needing more votes
        if frame_idx % 3 == 0:
            for (x1, y1, x2, y2), (display_id, _, _, _) in zip(vehicle_bboxes, vehicle_display_ids):
                # Only target if we have less than 3 votes (we want to ensure we get a plate)
                if len(vehicle_reads[display_id]) < 3:
                    box_h = y2 - y1
                    box_w = x2 - x1
                    if box_h > 40 and box_w > 40:
                        # Crop lower 60% of vehicle where plates usually are
                        crop_y1 = y1 + int(box_h * 0.4)
                        crop = frame[crop_y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            # Enhance contrast for OCR
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            enhanced = clahe.apply(gray)
                            
                            # Run raw OCR
                            ocr_results = plate_reader.reader.readtext(enhanced, allowlist=plate_reader.allowlist)
                            
                            if ocr_results:
                                raw_text = "".join([res[1] for res in ocr_results])
                                avg_conf = float(np.mean([float(res[2]) for res in ocr_results]))
                                
                                # Use fuzzy validation!
                                fixed_plate = fuzzy_fix_plate(raw_text)
                                if fixed_plate and avg_conf > 0.2:
                                    vehicle_reads[display_id].append((fixed_plate, avg_conf))

        # 4. Draw Vehicles and Associated Plates
        for (x1, y1, x2, y2), (display_id, cls_id, is_static, hist) in zip(vehicle_bboxes, vehicle_display_ids):
            color = get_id_color(display_id)

            if not is_static and len(hist) > 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in hist], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            class_name = detector.names[cls_id]
            status = " [P]" if is_static else ""
            
            # Use Voting to get best plate
            plate_info = ""
            best_plate_text, best_plate_score = get_best_plate(vehicle_reads[display_id])
            if best_plate_text:
                plate_info = f" | Plate: {best_plate_text} ({best_plate_score:.1f} score)"
            
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
        
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}")

    cap.release()
    out.release()
    print(f"Advanced tracking complete: {output_path.resolve()}")
    print(f"Total distinct license plates logged (via voting): {len([v for v in vehicle_reads.values() if len(v) > 0])}")

if __name__ == "__main__":
    main()
