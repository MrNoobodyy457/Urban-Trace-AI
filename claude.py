import argparse
import tempfile
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
parser = argparse.ArgumentParser(description="CCTV Vehicle Tracker (ByteTrack + stationary-identity registry)")
parser.add_argument("--input", type=str, default="input_vids/traffic1.mp4")
parser.add_argument("--output_dir", type=str, default="output_runs")
parser.add_argument("--static_disp_px", type=float, default=25.0,
                     help="Max pixel displacement over the window to be considered parked")
parser.add_argument("--static_window_s", type=float, default=1.5,
                     help="Seconds of history checked for the parked classification")
parser.add_argument("--max_gap_seconds", type=float, default=2.0,
                     help="Frame-timestamp gap treated as a discontinuity (dropped frames / hard cut)")
parser.add_argument("--min_confirm_frames", type=int, default=5,
                     help="Frames a track must persist before being drawn/counted (ghost-detection filter)")
parser.add_argument("--track_buffer_seconds", type=float, default=3.0,
                     help="How long ByteTrack keeps a lost track alive before giving up on it. "
                          "This alone won't fully solve long occlusions of static objects — "
                          "that's what the stationary-identity registry below is for — but a "
                          "too-short buffer (Ultralytics' default ~1s) makes it far worse.")
parser.add_argument("--anchor_radius_px", type=float, default=None,
                     help="Spatial tolerance for re-matching a static object to a prior identity. "
                          "Defaults to ~4.5%% of the frame diagonal if not set.")
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"tracked_{INPUT_VIDEO_PATH.stem}.mp4"

device = 0 if torch.cuda.is_available() else "cpu"

detector = YOLO("yolo11l.pt")
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Video FPS reported as: {fps:.2f}")

anchor_radius_px = args.anchor_radius_px or (np.hypot(width, height) * 0.045)

out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not out.isOpened():
    output_path = output_path.with_suffix(".avi")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# --- Generate a ByteTrack config with a buffer scaled to THIS video's fps ---
# Ultralytics' default track_buffer=30 frames means ~1s on a 30fps video.
# That's nowhere near enough to survive a car passing in front of a parked
# row, or a couple seconds of shadow-induced confidence flicker. We compute
# this from fps rather than hardcoding a frame count, same principle as
# everywhere else in this script.
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
print(f"Generated tracker config with track_buffer={tracker_cfg['track_buffer']} frames "
      f"(~{args.track_buffer_seconds}s) at {tmp_tracker_yaml}")


def get_id_color(track_id):
    np.random.seed(int(track_id) * 137)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


class StationaryIdentityRegistry:
    """
    ByteTrack (and most motion trackers) mint a new ID whenever a track is
    lost and later re-detected outside its buffer window. For genuinely
    moving vehicles that's usually fine — but for PARKED vehicles, this is
    exactly wrong: a bike that hasn't moved an inch shouldn't get a new ID
    just because a car passed in front of it for two seconds.

    This registry sits AFTER the tracker, not instead of it. It only
    intervenes for tracks already classified as static (near-zero
    displacement over the rolling window). For those, it looks up whether a
    same-class object was already registered near this position; if so, it
    overrides the raw tracker ID with the previously-assigned stable ID for
    display/counting purposes. Moving vehicles are untouched and keep
    whatever ID the tracker gives them, since spatial-proximity matching is
    the WRONG heuristic for fast-moving objects (that's what motion
    prediction is for).
    """

    def __init__(self, radius_px, cls_agnostic=False):
        self.radius_px = radius_px
        self.cls_agnostic = cls_agnostic
        self.anchors = {}  # display_id -> {"center": (x,y), "cls": int, "last_seen": frame_idx}
        self.next_display_id = 1
        self.raw_to_display = {}  # raw_track_id -> display_id, cached per tracker ID for stability

    def resolve(self, raw_track_id, center, cls_id, frame_idx):
        # If we've already bound this raw ID to a display ID this session, keep using it.
        if raw_track_id in self.raw_to_display:
            display_id = self.raw_to_display[raw_track_id]
            self.anchors[display_id].update(center=center, cls=cls_id, last_seen=frame_idx)
            return display_id

        # Otherwise, look for a nearby same-class anchor to re-adopt.
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

        # No match — register a brand new stable identity.
        display_id = self.next_display_id
        self.next_display_id += 1
        self.raw_to_display[raw_track_id] = display_id
        self.anchors[display_id] = {"center": center, "cls": cls_id, "last_seen": frame_idx}
        return display_id


stationary_registry = StationaryIdentityRegistry(radius_px=anchor_radius_px)

track_history = defaultdict(lambda: deque(maxlen=int(fps * 6)))
track_first_seen = {}

frame_idx = 0
last_frame_time = None
discontinuity_count = 0

print("Tracking...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    # --- Discontinuity detection ---
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if last_frame_time is not None and (current_time - last_frame_time) > args.max_gap_seconds:
        discontinuity_count += 1
        print(f"[frame {frame_idx}] Discontinuity detected: {current_time - last_frame_time:.2f}s gap "
              f"(#{discontinuity_count}). Resetting track history.")
        track_history.clear()
        track_first_seen.clear()
        # Note: we deliberately do NOT clear stationary_registry here. A parked
        # vehicle is still parked on the other side of a dropped-frames gap or
        # feed reconnect — that's real continuity, unlike a hard scene cut.
        # If you expect discontinuities to sometimes mean "different camera" or
        # "different scene," clear it here too; for a single fixed CCTV feed,
        # keeping it is the correct default.
    last_frame_time = current_time

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

    for box, raw_id, cls_id, conf in zip(xyxy, raw_ids, cls_ids, confs):
        raw_id = int(raw_id)
        cls_id = int(cls_id)
        x1, y1, x2, y2 = [int(v) for v in box]

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

        # --- This is the key change ---
        # Moving vehicles: display the raw tracker ID directly (motion
        # trackers are good at this, spatial-anchor matching is not).
        # Static vehicles: resolve through the stationary registry instead,
        # so re-detection after a brief occlusion reuses the SAME display ID
        # rather than minting a new one.
        if is_static:
            display_id = stationary_registry.resolve(raw_id, (cx, cy), cls_id, frame_idx)
        else:
            display_id = raw_id

        color = get_id_color(display_id)

        if not is_static and len(hist) > 2:
            pts = np.array([(int(p[0]), int(p[1])) for p in hist], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        class_name = detector.names[cls_id]
        status = " [P]" if is_static else ""
        label = f"#{display_id} {class_name}{status}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        badge_y1 = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, y1), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + 3, y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"Tracking complete: {output_path.resolve()}")
print(f"Discontinuities detected: {discontinuity_count}")
print(f"Distinct stable static identities registered: {len(stationary_registry.anchors)}")
