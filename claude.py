import argparse
import inspect
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
parser = argparse.ArgumentParser(description="CCTV Vehicle Tracker")
parser.add_argument("--input", type=str, default="input_vids/traffic3.mp4")
parser.add_argument("--output_dir", type=str, default="output_runs")
parser.add_argument("--tracker_method", type=str, default="bytetrack",
                     choices=["bytetrack", "deepocsort"])
parser.add_argument("--reid_weights", type=str, default="osnet_x0_25_msmt17.pt",
                     help="Only used if --tracker_method deepocsort")
parser.add_argument("--static_disp_px", type=float, default=25.0)
parser.add_argument("--static_window_s", type=float, default=1.5)
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"{args.tracker_method}_tracked_{INPUT_VIDEO_PATH.stem}.mp4"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
half = torch.cuda.is_available()

# ----------------- VERSION-ROBUST TRACKER LOADER -----------------
# Same defensive approach as before: boxmot has moved/renamed classes
# across releases, so try multiple import paths and only pass kwargs
# the installed constructor actually accepts.

def load_tracker(method):
    candidates = {
        "bytetrack": [
            ("boxmot.trackers.bbox.bytetrack", "ByteTrack"),      # v23.x
            ("boxmot", "BYTETracker"),                             # older
            ("boxmot.trackers.bytetrack.bytetrack", "ByteTrack"),  # mid-version
        ],
        "deepocsort": [
            ("boxmot.trackers.bbox.deepocsort", "DeepOcSort"),     # v23.x
            ("boxmot", "DeepOCSORT"),                              # older
            ("boxmot.trackers.deepocsort.deepocsort", "DeepOcSort"),
        ],
    }[method]

    cls = None
    errors = []
    for module_path, class_name in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"Loaded {method} tracker from {module_path}.{class_name}")
            break
        except (ImportError, AttributeError) as e:
            errors.append(f"{module_path}.{class_name}: {e}")
    if cls is None:
        raise ImportError(f"Could not load {method} tracker. Tried:\n" + "\n".join(errors))

    all_kwargs = {
        "reid_weights": Path(args.reid_weights),
        "model_weights": Path(args.reid_weights),
        "device": device,
        "half": half,
        "fp16": half,
        "per_class": False,
        "track_thresh": 0.3,
        "det_thresh": 0.3,
        "match_thresh": 0.8,
        "track_buffer": 30,
        "max_age": 30,
        "frame_rate": 30,
    }
    sig_params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in all_kwargs.items() if k in sig_params}
    print(f"Passing kwargs: {list(accepted.keys())}")
    return cls(**accepted)


tracker = load_tracker(args.tracker_method)

# ----------------- DETECTOR + VIDEO I/O -----------------
detector = YOLO("yolo11x.pt")
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
if not out.isOpened():
    output_path = output_path.with_suffix(".avi")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))


def get_id_color(track_id):
    np.random.seed(int(track_id) * 137)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


track_history = defaultdict(lambda: deque(maxlen=int(fps * 6)))
frame_idx = 0
print(f"Tracking with {args.tracker_method} on device={device}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    results = detector(
        source=frame, classes=VEHICLE_CLASSES, imgsz=1280,
        conf=0.28, iou=0.65, device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )[0]

    if len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        dets = np.column_stack([xyxy, conf, cls])
    else:
        dets = np.empty((0, 6))

    tracks = tracker.update(dets, frame)

    for row in tracks:
        x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id, cls_id = int(track_id), int(cls_id)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        track_history[track_id].append((cx, cy, frame_idx))

        window_s = args.static_window_s
        hist = track_history[track_id]
        pts_in_window = [p for p in hist if (frame_idx - p[2]) <= fps * window_s]
        is_static = False
        if len(pts_in_window) >= max(3, int(fps * window_s * 0.5)):
            xs, ys = [p[0] for p in pts_in_window], [p[1] for p in pts_in_window]
            is_static = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < args.static_disp_px

        color = get_id_color(track_id)
        if not is_static and len(hist) > 2:
            pts = np.array([(int(p[0]), int(p[1])) for p in hist], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
print(f"Tracking complete: {output_path.resolve()}")
