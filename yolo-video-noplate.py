import cv2
import easyocr
import os
import re
from ultralytics import YOLO

# 1. Paths configuration
input_path = "traffic_noplate.mp4"
output_path = "output_anpr.avi"  # Using .avi with XVID is universally supported across OpenCV builds

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input video '{input_path}' not found. Check the file path.")

# 2. Model & OCR initialization
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=True)  # Set to False if GPU is unavailable

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# 3. Setup Video Capture & Writer
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError(f"Could not open video file: {input_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Fallback FPS check (some codecs return 0 or invalid FPS)
if fps <= 0 or fps is None:
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    raise IOError(f"VideoWriter failed to open. Try changing codec or format to .mp4 with 'avc1'.")

print(f"Processing video: {width}x{height} @ {fps} FPS")

frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO inference
        results = model(frame, conf=0.35, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 > x1 and y2 > y1:
                plate_crop = frame[y1:y2, x1:x2]

                # Grayscale preprocessing
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                ocr_results = reader.readtext(gray)

                plate_text = ""
                for (_, text, score) in ocr_results:
                    if score > 0.25:
                        plate_text += clean_plate_text(text)

                label = plate_text if plate_text else "Plate"

                # Draw bounding box & text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + (len(label) * 12), y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Done! File saved to {output_path} (Total frames: {frame_count})")
