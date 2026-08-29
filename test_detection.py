import os
import cv2
import time
from src.anpr import PlateReader

video_path = os.path.join("data", "test_videos", "sample.mp4")

if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'!")
    exit(1)

# Matches the parameter name in PlateReader.__init__
reader = PlateReader(model_path="license_plate_detector.pt", gpu=False)
video = cv2.VideoCapture(video_path)

frame_count = 0
plates_found = 0
start_time = time.time()

print("Running License Plate Recognition...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    results = reader.extract_plate_from_frame(frame)

    for plate, conf in results:
        plates_found += 1
        print(f"Frame {frame_count}: plate={plate} conf={conf:.2f}")

    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"... {frame_count} frames processed, {plates_found} plates found so far ({elapsed:.1f}s)")

video.release()
print(f"Done. Processed {frame_count} frames in {time.time()-start_time:.1f}s, detected {plates_found} total plate reads.")