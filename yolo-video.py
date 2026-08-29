import torch
from ultralytics import YOLO

# Verify CUDA
device = 0 if torch.cuda.is_available() else "cpu"

# Load model
model = YOLO("yolo11x.pt")

# COCO vehicle class IDs: bicycle, car, motorcycle, bus, truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# Run tracking filtered strictly to vehicles on GPU
results = model.track(
    source="traffic1.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    device=device,
    classes=VEHICLE_CLASSES,  # Filters out pedestrians, signs, animals, etc.
    save=True,
    conf=0.25,
    iou=0.7
)

print("Tracking complete! Filtered video saved under: runs/detect/track/")
