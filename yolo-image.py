from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("traffic.jpg")

results[0].show()
