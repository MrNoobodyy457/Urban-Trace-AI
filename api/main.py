import argparse
import time
import cv2
from src.detection import VehicleDetector
from src.anpr import PlateReader


def process_video(video_path: str, gpu: bool = False):
    detector = VehicleDetector()
    reader = PlateReader(gpu=gpu)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    plates_found = 0
    start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        detections = detector.detect_frame(frame)
        for det in detections:
            plate, conf = reader.extract_plate(frame, det["bbox"])
            if plate:
                plates_found += 1
                print(
                    f"Frame {frame_count}: {det['class_name']} "
                    f"plate={plate} conf={conf:.2f}"
                )

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(
                f"... {frame_count} frames processed, "
                f"{plates_found} plates found so far, {elapsed:.1f}s elapsed"
            )

    video.release()
    total_time = time.time() - start_time
    print(
        f"Done. {frame_count} frames in {total_time:.1f}s, "
        f"{plates_found} total plate reads"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Number Plate Recognition (ANPR) Runner"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/test_videos/sample.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU acceleration for OCR"
    )

    args = parser.parse_args()
    process_video(args.video, gpu=args.gpu)


if __name__ == "__main__":
    main()