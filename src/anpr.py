import easyocr
import cv2
import re
import os
import torch
import numpy as np
from ultralytics import YOLO

class IOUTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, current_detections):
        updated_results = []
        new_tracks = {}

        for bbox, text, conf in current_detections:
            matched_id = None
            best_iou = 0.0

            for track_id, track_data in self.tracks.items():
                iou = self._compute_iou(bbox, track_data['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    matched_id = track_id

            if matched_id is not None:
                prev_conf = self.tracks[matched_id]['conf']
                prev_text = self.tracks[matched_id]['text']

                final_text = text if (text and conf >= prev_conf) else prev_text
                final_conf = max(conf, prev_conf)

                new_tracks[matched_id] = {'bbox': bbox, 'text': final_text, 'conf': final_conf}
                if final_text:
                    updated_results.append((bbox, final_text, final_conf))
            else:
                new_id = self.next_id
                self.next_id += 1
                new_tracks[new_id] = {'bbox': bbox, 'text': text, 'conf': conf}
                if text:
                    updated_results.append((bbox, text, conf))

        self.tracks = new_tracks
        return updated_results


class PlateReader:
    def __init__(self, model_path="license_plate_detector.pt", gpu=True):
        self.device = 'cuda' if (gpu and torch.cuda.is_available()) else 'cpu'
        print(f"Running ANPR pipeline on: {self.device.upper()}")

        if os.path.exists(model_path):
            self.detector = YOLO(model_path)
        else:
            self.detector = YOLO("yolov8x.pt")

        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        self.allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.iou_tracker = IOUTracker(iou_threshold=0.3)

    def is_valid_indian_registration(self, text):
        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
        return bool(re.match(pattern, text))

    def fix_character_confusions(self, text):
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean_text) < 7 or len(clean_text) > 11:
            return ""

        arr = list(clean_text)
        
        dict_char_to_num = {'0': 'O', '1': 'I', '4': 'A', '8': 'B', '5': 'S'}
        dict_num_to_char = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}

        for i in (0, 1):
            if i < len(arr) and arr[i] in dict_char_to_num:
                arr[i] = dict_char_to_num[arr[i]]

        for i in range(max(2, len(arr) - 4), len(arr)):
            if arr[i] in dict_num_to_char:
                arr[i] = dict_num_to_char[arr[i]]

        return "".join(arr)

    def preprocess_crop(self, crop):
        if crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        if h < 100:
            scale = 120.0 / h
            crop = cv2.resize(crop, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def extract_fallback_plate_boxes(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow plates mask
        lower_yellow = np.array([10, 40, 40])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White plates mask
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fallback_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 600 < area < 45000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 1.4 <= aspect_ratio <= 5.8:
                    fallback_boxes.append((x, y, x + w, y + h))

        return fallback_boxes

    def apply_nms(self, boxes, iou_thresh=0.4):
        """Non-Maximum Suppression: Removes duplicate bounding boxes pointing to the same plate."""
        if len(boxes) == 0:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.arange(len(boxes))

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return [boxes[k] for k in keep]

    def extract_plate_from_frame(self, frame, frame_idx=0):
        results = self.detector(frame, device=self.device, verbose=False)[0]
        candidate_boxes = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if float(box.conf[0]) >= 0.12:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    candidate_boxes.append((x1, y1, x2, y2))

        # Add fallback color candidate boxes
        candidate_boxes.extend(self.extract_fallback_plate_boxes(frame))

        # REMOVE DUPLICATE BOXES: Apply NMS across all collected bounding boxes
        unique_boxes = self.apply_nms(candidate_boxes, iou_thresh=0.35)

        raw_frame_detections = []
        for x1, y1, x2, y2 in unique_boxes:
            box_w, box_h = x2 - x1, y2 - y1

            if box_h == 0 or box_w == 0:
                continue

            aspect_ratio = box_w / float(box_h)
            if aspect_ratio < 1.3 or aspect_ratio > 6.0:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = self.preprocess_crop(crop)
            ocr_results = self.reader.readtext(processed, allowlist=self.allowlist)

            if not ocr_results:
                ocr_results = self.reader.readtext(crop, allowlist=self.allowlist)

            valid_text, avg_conf = "", 0.0
            if ocr_results:
                ocr_results = sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
                raw_text = "".join([res[1] for res in ocr_results])
                avg_conf = float(np.mean([float(res[2]) for res in ocr_results]))
                corrected = self.fix_character_confusions(raw_text)

                if self.is_valid_indian_registration(corrected) and avg_conf >= 0.18:
                    valid_text = corrected

            raw_frame_detections.append(((x1, y1, x2, y2), valid_text, avg_conf))

        tracked_results = self.iou_tracker.update(raw_frame_detections)
        return [(text, conf, bbox) for bbox, text, conf in tracked_results if text]
