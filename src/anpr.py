import easyocr
import cv2
import re
import os
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=20):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.track_age = {}

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
        matched_indices = set()

        for bbox, text, conf in current_detections:
            matched_id = None
            best_iou = 0.0

            for track_id, track_data in self.tracks.items():
                iou = self._compute_iou(bbox, track_data['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    matched_id = track_id

            if matched_id is not None:
                matched_indices.add(matched_id)
                prev_conf = self.tracks[matched_id]['conf']
                prev_text = self.tracks[matched_id]['text']

                if text and conf >= prev_conf * 0.75:
                    final_text = text
                    final_conf = max(conf, prev_conf)
                else:
                    final_text = prev_text
                    final_conf = max(conf, prev_conf)

                new_tracks[matched_id] = {'bbox': bbox, 'text': final_text, 'conf': final_conf}
                self.track_age[matched_id] = 0
                if final_text:
                    updated_results.append((bbox, final_text, final_conf))
            else:
                new_id = self.next_id
                self.next_id += 1
                new_tracks[new_id] = {'bbox': bbox, 'text': text, 'conf': conf}
                self.track_age[new_id] = 0
                if text:
                    updated_results.append((bbox, text, conf))

        for track_id, track_data in self.tracks.items():
            if track_id not in matched_indices:
                self.track_age[track_id] = self.track_age.get(track_id, 0) + 1
                if self.track_age[track_id] <= self.max_age:
                    new_tracks[track_id] = track_data

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

        self.vehicle_detector = YOLO("yolov8l.pt")

        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        self.allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.iou_tracker = IOUTracker(iou_threshold=0.3, max_age=20)

    def is_valid_indian_registration(self, text):
        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
        return bool(re.match(pattern, text))

    def fix_character_confusions(self, text):
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean_text) < 7 or len(clean_text) > 11:
            return ""

        arr = list(clean_text)

        dict_num_to_letter_state = {'0': 'O', '1': 'I', '4': 'A', '8': 'B', '5': 'S', '2': 'Z'}
        for i in range(min(2, len(arr))):
            if arr[i].isdigit() and arr[i] in dict_num_to_letter_state:
                arr[i] = dict_num_to_letter_state[arr[i]]

        dict_letter_to_num_district = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}
        for i in range(2, min(4, len(arr))):
            if arr[i].isalpha() and arr[i] in dict_letter_to_num_district:
                arr[i] = dict_letter_to_num_district[arr[i]]

        for i in range(max(len(arr) - 4, 0), len(arr)):
            if arr[i].isalpha() and arr[i] in dict_letter_to_num_district:
                arr[i] = dict_letter_to_num_district[arr[i]]

        return "".join(arr)

    def preprocess_crop_aggressive(self, crop):
        """Even more aggressive preprocessing for difficult images"""
        if crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        
        if h < 80:
            scale = 200.0 / h
        elif h < 120:
            scale = 160.0 / h
        else:
            scale = max(140.0 / h, 1.0)
            
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        padded = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(padded)
        
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        
        _, binary = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

    def _preprocess_balanced(self, crop):
        """Balanced preprocessing"""
        h, w = crop.shape[:2]
        if h < 100:
            crop = cv2.resize(crop, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(padded)
        return enhanced

    def _preprocess_contrast_enhanced(self, crop):
        """High contrast preprocessing"""
        h, w = crop.shape[:2]
        if h < 100:
            crop = cv2.resize(crop, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(padded)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_vehicle_regions(self, frame):
        """Detect vehicles first, then search for plates inside them"""
        results = self.vehicle_detector(frame, device=self.device, verbose=False, conf=0.40)[0]
        
        vehicle_regions = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                
                if class_id in [2, 3, 5, 7] and conf >= 0.40:
                    vw = x2 - x1
                    vh = y2 - y1
                    if vw > 50 and vh > 50:
                        vehicle_regions.append((x1, y1, x2, y2))
        
        return vehicle_regions

    def extract_fallback_plate_boxes(self, frame):
        """Smart fallback detection - more coverage with validation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([10, 35, 40])
        upper_yellow = np.array([38, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fallback_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 350 < area < 60000:
                x, y, w, h = cv2.boundingRect(cnt)
                
                hull = cv2.convexHull(cnt)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                if solidity < 0.65:
                    continue
                
                aspect_ratio = w / float(h) if h > 0 else 0
                if 2.0 <= aspect_ratio <= 6.5:
                    fallback_boxes.append((x, y, x + w, y + h))

        return fallback_boxes

    def apply_nms(self, boxes, iou_thresh=0.35):
        """Softer NMS to keep more boxes"""
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

    def read_plate_text_aggressive(self, crop):
        """Multiple OCR attempts with strict validation"""
        attempts = [
            self.preprocess_crop_aggressive,
            self._preprocess_balanced,
            self._preprocess_contrast_enhanced,
        ]
        
        best_result = ("", 0.0)

        for preprocess_fn in attempts:
            try:
                processed = preprocess_fn(crop)
                
                ocr_results = self.reader.readtext(processed, allowlist=self.allowlist)
                
                if not ocr_results:
                    ocr_results = self.reader.readtext(crop, allowlist=self.allowlist)

                if ocr_results:
                    ocr_results = sorted(ocr_results, key=lambda r: r[0][0][0])
                    raw_text = "".join([res[1] for res in ocr_results])
                    confidences = [float(res[2]) for res in ocr_results]
                    avg_conf = float(np.mean(confidences))
                    corrected = self.fix_character_confusions(raw_text)

                    if self.is_valid_indian_registration(corrected):
                        if avg_conf > best_result[1]:
                            best_result = (corrected, avg_conf)
                        if avg_conf >= 0.60:
                            return corrected, avg_conf
            except:
                pass

        return best_result if best_result[0] else ("", 0.0)

    def extract_plate_from_frame(self, frame, frame_idx=0):
        """Balanced extraction - catches more cars with strict validation"""
        h, w = frame.shape[:2]
        candidate_boxes = []

        vehicle_regions = self.extract_vehicle_regions(frame)

        results = self.detector(frame, device=self.device, verbose=False, conf=0.20, iou=0.45)[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= 0.20:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                        candidate_boxes.append((x1, y1, x2, y2))

        fallback = self.extract_fallback_plate_boxes(frame)
        candidate_boxes.extend(fallback)

        for vx1, vy1, vx2, vy2 in vehicle_regions:
            vh = vy2 - vy1
            vw = vx2 - vx1
            
            search_y_start = int(vy1 + vh * 0.5)
            search_y_end = vy2
            
            vehicle_crop = frame[search_y_start:search_y_end, vx1:vx2]
            if vehicle_crop.size == 0:
                continue
                
            vehicle_hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            
            lower_yellow = np.array([10, 35, 40])
            upper_yellow = np.array([38, 255, 255])
            mask = cv2.inRange(vehicle_hsv, lower_yellow, upper_yellow)
            
            lower_white = np.array([0, 0, 120])
            upper_white = np.array([180, 60, 255])
            mask2 = cv2.inRange(vehicle_hsv, lower_white, upper_white)
            
            combined = cv2.bitwise_or(mask, mask2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 300 < area < 45000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    
                    hull = cv2.convexHull(cnt)
                    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                    if solidity < 0.68:
                        continue
                    
                    aspect = bw / float(bh) if bh > 0 else 0
                    if 2.0 <= aspect <= 6.5:
                        candidate_boxes.append((vx1 + x, search_y_start + y, vx1 + x + bw, search_y_start + y + bh))

        unique_boxes = self.apply_nms(candidate_boxes, iou_thresh=0.35)

        raw_frame_detections = []
        for x1, y1, x2, y2 in unique_boxes:
            box_w, box_h = x2 - x1, y2 - y1
            if box_h == 0 or box_w == 0:
                continue

            aspect_ratio = box_w / float(box_h)
            if aspect_ratio < 2.0 or aspect_ratio > 6.5:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            valid_text, avg_conf = self.read_plate_text_aggressive(crop)

            if valid_text and avg_conf >= 0.35:
                raw_frame_detections.append(((x1, y1, x2, y2), valid_text, avg_conf))

        tracked_results = self.iou_tracker.update(raw_frame_detections)
        return [(text, conf, bbox) for bbox, text, conf in tracked_results if text]