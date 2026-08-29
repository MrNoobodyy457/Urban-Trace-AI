import easyocr
import cv2
import re
import os
import numpy as np
from ultralytics import YOLO

class PlateReader:
    def __init__(self, model_path="license_plate_detector.pt", gpu=False):
        if os.path.exists(model_path):
            print(f"Loading license plate detector: {model_path}")
            self.detector = YOLO(model_path)
        else:
            print(f"Warning: {model_path} not found. Falling back to yolov8x.pt...")
            self.detector = YOLO("yolov8x.pt")

        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.best_reads = {}

    def fix_character_confusions(self, text):
        """Standardize common OCR letter/number confusions based on position."""
        text = list(re.sub(r'[^A-Z0-9]', '', text.upper()))
        if len(text) < 8:
            return "".join(text)

        # Fix State Code (First 2 chars must be letters, e.g., 'KL')
        dict_char_to_num = {'0': 'O', '1': 'I', '4': 'A', '8': 'B', '5': 'S'}
        dict_num_to_char = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}

        # Fix State Code (Indices 0, 1)
        for i in range(2):
            if text[i] in dict_char_to_num:
                text[i] = dict_char_to_num[text[i]]

        # Fix District Code (Indices 2, 3 must be numbers)
        for i in range(2, 4):
            if text[i] in dict_num_to_char:
                text[i] = dict_num_to_char[text[i]]

        # Fix Last 4 Digits (Must be numbers)
        for i in range(len(text) - 4, len(text)):
            if text[i] in dict_num_to_char:
                text[i] = dict_num_to_char[text[i]]

        return "".join(text)

    def preprocess_crop(self, crop):
        if crop.size == 0:
            return crop
        
        h, w = crop.shape[:2]
        if h < 100:
            scale = 120.0 / h
            crop = cv2.resize(crop, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Remove small dot separators (like the dots in KL.41.H.1538)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(morphed)

    def extract_plate_from_frame(self, frame):
        results = self.detector(frame, verbose=False)[0]
        detected_plates = []

        for box in results.boxes:
            if float(box.conf[0]) < 0.35:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = self.preprocess_crop(crop)
            ocr_results = self.reader.readtext(processed, allowlist=self.allowlist)

            if not ocr_results:
                ocr_results = self.reader.readtext(crop, allowlist=self.allowlist)

            for res in ocr_results:
                text_str = res[1]
                ocr_conf = float(res[2])

                if ocr_conf < 0.30:
                    continue

                # Apply character positional correction
                corrected_text = self.fix_character_confusions(text_str)

                # Validate against Indian License Plate Format: (State)(District)(Series)(Number)
                indian_plate_pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

                if re.match(indian_plate_pattern, corrected_text):
                    key = corrected_text[-4:]
                    prev_conf = self.best_reads.get(key, 0.0)

                    if ocr_conf > prev_conf:
                        self.best_reads[key] = ocr_conf
                        detected_plates.append((corrected_text, ocr_conf))

        return detected_plates