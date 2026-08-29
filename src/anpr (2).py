import easyocr
import cv2
import re
import numpy as np

class PlateReader:
    def __init__(self, gpu=False):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.last_seen_plate = None

    def preprocess_crop(self, crop):
        """Enhance contrast and binarize crop for sharper OCR read."""
        if crop.size == 0:
            return crop
            
        h, w = crop.shape[:2]
        if h < 80:
            scale = 80.0 / h
            crop = cv2.resize(crop, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_plate(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        
        box_h = y2 - y1
        box_w = x2 - x1
        if box_h <= 10 or box_w <= 10:
            return None, 0.0

        # Dynamic vehicle crop: capture lower half where plates sit
        crop_y1 = int(y1 + box_h * 0.30)
        crop = frame[crop_y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        # Attempt read on standard image first
        results = self.reader.readtext(crop)
        
        # Fallback to preprocessed image if standard read finds nothing
        if not results:
            processed = self.preprocess_crop(crop)
            results = self.reader.readtext(processed)
            
        if not results:
            return None, 0.0

        # Join OCR detections with a space to preserve two-part plate formats
        raw_text = ' '.join([r[1] for r in results]).upper()
        
        # Keep spaces along with letters and numbers
        clean_text = re.sub(r'[^A-Z0-9 ]', '', raw_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        confidence = float(np.mean([r[2] for r in results]))

        # Regex accepts single-word or spaced alphanumeric sequences (4 to 12 chars)
        match = re.search(r'\b[A-Z0-9]{2,4}\s?[A-Z0-9]{3,5}\b', clean_text)
        
        if match and confidence >= 0.25:
            plate_str = match.group()

            # Skip consecutive identical reads to prevent log spam
            if plate_str == self.last_seen_plate:
                return None, confidence
            
            self.last_seen_plate = plate_str
            return plate_str, confidence

        return None, confidence