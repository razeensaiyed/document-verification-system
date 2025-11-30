# app/services/signature_stamp_service.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class SignatureStampService:
    """Service for signature and stamp comparison"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_image_region(self, image_path, bbox):
        """Extract region from image using bounding box"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        
        x = int(bbox['x'] * width)
        y = int(bbox['y'] * height)
        w = int(bbox['width'] * width)
        h = int(bbox['height'] * height)
        
        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        region = image[y:y+h, x:x+w]
        return region
    
    def preprocess_for_comparison(self, image):
        """Preprocess image for comparison"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def resize_to_match(self, img1, img2):
        """Resize images to same dimensions"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
        return img1_resized, img2_resized
    
    def compare_using_ssim(self, img1, img2):
        """Compare images using SSIM"""
        proc1 = self.preprocess_for_comparison(img1)
        proc2 = self.preprocess_for_comparison(img2)
        
        proc1, proc2 = self.resize_to_match(proc1, proc2)
        
        score, _ = ssim(proc1, proc2, full=True)
        
        return float(score)
    
    def compare_using_histogram(self, img1, img2):
        """Compare images using histogram correlation"""
        proc1 = self.preprocess_for_comparison(img1)
        proc2 = self.preprocess_for_comparison(img2)
        
        proc1, proc2 = self.resize_to_match(proc1, proc2)
        
        hist1 = cv2.calcHist([proc1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([proc2], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(correlation)
    
    def compare_images(self, img1, img2):
        """Compare two images using multiple methods"""
        ssim_score = self.compare_using_ssim(img1, img2)
        hist_score = self.compare_using_histogram(img1, img2)
        
        # Combined score (weighted average)
        combined = ssim_score * 0.6 + hist_score * 0.4
        
        return {
            'ssim': ssim_score,
            'histogram': hist_score,
            'combined': combined
        }
    
    def compare_signatures(self, sig1_img, sig2_img, threshold=0.5):
        """Compare two signature images"""
        results = self.compare_images(sig1_img, sig2_img)
        
        similarity = results['combined']
        is_match = similarity >= threshold
        
        return {
            'match': bool(is_match),
            'confidence': float(round(similarity, 4)),
            'threshold_used': threshold,
            'details': {
                'ssim': float(round(results['ssim'], 4)),
                'histogram': float(round(results['histogram'], 4))
            }
        }
    
    def compare_stamps(self, stamp1_img, stamp2_img, threshold=0.6):
        """Compare two stamp images"""
        results = self.compare_images(stamp1_img, stamp2_img)
        
        similarity = results['combined']
        is_match = similarity >= threshold
        
        return {
            'match': bool(is_match),
            'confidence': float(round(similarity, 4)),
            'threshold_used': threshold,
            'details': {
                'ssim': float(round(results['ssim'], 4)),
                'histogram': float(round(results['histogram'], 4))
            }
        }