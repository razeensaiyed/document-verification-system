# app/services/image_annotation_service.py

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path


class ImageAnnotationService:
    """Service for annotating images with bounding boxes"""
    
    def __init__(self):
        self.colors = {
            'face': (0, 255, 0),      # Green
            'signature': (0, 0, 255),  # Blue (BGR format for OpenCV)
            'stamp': (0, 255, 255)     # Yellow
        }
        
        self.color_names = {
            'face': 'Green',
            'signature': 'Blue',
            'stamp': 'Yellow'
        }
    
    def draw_bounding_boxes(self, image_path, annotations, output_path):
        """
        Draw bounding boxes on an image
        
        Args:
            image_path: Path to original image
            annotations: List of dicts with 'type', 'bbox', 'confidence'
            output_path: Path to save annotated image
        
        Returns:
            Path to annotated image
        """
        # Read image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Draw each annotation
        for annotation in annotations:
            bbox = annotation['bbox']
            entity_type = annotation['type']
            confidence = annotation.get('confidence', 0)
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox['x'] * width)
            y1 = int(bbox['y'] * height)
            x2 = int((bbox['x'] + bbox['width']) * width)
            y2 = int((bbox['y'] + bbox['height']) * height)
            
            # Get color for this entity type
            color = self.colors.get(entity_type, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{entity_type.upper()} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw filled rectangle for label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness
            )
        
        # Save annotated image
        cv2.imwrite(str(output_path), image)
        
        return output_path
    
    def annotate_document(self, image_path, faces, signatures, stamps, output_dir):
        """
        Annotate a document with all detected entities
        
        Args:
            image_path: Path to original document
            faces: List of face detections with bounding boxes
            signatures: List of signature detections
            stamps: List of stamp detections
            output_dir: Directory to save annotated image
        
        Returns:
            Path to annotated image
        """
        annotations = []
        
        # Add face annotations
        for face in faces:
            # Convert face coordinates from pixel to normalized
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            coords = face['coordinates']  # [x1, y1, x2, y2] in pixels
            
            annotations.append({
                'type': 'face',
                'bbox': {
                    'x': coords[0] / width,
                    'y': coords[1] / height,
                    'width': (coords[2] - coords[0]) / width,
                    'height': (coords[3] - coords[1]) / height
                },
                'confidence': face['confidence'] / 100  # Convert from percentage
            })
        
        # Add signature annotations
        for signature in signatures:
            annotations.append({
                'type': 'signature',
                'bbox': signature['bounding_box'],
                'confidence': signature['confidence']
            })
        
        # Add stamp annotations
        for stamp in stamps:
            annotations.append({
                'type': 'stamp',
                'bbox': stamp['bounding_box'],
                'confidence': stamp['confidence']
            })
        
        # Generate output filename
        input_filename = Path(image_path).stem
        output_filename = f"{input_filename}_annotated.jpg"
        output_path = Path(output_dir) / output_filename
        
        # Draw all bounding boxes
        if annotations:
            self.draw_bounding_boxes(image_path, annotations, output_path)
            return str(output_path)
        else:
            # No annotations, just copy original
            import shutil
            shutil.copy(image_path, output_path)
            return str(output_path)
    
    def create_legend_image(self, output_path, width=300, height=150):
        """
        Create a legend image showing color codes
        
        Args:
            output_path: Path to save legend image
            width: Width of legend image
            height: Height of legend image
        """
        # Create white background
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw title
        cv2.putText(
            legend,
            "Detection Legend",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        y_offset = 60
        for entity_type, color in self.colors.items():
            # Draw colored box
            cv2.rectangle(legend, (10, y_offset), (40, y_offset + 20), color, -1)
            
            # Draw label
            cv2.putText(
                legend,
                f"= {entity_type.capitalize()}",
                (50, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            
            y_offset += 30
        
        # Save legend
        cv2.imwrite(str(output_path), legend)
        
        return output_path