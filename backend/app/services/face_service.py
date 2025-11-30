# app/services/face_service.py

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from numpy import asarray
import numpy as np
from scipy.spatial.distance import cosine


class FaceComparisonService:
    """Service for face detection and comparison"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=self.device
        )
        
        # Initialize InceptionResnetV1 for face embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def extract_face(self, image, box):
        """Extract face from image using bounding box"""
        pixels = asarray(image)
        x1, y1, x2, y2 = [int(c) for c in box]
        face_pixels = pixels[y1:y2, x1:x2]
        face_image = Image.fromarray(face_pixels)
        return face_image
    
    def get_face_embedding(self, face_image):
        """Convert face image to 512-dimensional embedding"""
        # To ensure RBG mode
        if face_image.mode == 'RGBA':
            face_image = face_image.convert('RGB')
        face_resized = face_image.resize((160, 160))
        
        face_tensor = torch.tensor(asarray(face_resized)).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.permute(2, 0, 1)
        face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.6):
        """Compare two face embeddings using cosine similarity"""
        cos_distance = cosine(embedding1, embedding2)
        similarity = 1 - cos_distance
        is_match = similarity >= threshold
        
        return {
            'match': bool(is_match),
            'similarity': float(round(similarity, 4)),
            'distance': float(round(cos_distance, 4)),
            'threshold_used': threshold
        }
    
    def extract_faces_from_document(self, image_path):
        """Extract all faces from a document"""
        try:
            image = Image.open(image_path)
            # adding to convert RGBA to RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image: {image_path}") from e
        
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return []
        
        faces_data = []
        
        for i, box in enumerate(boxes):
            coords = [round(c) for c in box.tolist()]
            confidence = round(probs[i] * 100, 2)
            
            face_img = self.extract_face(image, box)
            embedding = self.get_face_embedding(face_img)
            
            faces_data.append({
                'face_id': i + 1,
                'coordinates': coords,
                'confidence': float(confidence),
                'embedding': embedding
            })
        
        return faces_data
    
    def compare_faces(self, doc1_path, doc2_path, similarity_threshold=0.6):
        """Complete face comparison workflow"""
        # Extract faces from both documents
        doc1_faces = self.extract_faces_from_document(doc1_path)
        doc2_faces = self.extract_faces_from_document(doc2_path)
        
        if not doc1_faces or not doc2_faces:
            return {
                'status': 'error',
                'message': 'Could not detect faces in one or both documents',
                'doc1_faces_count': len(doc1_faces),
                'doc2_faces_count': len(doc2_faces)
            }
        
        # Compare all face combinations
        comparison_results = []
        
        for face1 in doc1_faces:
            for face2 in doc2_faces:
                result = self.compare_embeddings(
                    face1['embedding'],
                    face2['embedding'],
                    threshold=similarity_threshold
                )
                
                comparison_results.append({
                    'doc1_face_id': face1['face_id'],
                    'doc2_face_id': face2['face_id'],
                    'match': result['match'],
                    'similarity': result['similarity'],
                    'distance': result['distance']
                })
        
        # Determine overall match
        overall_match = any(res['match'] for res in comparison_results)
        best_match = max(comparison_results, key=lambda x: x['similarity'])
        
        return {
            'status': 'success',
            'overall_match': overall_match,
            'doc1_faces_count': len(doc1_faces),
            'doc2_faces_count': len(doc2_faces),
            'comparisons': comparison_results,
            'best_match': best_match,
            'threshold_used': similarity_threshold
        }