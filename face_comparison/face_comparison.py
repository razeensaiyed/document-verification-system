from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from numpy import asarray
import numpy as np
from scipy.spatial.distance import cosine

# Function to extract the face using the detected box
def extract_face(image, box):
    """Crops the image based on the bounding box coordinates (x_min, y_min, x_max, y_max)"""
    pixels = asarray(image)
    x1, y1, x2, y2 = [int(c) for c in box]
    face_pixels = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face_pixels)
    return face_image

# Function to get face embedding
def get_face_embedding(face_image, resnet):
    """
    Converts a face image into a 512-dimensional embedding vector
    Args:
        face_image: PIL Image of the cropped face
        resnet: Pre-trained InceptionResnetV1 model
    Returns:
        numpy array: 512-dimensional embedding vector
    """
    # Resize face to 160x160 (required by FaceNet)
    face_resized = face_image.resize((160, 160))
    
    # Convert to tensor and normalize to [-1, 1]
    face_tensor = torch.tensor(asarray(face_resized)).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    
    # Rearrange dimensions from (H, W, C) to (C, H, W)
    face_tensor = face_tensor.permute(2, 0, 1)
    
    # Add batch dimension (1, C, H, W)
    face_tensor = face_tensor.unsqueeze(0)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_tensor = face_tensor.to(device)
    
    # Get embedding (no gradient needed for inference)
    with torch.no_grad():
        embedding = resnet(face_tensor)
    
    # Convert to numpy and flatten
    return embedding.cpu().numpy().flatten()

# Function to compare two face embeddings
def compare_faces(embedding1, embedding2, threshold=0.6):
    """
    Compares two face embeddings using cosine similarity
    Args:
        embedding1: First face embedding (512-dim vector)
        embedding2: Second face embedding (512-dim vector)
        threshold: Similarity threshold for match (default: 0.6)
    Returns:
        dict: Contains 'match' (bool), 'similarity' (float), 'distance' (float)
    """
    # Calculate cosine distance (0 = identical, 2 = opposite)
    cos_distance = cosine(embedding1, embedding2)
    
    # Convert to similarity score (1 = identical, 0 = opposite)
    similarity = 1 - cos_distance
    
    # Determine if faces match based on threshold
    is_match = similarity >= threshold
    
    return {
        'match': is_match,
        'similarity': round(float(similarity), 4),
        'distance': round(float(cos_distance), 4),
        'threshold_used': threshold
    }

# Function to process a single document and extract face embeddings
def extract_faces_from_document(image_path, mtcnn, resnet):
    """
    Extracts all faces from a document and returns their embeddings
    Args:
        image_path: Path to the document image
        mtcnn: MTCNN face detector
        resnet: InceptionResnetV1 model for embeddings
    Returns:
        list: List of dicts containing face info, images, and embeddings
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File '{image_path}' not found.")
        return []
    
    # Detect faces
    boxes, probs = mtcnn.detect(image)
    
    if boxes is None:
        print(f"❌ No faces detected in {image_path}")
        return []
    
    faces_data = []
    print(f"✅ Detected {len(boxes)} face(s) in {image_path}")
    
    for i, box in enumerate(boxes):
        coords = [round(c) for c in box.tolist()]
        confidence = round(probs[i] * 100, 2)
        
        print(f"  Face {i+1}: Coordinates {coords}, Confidence: {confidence}%")
        
        # Extract face image
        face_img = extract_face(image, box)
        
        # Save extracted face for visual verification
        output_filename = f'{image_path.split(".")[0]}_face_{i+1}.jpg'
        face_img.save(output_filename)
        print(f"  💾 Saved extracted face: {output_filename}")
        
        # Get embedding
        embedding = get_face_embedding(face_img, resnet)
        
        faces_data.append({
            'face_id': i + 1,
            'coordinates': coords,
            'confidence': confidence,
            'face_image': face_img,
            'embedding': embedding
        })
    
    return faces_data

# Main comparison workflow
def compare_documents(doc1_path, doc2_path, similarity_threshold=0.6):
    """
    Complete workflow to compare faces between two documents
    Args:
        doc1_path: Path to first document
        doc2_path: Path to second document
        similarity_threshold: Threshold for considering faces as matching
    Returns:
        dict: Comparison results
    """
    print("=" * 70)
    print("FACE COMPARISON MODULE - INITIALIZING")
    print("=" * 70)
    
    # Initialize MTCNN for face detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
        device=device
    )
    
    # Initialize InceptionResnetV1 for face embeddings
    # Using 'vggface2' pretrained weights (trained on 3.3M faces)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("✅ Models loaded successfully\n")
    
    # Extract faces from both documents
    print(f"📄 Processing Document 1: {doc1_path}")
    doc1_faces = extract_faces_from_document(doc1_path, mtcnn, resnet)
    
    print(f"\n📄 Processing Document 2: {doc2_path}")
    doc2_faces = extract_faces_from_document(doc2_path, mtcnn, resnet)
    
    # Compare faces
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    if not doc1_faces or not doc2_faces:
        return {
            'status': 'error',
            'message': 'Could not detect faces in one or both documents',
            'doc1_faces_count': len(doc1_faces),
            'doc2_faces_count': len(doc2_faces)
        }
    
    # If multiple faces detected, compare all combinations
    comparison_results = []
    
    for face1 in doc1_faces:
        for face2 in doc2_faces:
            result = compare_faces(
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
            
            match_emoji = "✅" if result['match'] else "❌"
            print(f"{match_emoji} Doc1 Face {face1['face_id']} vs Doc2 Face {face2['face_id']}: "
                  f"Similarity = {result['similarity']}, Match = {result['match']}")
    
    # Determine overall match (if any pair matches)
    overall_match = any(res['match'] for res in comparison_results)
    best_match = max(comparison_results, key=lambda x: x['similarity'])
    
    print("\n" + "=" * 70)
    print(f"🎯 FINAL VERDICT: {'MATCH' if overall_match else 'NO MATCH'}")
    print(f"📊 Best Similarity Score: {best_match['similarity']} "
          f"(Face {best_match['doc1_face_id']} ↔ Face {best_match['doc2_face_id']})")
    print("=" * 70)
    
    return {
        'status': 'success',
        'overall_match': overall_match,
        'doc1_faces_count': len(doc1_faces),
        'doc2_faces_count': len(doc2_faces),
        'comparisons': comparison_results,
        'best_match': best_match,
        'threshold_used': similarity_threshold
    }

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    document1 = 'test_face_image.jpg'
    document2 = 'test_face_image_2.jpg'
    
    results = compare_documents(document1, document2, similarity_threshold=0.6)
    
    # Save results to JSON
    import json
    
    # Remove face_image objects (not JSON serializable) and convert bool types
    def clean_for_json(obj):
        if isinstance(obj, dict):
            # Remove PIL Image objects
            cleaned = {k: clean_for_json(v) for k, v in obj.items() if k != 'face_image'}
            return cleaned
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()  # Convert numpy types to Python types
        elif isinstance(obj, bool):
            return bool(obj)  # Ensure Python bool
        return obj
    
    json_safe_results = clean_for_json(results)
    
    with open('face_comparison_results.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    print("\n💾 Results saved to 'face_comparison_results.json'")
    
    print("\n✨ Comparison complete!")