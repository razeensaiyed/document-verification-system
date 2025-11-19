import os
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json

# Configuration
PROJECT_ID = "ai-internship-100206"
LOCATION = "us"
SIGNATURE_PROCESSOR_ID = "7316c9d4b7355419"
STAMP_PROCESSOR_ID = "35e7c9ec94e9868c"

class DocumentProcessor:
    """Handles extraction and comparison of signatures and stamps"""
    
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        
        # Initialize Document AI client
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    def extract_from_document(self, document_path, processor_id, entity_type):
        """
        Extract entities (signatures or stamps) from a document
        
        Args:
            document_path: Path to document image
            processor_id: Processor ID for extraction
            entity_type: 'signature' or 'stamp'
        
        Returns:
            List of extracted entities with bounding boxes
        """
        print(f"  📄 Processing: {document_path}")
        
        # Read document
        with open(document_path, "rb") as f:
            document_content = f.read()
        
        # Determine MIME type
        if document_path.lower().endswith('.pdf'):
            mime_type = "application/pdf"
        elif document_path.lower().endswith('.png'):
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        
        # Create processor path
        processor_name = self.client.processor_path(
            self.project_id, self.location, processor_id
        )
        
        # Process document
        raw_document = documentai.RawDocument(content=document_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        
        result = self.client.process_document(request=request)
        document = result.document
        
        # Extract entities
        entities = []
        for entity in document.entities:
            # Get bounding box
            bbox = self._get_normalized_bbox(entity)
            if bbox:
                entities.append({
                    'type': entity.type_,
                    'confidence': entity.confidence,
                    'bounding_box': bbox,
                    'text': entity.mention_text if hasattr(entity, 'mention_text') else None
                })
        
        print(f"     ✅ Found {len(entities)} {entity_type}(s)")
        return entities
    
    def _get_normalized_bbox(self, entity):
        """Extract normalized bounding box from entity"""
        if not entity.page_anchor.page_refs:
            return None
        
        page_ref = entity.page_anchor.page_refs[0]
        
        if not page_ref.bounding_poly.normalized_vertices:
            return None
        
        vertices = page_ref.bounding_poly.normalized_vertices
        
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return {
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
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
        
        # Add small padding
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
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def compare_images(self, img1, img2):
        """Compare two images using SSIM and histogram correlation"""
        # Preprocess
        proc1 = self.preprocess_for_comparison(img1)
        proc2 = self.preprocess_for_comparison(img2)
        
        # Resize to match
        h1, w1 = proc1.shape[:2]
        h2, w2 = proc2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        proc1 = cv2.resize(proc1, (target_w, target_h))
        proc2 = cv2.resize(proc2, (target_w, target_h))
        
        # Calculate SSIM
        ssim_score = ssim(proc1, proc2)
        
        # Calculate histogram correlation
        hist1 = cv2.calcHist([proc1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([proc2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Combined score (weighted average)
        combined = ssim_score * 0.6 + hist_corr * 0.4
        
        return {
            'ssim': float(ssim_score),
            'histogram': float(hist_corr),
            'combined': float(combined)
        }


def test_signature_extraction_and_comparison(doc1_path, doc2_path):
    """
    Complete test: Extract and compare signatures from two documents
    """
    print("\n" + "=" * 70)
    print("SIGNATURE EXTRACTION AND COMPARISON TEST")
    print("=" * 70)
    
    processor = DocumentProcessor(PROJECT_ID, LOCATION)
    
    # Extract signatures from both documents
    print("\n🔍 Extracting signatures...")
    doc1_signatures = processor.extract_from_document(doc1_path, SIGNATURE_PROCESSOR_ID, "signature")
    doc2_signatures = processor.extract_from_document(doc2_path, SIGNATURE_PROCESSOR_ID, "signature")
    
    if not doc1_signatures or not doc2_signatures:
        print("\n⚠️  No signatures found in one or both documents")
        return None
    
    # Extract image regions
    print("\n✂️  Extracting signature regions...")
    sig1_img = processor.extract_image_region(doc1_path, doc1_signatures[0]['bounding_box'])
    sig2_img = processor.extract_image_region(doc2_path, doc2_signatures[0]['bounding_box'])
    
    # Save extracted signatures for visual inspection
    cv2.imwrite('extracted_signature_1.jpg', sig1_img)
    cv2.imwrite('extracted_signature_2.jpg', sig2_img)
    print("     💾 Saved: extracted_signature_1.jpg")
    print("     💾 Saved: extracted_signature_2.jpg")
    
    # Compare signatures
    print("\n🔄 Comparing signatures...")
    comparison = processor.compare_images(sig1_img, sig2_img)
    
    # Determine match (threshold: 0.5)
    threshold = 0.5
    is_match = comparison['combined'] >= threshold
    
    result = {
        'match': is_match,
        'confidence': comparison['combined'],
        'threshold': threshold,
        'details': comparison
    }
    
    print("\n" + "=" * 70)
    print("SIGNATURE COMPARISON RESULTS")
    print("=" * 70)
    print(f"SSIM Score: {comparison['ssim']:.4f}")
    print(f"Histogram Correlation: {comparison['histogram']:.4f}")
    print(f"Combined Score: {comparison['combined']:.4f}")
    print(f"Threshold: {threshold}")
    print(f"{'✅ MATCH' if is_match else '❌ NO MATCH'}")
    print("=" * 70)
    
    return result


def test_stamp_extraction_and_comparison(doc1_path, doc2_path):
    """
    Complete test: Extract and compare stamps from two documents
    """
    print("\n" + "=" * 70)
    print("STAMP EXTRACTION AND COMPARISON TEST")
    print("=" * 70)
    
    processor = DocumentProcessor(PROJECT_ID, LOCATION)
    
    # Extract stamps from both documents
    print("\n🔍 Extracting stamps...")
    doc1_stamps = processor.extract_from_document(doc1_path, STAMP_PROCESSOR_ID, "stamp")
    doc2_stamps = processor.extract_from_document(doc2_path, STAMP_PROCESSOR_ID, "stamp")
    
    if not doc1_stamps or not doc2_stamps:
        print("\n⚠️  No stamps found in one or both documents")
        return None
    
    # Extract image regions
    print("\n✂️  Extracting stamp regions...")
    stamp1_img = processor.extract_image_region(doc1_path, doc1_stamps[0]['bounding_box'])
    stamp2_img = processor.extract_image_region(doc2_path, doc2_stamps[0]['bounding_box'])
    
    # Save extracted stamps for visual inspection
    cv2.imwrite('extracted_stamp_1.jpg', stamp1_img)
    cv2.imwrite('extracted_stamp_2.jpg', stamp2_img)
    print("     💾 Saved: extracted_stamp_1.jpg")
    print("     💾 Saved: extracted_stamp_2.jpg")
    
    # Compare stamps
    print("\n🔄 Comparing stamps...")
    comparison = processor.compare_images(stamp1_img, stamp2_img)
    
    # Determine match (threshold: 0.6 for stamps - usually more consistent)
    threshold = 0.6
    is_match = comparison['combined'] >= threshold
    
    result = {
        'match': is_match,
        'confidence': comparison['combined'],
        'threshold': threshold,
        'details': comparison
    }
    
    print("\n" + "=" * 70)
    print("STAMP COMPARISON RESULTS")
    print("=" * 70)
    print(f"SSIM Score: {comparison['ssim']:.4f}")
    print(f"Histogram Correlation: {comparison['histogram']:.4f}")
    print(f"Combined Score: {comparison['combined']:.4f}")
    print(f"Threshold: {threshold}")
    print(f"{'✅ MATCH' if is_match else '❌ NO MATCH'}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    print("🚀 Starting Full Extraction and Comparison Test\n")
    
    # Update these paths to your test documents
    DOC1_PATH = "test_face_image.jpg"  # Update with your document path
    DOC2_PATH = "test_face_image_2.jpg"  # Update with your document path
    
    # Check if files exist
    if not os.path.exists(DOC1_PATH):
        print(f"❌ Error: {DOC1_PATH} not found")
        exit(1)
    if not os.path.exists(DOC2_PATH):
        print(f"❌ Error: {DOC2_PATH} not found")
        exit(1)
    
    print(f"📄 Document 1: {DOC1_PATH}")
    print(f"📄 Document 2: {DOC2_PATH}")
    
    # Test signature extraction and comparison
    try:
        sig_result = test_signature_extraction_and_comparison(DOC1_PATH, DOC2_PATH)
    except Exception as e:
        print(f"\n❌ Signature test failed: {e}")
        sig_result = None
    
    # Test stamp extraction and comparison
    try:
        stamp_result = test_stamp_extraction_and_comparison(DOC1_PATH, DOC2_PATH)
    except Exception as e:
        print(f"\n❌ Stamp test failed: {e}")
        stamp_result = None
    
    # Save results
    results = {
        'signatures': sig_result,
        'stamps': stamp_result
    }
    
    with open('extraction_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to: extraction_comparison_results.json")
    print("\n✨ Test complete!")