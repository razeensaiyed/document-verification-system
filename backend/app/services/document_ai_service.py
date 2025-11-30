# app/services/document_ai_service.py

from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions


class DocumentAIService:
    """Service for Google Cloud Document AI integration"""
    
    def __init__(self, project_id, location='us'):
        self.project_id = project_id
        self.location = location
        
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    def process_document(self, document_path, processor_id):
        """Process a document using a custom processor"""
        with open(document_path, "rb") as f:
            document_content = f.read()
        
        mime_type = "application/pdf" if document_path.endswith('.pdf') else "image/jpeg"
        
        processor_name = self.client.processor_path(
            self.project_id, self.location, processor_id
        )
        
        raw_document = documentai.RawDocument(content=document_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        
        result = self.client.process_document(request=request)
        
        return result.document
    
    def extract_signatures(self, document_path, signature_processor_id):
        """Extract signatures from a document"""
        document = self.process_document(document_path, signature_processor_id)
        
        signatures = []
        
        for entity in document.entities:
            bbox = self._get_normalized_bbox(entity)
            if bbox:
                signatures.append({
                    'confidence': float(entity.confidence),
                    'bounding_box': bbox,
                    'type': entity.type_
                })
        
        return signatures
    
    def extract_stamps(self, document_path, stamp_processor_id):
        """Extract stamps from a document"""
        document = self.process_document(document_path, stamp_processor_id)
        
        stamps = []
        
        for entity in document.entities:
            bbox = self._get_normalized_bbox(entity)
            if bbox:
                stamps.append({
                    'confidence': float(entity.confidence),
                    'bounding_box': bbox,
                    'type': entity.type_
                })
        
        return stamps
    
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
            'x': float(x_min),
            'y': float(y_min),
            'width': float(x_max - x_min),
            'height': float(y_max - y_min)
        }