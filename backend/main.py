from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
from typing import List
import uuid

from app.services.face_service import FaceComparisonService
from app.services.document_ai_service import DocumentAIService
from app.services.signature_stamp_service import SignatureStampService
from app.services.image_annotation_service import ImageAnnotationService
from app.utils.file_handler import save_upload_file, cleanup_files

# Initialize FastAPI app
app = FastAPI(
    title="Document Verification API",
    description="AI-powered document verification using face, signature, and stamp comparison",
    version="1.0.0"
)

# Add CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_service = FaceComparisonService()
document_ai_service = DocumentAIService(
    project_id="ai-internship-100206",
    location="us"
)
signature_stamp_service = SignatureStampService()
annotation_service = ImageAnnotationService()

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
ANNOTATED_DIR = Path("annotated")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Document Verification API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "face_comparison": "operational",
            "document_ai": "operational",
            "signature_stamp_comparison": "operational"
        }
    }


@app.post("/api/compare")
async def compare_documents(
    document1: UploadFile = File(...),
    document2: UploadFile = File(...)
):
    """
    Compare two documents by analyzing faces, signatures, and stamps
    
    Args:
        document1: First document image file
        document2: Second document image file
    
    Returns:
        JSON with comparison results for faces, signatures, and stamps
    """
    session_id = str(uuid.uuid4())
    doc1_path = None
    doc2_path = None
    
    try:
        # Validate file types
        allowed_extensions = {".jpg", ".jpeg", ".png", ".pdf"}
        doc1_ext = Path(document1.filename).suffix.lower()
        doc2_ext = Path(document2.filename).suffix.lower()
        
        if doc1_ext not in allowed_extensions or doc2_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded files
        doc1_path = UPLOAD_DIR / f"{session_id}_doc1{doc1_ext}"
        doc2_path = UPLOAD_DIR / f"{session_id}_doc2{doc2_ext}"
        
        await save_upload_file(document1, doc1_path)
        await save_upload_file(document2, doc2_path)
        
        print(f"📁 Files saved: {doc1_path}, {doc2_path}")
        
        # Initialize results
        results = {
            "session_id": session_id,
            "document1_name": document1.filename,
            "document2_name": document2.filename,
            "faces": None,
            "signatures": None,
            "stamps": None,
            "overall_verdict": None
        }
        
        # 1. FACE COMPARISON
        print("👤 Starting face comparison...")
        try:
            face_results = face_service.compare_faces(str(doc1_path), str(doc2_path))
            results["faces"] = face_results
            print(f"✅ Face comparison complete: {face_results.get('overall_match', False)}")
        except Exception as e:
            print(f"⚠️ Face comparison failed: {e}")
            results["faces"] = {"error": str(e), "status": "failed"}
        
        # 2. SIGNATURE EXTRACTION AND COMPARISON
        print("✍️ Starting signature extraction...")
        try:
            # Extract signatures from both documents
            doc1_signatures = document_ai_service.extract_signatures(
                str(doc1_path),
                "7316c9d4b7355419"
            )
            doc2_signatures = document_ai_service.extract_signatures(
                str(doc2_path),
                "7316c9d4b7355419"
            )
            
            if doc1_signatures and doc2_signatures:
                # Compare ALL signature combinations
                signature_comparisons = []
                
                for idx1, sig1 in enumerate(doc1_signatures, 1):
                    sig1_img = signature_stamp_service.extract_image_region(
                        str(doc1_path),
                        sig1['bounding_box']
                    )
                    
                    for idx2, sig2 in enumerate(doc2_signatures, 1):
                        sig2_img = signature_stamp_service.extract_image_region(
                            str(doc2_path),
                            sig2['bounding_box']
                        )
                        
                        comparison = signature_stamp_service.compare_signatures(sig1_img, sig2_img, threshold=0.35)
                        signature_comparisons.append({
                            'doc1_signature_id': idx1,
                            'doc2_signature_id': idx2,
                            'match': comparison['match'],
                            'confidence': comparison['confidence'],
                            'details': comparison['details']
                        })
                        
                        match_status = "✅" if comparison['match'] else "❌"
                        print(f"  {match_status} Sig {idx1} vs Sig {idx2}: {comparison['confidence']:.4f}")
                
                # Determine if ANY signatures match
                any_match = any(comp['match'] for comp in signature_comparisons)
                best_sig_match = max(signature_comparisons, key=lambda x: x['confidence'])
                
                results["signatures"] = {
                    'status': 'success',
                    'overall_match': any_match,
                    'doc1_signatures_count': len(doc1_signatures),
                    'doc2_signatures_count': len(doc2_signatures),
                    'comparisons': signature_comparisons,
                    'best_match': best_sig_match,
                    'threshold_used': 0.5
                }
                print(f"✅ Signature comparison complete: {any_match}")
            else:
                results["signatures"] = {
                    "error": "Signatures not detected in one or both documents",
                    "status": "not_found",
                    "doc1_signatures_count": len(doc1_signatures) if doc1_signatures else 0,
                    "doc2_signatures_count": len(doc2_signatures) if doc2_signatures else 0
                }
                print("⚠️ Signatures not detected")
        except Exception as e:
            print(f"⚠️ Signature comparison failed: {e}")
            results["signatures"] = {"error": str(e), "status": "failed"}
        
        # 3. STAMP EXTRACTION AND COMPARISON
        print("📮 Starting stamp extraction...")
        try:
            # Extract stamps from both documents
            doc1_stamps = document_ai_service.extract_stamps(
                str(doc1_path),
                "35e7c9ec94e9868c"
            )
            doc2_stamps = document_ai_service.extract_stamps(
                str(doc2_path),
                "35e7c9ec94e9868c"
            )
            
            if doc1_stamps and doc2_stamps:
                # Compare ALL stamp combinations
                stamp_comparisons = []
                
                for idx1, stamp1 in enumerate(doc1_stamps, 1):
                    stamp1_img = signature_stamp_service.extract_image_region(
                        str(doc1_path),
                        stamp1['bounding_box']
                    )
                    
                    for idx2, stamp2 in enumerate(doc2_stamps, 1):
                        stamp2_img = signature_stamp_service.extract_image_region(
                            str(doc2_path),
                            stamp2['bounding_box']
                        )
                        
                        comparison = signature_stamp_service.compare_stamps(stamp1_img, stamp2_img)
                        stamp_comparisons.append({
                            'doc1_stamp_id': idx1,
                            'doc2_stamp_id': idx2,
                            'match': comparison['match'],
                            'confidence': comparison['confidence'],
                            'details': comparison['details']
                        })
                        
                        match_status = "✅" if comparison['match'] else "❌"
                        print(f"  {match_status} Stamp {idx1} vs Stamp {idx2}: {comparison['confidence']:.4f}")
                
                # Determine if ANY stamps match
                any_match = any(comp['match'] for comp in stamp_comparisons)
                best_stamp_match = max(stamp_comparisons, key=lambda x: x['confidence'])
                
                results["stamps"] = {
                    'status': 'success',
                    'overall_match': any_match,
                    'doc1_stamps_count': len(doc1_stamps),
                    'doc2_stamps_count': len(doc2_stamps),
                    'comparisons': stamp_comparisons,
                    'best_match': best_stamp_match,
                    'threshold_used': 0.6
                }
                print(f"✅ Stamp comparison complete: {any_match}")
            else:
                results["stamps"] = {
                    "error": "Stamps not detected in one or both documents",
                    "status": "not_found",
                    "doc1_stamps_count": len(doc1_stamps) if doc1_stamps else 0,
                    "doc2_stamps_count": len(doc2_stamps) if doc2_stamps else 0
                }
                print("⚠️ Stamps not detected")
        except Exception as e:
            print(f"⚠️ Stamp comparison failed: {e}")
            results["stamps"] = {"error": str(e), "status": "failed"}
        
        # 4. DETERMINE OVERALL VERDICT
        overall_match = determine_overall_verdict(results)
        results["overall_verdict"] = overall_match
        
        print(f"\n🎯 OVERALL VERDICT: {'MATCH' if overall_match else 'NO MATCH'}")
        
        # 5. CREATE ANNOTATED IMAGES
        print("\n🎨 Creating annotated images...")
        try:
            # Prepare annotations for document 1
            doc1_faces = face_results.get('doc1_faces_count', 0) > 0
            doc1_sigs = doc1_signatures if 'doc1_signatures' in locals() else []
            doc1_stamps_list = doc1_stamps if 'doc1_stamps' in locals() else []
            
            # Get face data for annotation
            doc1_face_data = []
            doc2_face_data = []
            
            if face_results.get('status') == 'success':
                # Re-extract faces to get coordinates (we need them for annotation)
                doc1_face_data = face_service.extract_faces_from_document(str(doc1_path))
                doc2_face_data = face_service.extract_faces_from_document(str(doc2_path))
            
            # Annotate document 1
            annotated_doc1_path = annotation_service.annotate_document(
                str(doc1_path),
                doc1_face_data,
                doc1_sigs,
                doc1_stamps_list,
                ANNOTATED_DIR
            )
            
            # Annotate document 2
            annotated_doc2_path = annotation_service.annotate_document(
                str(doc2_path),
                doc2_face_data,
                doc2_signatures if 'doc2_signatures' in locals() else [],
                doc2_stamps if 'doc2_stamps' in locals() else [],
                ANNOTATED_DIR
            )
            
            results["annotated_images"] = {
                "document1": Path(annotated_doc1_path).name,
                "document2": Path(annotated_doc2_path).name
            }
            
            print(f"✅ Annotated images saved:")
            print(f"   Doc1: {annotated_doc1_path}")
            print(f"   Doc2: {annotated_doc2_path}")
            
        except Exception as e:
            print(f"⚠️ Image annotation failed: {e}")
            results["annotated_images"] = {"error": str(e)}
        
        print(f"\n🎯 OVERALL VERDICT: {'MATCH' if overall_match else 'NO MATCH'}")
        
        return JSONResponse(content=results)
    
    except Exception as e:
        # Cleanup on error
        if doc1_path and doc1_path.exists():
            doc1_path.unlink()
        if doc2_path and doc2_path.exists():
            doc2_path.unlink()
        
        raise HTTPException(status_code=500, detail=str(e))


def determine_overall_verdict(results: dict) -> str:
    """
    Determine overall match verdict based on individual comparisons
    
    Logic:
    - If faces match: MATCH (faces are most reliable)
    - If faces don't match but signatures and stamps match: REVIEW REQUIRED
    - Otherwise: NO MATCH
    """
    face_match = False
    sig_match = False
    stamp_match = False
    
    # Check face match
    if results["faces"] and isinstance(results["faces"], dict):
        if "overall_match" in results["faces"]:
            face_match = results["faces"]["overall_match"]
    
    # Check signature match
    if results["signatures"] and isinstance(results["signatures"], dict):
        if "match" in results["signatures"]:
            sig_match = results["signatures"]["match"]
    
    # Check stamp match
    if results["stamps"] and isinstance(results["stamps"], dict):
        if "match" in results["stamps"]:
            stamp_match = results["stamps"]["match"]
    
    # Decision logic
    if face_match:
        return "MATCH"
    elif sig_match and stamp_match:
        return "REVIEW REQUIRED"
    else:
        return "NO MATCH"


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...)
):
    """
    Upload documents for later processing
    Returns upload IDs for each file
    """
    if len(files) != 2:
        raise HTTPException(
            status_code=400,
            detail="Exactly 2 documents required"
        )
    
    session_id = str(uuid.uuid4())
    uploaded_files = []
    
    for idx, file in enumerate(files, 1):
        file_ext = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{session_id}_doc{idx}{file_ext}"
        
        await save_upload_file(file, file_path)
        
        uploaded_files.append({
            "filename": file.filename,
            "file_id": f"{session_id}_doc{idx}",
            "size": file_path.stat().st_size
        })
    
    return {
        "session_id": session_id,
        "files": uploaded_files,
        "message": "Files uploaded successfully"
    }


@app.delete("/api/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up uploaded files for a session
    """
    try:
        cleanup_files(UPLOAD_DIR, session_id)
        cleanup_files(ANNOTATED_DIR, session_id)
        return {"message": "Cleanup successful", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/annotated/{filename}")
async def get_annotated_image(filename: str):
    """
    Retrieve an annotated image
    """
    file_path = ANNOTATED_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)