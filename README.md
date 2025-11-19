# Document Verification System

Hello dear reader. I am a 19 year old Computer Engineering student based in Mumbai, India. I have created an AI-powered document verification system that compares identity documents by analyzing faces, signatures, and stamps using deep learning and computer vision techniques.

## Features

- **Face Detection & Comparison**: Using MTCNN for detection and FaceNet (InceptionResnetV1) for embedding-based comparison
- **Signature Extraction**: Using Google Cloud Document AI custom processors with pretrained foundation models
- **Stamp Extraction & Comparison**: Multi-method comparison using SSIM, ORB feature matching, and histogram correlation
- **Configurable Thresholds**: Adjustable similarity thresholds for different verification scenarios
- **Detailed Results**: JSON output with similarity scores, match verdicts, and detection confidence

## Project Structure
```
document-verification-system/
├── face_comparison/           # Face detection and comparison module
│   └── face_comparison.py    # Complete face verification pipeline
├── sign_and_stamp_comparison/ # Signature and stamp extraction/comparison
│   └── full_extraction_test.py  # Document AI integration and comparison
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

- **Python 3.11**
- **Deep Learning**: PyTorch, FaceNet-PyTorch
- **Cloud AI**: Google Cloud Document AI (Custom Extractors)
- **Computer Vision**: OpenCV, scikit-image
- **Scientific Computing**: NumPy, SciPy

## Prerequisites

- Python 3.11 or higher
- Google Cloud account with Document AI API enabled
- Service account with Document AI API User role
- Custom processors trained for signature and stamp extraction

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/document-verification-system.git
cd document-verification-system
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Google Cloud Credentials
```bash
# Set environment variable to your service account key
# Windows
setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\to\your\service-account-key.json"

# macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### 5. Configure Processor IDs

Update the following values in the scripts:

**In `sign_and_stamp_comparison/full_extraction_test.py`:**
```python
PROJECT_ID = "your-project-id"
SIGNATURE_PROCESSOR_ID = "your-signature-processor-id"
STAMP_PROCESSOR_ID = "your-stamp-processor-id"
```

## Usage

### Face Comparison
```bash
cd face_comparison
python face_comparison.py
```

**What it does:**
- Detects faces in two documents using MTCNN
- Extracts 512-dimensional embeddings using FaceNet
- Compares embeddings using cosine similarity
- Saves extracted face images and JSON results

**Configuration:**
```python
# In face_comparison.py
document1 = 'your_document1.jpg'
document2 = 'your_document2.jpg'
similarity_threshold = 0.6  # Adjust as needed
```

### Signature & Stamp Extraction and Comparison
```bash
cd sign_and_stamp_comparison
python full_extraction_test.py
```

**What it does:**
- Extracts signatures/stamps using Google Cloud Document AI
- Crops extracted regions from documents
- Compares using multiple techniques (SSIM, ORB, histogram)
- Saves extracted images and JSON results

**Configuration:**
```python
# In full_extraction_test.py
DOC1_PATH = "document1.jpg"
DOC2_PATH = "document2.jpg"
```

## Configuration

### Default Thresholds

- **Face Comparison**: 0.6 (similarity score)
  - 0.7+ = High confidence match
  - 0.6-0.7 = Moderate match
  - <0.6 = No match

- **Signature Comparison**: 0.5 (combined score)
  - More lenient due to signature variations

- **Stamp Comparison**: 0.6 (combined score)
  - Higher threshold for official stamps

### Adjusting Thresholds
```python
# Face comparison
compare_documents(doc1, doc2, similarity_threshold=0.7)  # Stricter

# Signature comparison
comparator.compare_signatures(sig1, sig2, threshold=0.4)  # More lenient

# Stamp comparison
comparator.compare_stamps(stamp1, stamp2, threshold=0.7)  # Stricter
```

## Output Format

### Face Comparison Results
```json
{
  "status": "success",
  "overall_match": true,
  "doc1_faces_count": 1,
  "doc2_faces_count": 1,
  "comparisons": [
    {
      "doc1_face_id": 1,
      "doc2_face_id": 1,
      "match": true,
      "similarity": 0.6179,
      "distance": 0.3821
    }
  ],
  "best_match": {
    "doc1_face_id": 1,
    "doc2_face_id": 1,
    "match": true,
    "similarity": 0.6179
  },
  "threshold_used": 0.6
}
```

### Signature/Stamp Comparison Results
```json
{
  "signatures": {
    "match": false,
    "confidence": 0.3839,
    "threshold": 0.5,
    "details": {
      "ssim": 0.0110,
      "histogram": 0.9433,
      "combined": 0.3839
    }
  },
  "stamps": {
    "match": true,
    "confidence": 0.7234,
    "threshold": 0.6
  }
}
```

## Testing

### Test with Sample Documents
```bash
# Positive test (same person)
python face_comparison.py  # Using matching ID documents

# Negative test (different people)
# Update script with different person's documents
python face_comparison.py
```

### Expected Results

- **Same person**: Similarity > 0.6, Match = True
- **Different people**: Similarity < 0.5, Match = False

## Project Status

- ✅ **Face extraction and comparison** - Complete
- ✅ **Signature extraction (Google Cloud AI)** - Complete
- ✅ **Stamp extraction and comparison** - Complete
- ✅ **Multi-method comparison algorithms** - Complete
- 🚧 **Web frontend interface** - In Development
- 🚧 **FastAPI backend** - Planned
- 🚧 **Batch processing** - Planned

## Security Notes

- **Never commit** Google Cloud service account JSON files
- Keep API keys and credentials in environment variables
- Use `.gitignore` to exclude sensitive files
- Consider using Google Secret Manager for production

## Contributing

This is a personal project for educational purposes. Feedback and suggestions are welcome!

## License

This project is for educational and research purposes.

## Author

**Razeen Husain Aejaz Husain Saiyed**

## Acknowledgments

- Google Cloud Document AI for signature/stamp extraction
- FaceNet-PyTorch for face recognition implementation
- MTCNN for robust face detection
- OpenCV and scikit-image for image processing

## Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This system is designed for document verification and identity validation purposes. Ensure compliance with local privacy and data protection regulations when handling identity documents.