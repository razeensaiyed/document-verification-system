# Document Verification System

AI-powered document verification system that compares identity documents by analyzing faces, signatures, and stamps using deep learning and computer vision.

## 🎯 Project Overview

This system verifies if two identity documents (ID cards, passports, admit cards) belong to the same person by analyzing:
- **Face Recognition** - Using FaceNet with 512-dimensional embeddings
- **Signature Comparison** - Using Google Cloud Document AI + multi-method comparison
- **Stamp Verification** - Using computer vision (SSIM, ORB, histogram correlation)

**Live Demo:** [Add deployment link when deployed]

## 📁 Project Structure
```
document-verification-system/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── services/          # Business logic
│   │   │   ├── face_service.py
│   │   │   ├── document_ai_service.py
│   │   │   ├── signature_stamp_service.py
│   │   │   └── image_annotation_service.py
│   │   └── utils/             # Helper functions
│   │       └── file_handler.py
│   └── main.py                # FastAPI application
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── App.css            # Styles
│   └── package.json
│
├── face_comparison/            # Original face comparison module
│   └── face_comparison.py
│
├── sign_and_stamp_comparison/  # Extraction testing
│   └── full_extraction_test.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

### **Backend**
- **Python 3.11**
- **FastAPI** - Modern REST API framework
- **FaceNet (PyTorch)** - Face recognition
- **Google Cloud Document AI** - Signature/stamp extraction
- **OpenCV** - Image processing
- **scikit-image** - SSIM comparison

### **Frontend**
- **React 18** - User interface
- **Axios** - API calls
- **Modern CSS** - Responsive design

### **AI/ML**
- **MTCNN** - Face detection
- **InceptionResnetV1** - Face embeddings
- **Document AI Custom Extractors** - Signature/stamp detection

## 🚀 Setup Instructions

### **Prerequisites**
- Python 3.11
- Node.js 16+
- Google Cloud account with Document AI enabled

### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/document-verification-system.git
cd document-verification-system
```

### **2. Backend Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set Google Cloud credentials
set GOOGLE_APPLICATION_CREDENTIALS="path\to\service-account-key.json"

# Run backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will run on: http://localhost:8000

### **3. Frontend Setup**
```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm start
```

Frontend will run on: http://localhost:3000

## 📖 Usage

1. **Start Backend** (Terminal 1)
2. **Start Frontend** (Terminal 2)
3. **Open browser** to http://localhost:3000
4. **Upload two documents** (drag & drop or click)
5. **Click "Compare Documents"**
6. **View results** with annotated images and similarity scores

## 🔧 Configuration

### **Google Cloud Setup**

1. Create project on Google Cloud Console
2. Enable Document AI API
3. Create custom processors:
   - Signature extractor
   - Stamp extractor
4. Create service account with "Document AI API User" role
5. Download service account key JSON

### **Update Processor IDs**

In `backend/main.py`:
```python
SIGNATURE_PROCESSOR_ID = "your-signature-processor-id"
STAMP_PROCESSOR_ID = "your-stamp-processor-id"
```

### **Adjust Thresholds**

- **Face similarity**: 0.6 (60% match)
- **Signature similarity**: 0.5 (50% match)
- **Stamp similarity**: 0.6 (60% match)

Modify in respective service files.

## 📊 API Endpoints

### **GET /**
Health check

### **GET /health**
Detailed service status

### **POST /api/compare**
Compare two documents
- **Input**: Two image files (multipart/form-data)
- **Output**: JSON with comparison results

### **GET /api/annotated/{filename}**
Retrieve annotated image

## 🎨 Features

✅ **Drag & Drop Upload** - Easy file upload  
✅ **Multi-Modal Verification** - Face + Signature + Stamp  
✅ **Visual Annotations** - Colored bounding boxes  
✅ **Detailed Results** - Similarity scores for each component  
✅ **Download Reports** - JSON export  
✅ **Responsive Design** - Works on desktop and mobile  

## 📈 Performance

- **Face Detection**: 99-100% confidence on clear images
- **Processing Time**: 5-10 seconds per document pair
- **Accuracy**: 
  - Same person: 0.60-0.85 similarity
  - Different people: 0.30-0.50 similarity

## 🔒 Security

- ✅ No credentials in repository
- ✅ Environment variables for API keys
- ✅ Temporary file cleanup
- ✅ Input validation
- ✅ CORS configured for development

## 🐛 Known Issues

1. **PNG Transparency**: Images with alpha channel automatically converted to RGB
2. **Signature Detection**: Some signatures detected as stamps (processor training needed)
3. **Processing Time**: First request slower (model loading)

## 📝 Future Enhancements

- [ ] Batch processing (multiple document pairs)
- [ ] PDF report generation
- [ ] User authentication
- [ ] Database storage for results
- [ ] Improved processor training
- [ ] Mobile app version

## 👤 Author

**Razeen Husain Aejaz Husain Saiyed**
- Computer Engineering Student
- Mumbai, India

## 🙏 Acknowledgments

- FaceNet-PyTorch for face recognition
- Google Cloud Document AI
- OpenCV and scikit-image communities

## 📄 License

This project is for educational purposes.

## 📞 Support

For issues or questions, open an issue on GitHub.

---

**Built with ❤️ using AI and Computer Vision**