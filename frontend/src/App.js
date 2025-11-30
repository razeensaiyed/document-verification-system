import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [document1, setDocument1] = useState(null);
  const [document2, setDocument2] = useState(null);
  const [preview1, setPreview1] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e, docNumber) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf'];
      if (!validTypes.includes(file.type)) {
        alert('Please upload JPG, PNG, or PDF files only');
        return;
      }

      // Set file and preview
      if (docNumber === 1) {
        setDocument1(file);
        setPreview1(URL.createObjectURL(file));
      } else {
        setDocument2(file);
        setPreview2(URL.createObjectURL(file));
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e, docNumber) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file) {
      const event = { target: { files: [file] } };
      handleFileChange(event, docNumber);
    }
  };

  const handleCompare = async () => {
    if (!document1 || !document2) {
      alert('Please upload both documents');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('document1', document1);
    formData.append('document2', document2);

    try {
      const response = await axios.post(`${API_URL}/api/compare`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds
      });

      setResults(response.data);
    } catch (err) {
      console.error('Error:', err);
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again.');
      } else if (err.response) {
        setError(`Error: ${err.response.status} - ${err.response.data.detail || 'Unknown error'}`);
      } else if (err.request) {
        setError('Cannot connect to backend server. Please ensure it is running on http://localhost:8000');
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setDocument1(null);
    setDocument2(null);
    setPreview1(null);
    setPreview2(null);
    setResults(null);
    setError(null);
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `verification_results_${results.session_id || 'unknown'}.json`;
    link.click();
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🔍 DOCUMENT VERIFIER</h1>
          <p className="subtitle">Verify Identity Documents</p>
        </header>

        {!results && (
          <>
            <div className="upload-section">
              <div className="upload-grid">
                <div className="upload-box">
                  <h3>📄 DOCUMENT 1</h3>
                  <div
                    className="drop-zone"
                    onDragOver={handleDragOver}
                    onDrop={(e) => handleDrop(e, 1)}
                    onClick={() => document.getElementById('file1').click()}
                  >
                    {preview1 ? (
                      <img src={preview1} alt="Document 1" className="preview-image" />
                    ) : (
                      <div className="drop-zone-content">
                        <div className="upload-icon">📁</div>
                        <p>Drag & Drop or Click to Upload</p>
                        <span className="file-types">JPG, PNG, PDF</span>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    id="file1"
                    accept=".jpg,.jpeg,.png,.pdf"
                    onChange={(e) => handleFileChange(e, 1)}
                    style={{ display: 'none' }}
                  />
                  {document1 && (
                    <div className="file-info">
                      ✅ {document1.name}
                    </div>
                  )}
                </div>

                <div className="upload-box">
                  <h3>📄 DOCUMENT 2</h3>
                  <div
                    className="drop-zone"
                    onDragOver={handleDragOver}
                    onDrop={(e) => handleDrop(e, 2)}
                    onClick={() => document.getElementById('file2').click()}
                  >
                    {preview2 ? (
                      <img src={preview2} alt="Document 2" className="preview-image" />
                    ) : (
                      <div className="drop-zone-content">
                        <div className="upload-icon">📁</div>
                        <p>Drag & Drop or Click to Upload</p>
                        <span className="file-types">JPG, PNG, PDF</span>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    id="file2"
                    accept=".jpg,.jpeg,.png,.pdf"
                    onChange={(e) => handleFileChange(e, 2)}
                    style={{ display: 'none' }}
                  />
                  {document2 && (
                    <div className="file-info">
                      ✅ {document2.name}
                    </div>
                  )}
                </div>
              </div>

              <button
                className="compare-button"
                onClick={handleCompare}
                disabled={!document1 || !document2 || loading}
              >
                {loading ? '🔄 Processing...' : '🔍 COMPARE DOCUMENTS'}
              </button>
            </div>

            {error && (
              <div className="error-box">
                <h3>❌ Error</h3>
                <p>{error}</p>
              </div>
            )}
          </>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Analyzing documents...</p>
            <p className="loading-subtext">This may take 10-15 seconds</p>
          </div>
        )}

        {results && !loading && (
          <div className="results-section">
            <div className={`verdict-box verdict-${results.overall_verdict.toLowerCase().replace(' ', '-')}`}>
              <h2>
                {results.overall_verdict === 'MATCH' && '✅ MATCH'}
                {results.overall_verdict === 'NO MATCH' && '❌ NO MATCH'}
                {results.overall_verdict === 'REVIEW REQUIRED' && '⚠️ REVIEW REQUIRED'}
              </h2>
              <p>
                {results.overall_verdict === 'MATCH' && 'The documents likely belong to the same person'}
                {results.overall_verdict === 'NO MATCH' && 'The documents do not appear to belong to the same person'}
                {results.overall_verdict === 'REVIEW REQUIRED' && 'Manual review recommended'}
              </p>
            </div>

            <div className="annotated-section">
              <h3>🎨 Annotated Documents</h3>
              <p className="legend">
                <span className="legend-item"><span className="legend-color green"></span> Faces</span>
                <span className="legend-item"><span className="legend-color blue"></span> Signatures</span>
                <span className="legend-item"><span className="legend-color yellow"></span> Stamps</span>
              </p>

              <div className="annotated-grid">
                <div className="annotated-box">
                  <h4>Document 1 (Annotated)</h4>
                  {results.annotated_images?.document1 ? (
                    <img
                      src={`${API_URL}/api/annotated/${results.annotated_images.document1.split(/[/\\]/).pop()}`}
                      alt="Annotated Document 1"
                      className="annotated-image"
                    />
                  ) : (
                    <p>No annotated image available</p>
                  )}
                </div>
                <div className="annotated-box">
                  <h4>Document 2 (Annotated)</h4>
                  {results.annotated_images?.document2 ? (
                    <img
                      src={`${API_URL}/api/annotated/${results.annotated_images.document2.split(/[/\\]/).pop()}`}
                      alt="Annotated Document 2"
                      className="annotated-image"
                    />
                  ) : (
                    <p>No annotated image available</p>
                  )}
                </div>
              </div>
            </div>

            <div className="details-section">
              <h3>📋 Extraction Summary</h3>
              <div className="summary-grid">
                <div className="summary-card">
                  <h4>Document 1</h4>
                  <p>👤 Faces: {results.faces?.doc1_faces_count || 0}</p>
                </div>
                <div className="summary-card">
                  <h4>Document 2</h4>
                  <p>👤 Faces: {results.faces?.doc2_faces_count || 0}</p>
                </div>
              </div>

              <h3>🔬 Detailed Results</h3>

              {/* Face Comparison */}
              <div className="result-card">
                <h4>👤 Face Comparison</h4>
                {results.faces?.status === 'success' ? (
                  <>
                    <p className={results.faces.overall_match ? 'match-text' : 'no-match-text'}>
                      {results.faces.overall_match ? '✅ Faces Match' : '❌ Faces Do Not Match'}
                    </p>
                    {results.faces.comparisons?.map((comp, idx) => (
                      <div key={idx} className="comparison-detail">
                        <span>Face {comp.doc1_face_id} ↔ Face {comp.doc2_face_id}</span>
                        <span className="similarity-score">Similarity: {comp.similarity.toFixed(4)}</span>
                      </div>
                    ))}
                  </>
                ) : (
                  <p className="warning-text">⚠️ {results.faces?.message || 'Face comparison unavailable'}</p>
                )}
              </div>

              {/* Signature Comparison */}
              <div className="result-card">
                <h4>✍️ Signature Comparison</h4>
                {results.signatures?.error ? (
                  <p className="warning-text">⚠️ {results.signatures.error}</p>
                ) : results.signatures?.status === 'success' ? (
                  <>
                    <p className={results.signatures.overall_match ? 'match-text' : 'no-match-text'}>
                      {results.signatures.overall_match ? '✅ At Least One Signature Matches' : '❌ No Signatures Match'}
                    </p>
                    <p style={{ marginTop: '0.5rem', color: '#4a5568' }}>
                      Document 1: {results.signatures.doc1_signatures_count} signature(s) |
                      Document 2: {results.signatures.doc2_signatures_count} signature(s)
                    </p>

                    <div style={{ marginTop: '1rem' }}>
                      <strong>Comparison Details:</strong>
                      {results.signatures.comparisons?.map((comp, idx) => (
                        <div key={idx} className="comparison-detail">
                          <span>
                            Signature {comp.doc1_signature_id} ↔ Signature {comp.doc2_signature_id}
                            {comp.match ? ' ✅' : ' ❌'}
                          </span>
                          <span className="similarity-score">
                            {comp.confidence.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>

                    {results.signatures.best_match && (
                      <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#edf2f7', borderRadius: '8px' }}>
                        <strong>Best Match:</strong> Signature {results.signatures.best_match.doc1_signature_id} ↔
                        Signature {results.signatures.best_match.doc2_signature_id}
                        ({results.signatures.best_match.confidence.toFixed(4)})
                      </div>
                    )}
                  </>
                ) : (
                  <p className="warning-text">⚠️ No signature data available</p>
                )}
              </div>

              {/* Stamp Comparison */}
              <div className="result-card">
                <h4>📮 Stamp Comparison</h4>
                {results.stamps?.error ? (
                  <p className="warning-text">⚠️ {results.stamps.error}</p>
                ) : results.stamps?.status === 'success' ? (
                  <>
                    <p className={results.stamps.overall_match ? 'match-text' : 'no-match-text'}>
                      {results.stamps.overall_match ? '✅ At Least One Stamp Matches' : '❌ No Stamps Match'}
                    </p>
                    <p style={{ marginTop: '0.5rem', color: '#4a5568' }}>
                      Document 1: {results.stamps.doc1_stamps_count} stamp(s) |
                      Document 2: {results.stamps.doc2_stamps_count} stamp(s)
                    </p>

                    <div style={{ marginTop: '1rem' }}>
                      <strong>Comparison Details:</strong>
                      {results.stamps.comparisons?.map((comp, idx) => (
                        <div key={idx} className="comparison-detail">
                          <span>
                            Stamp {comp.doc1_stamp_id} ↔ Stamp {comp.doc2_stamp_id}
                            {comp.match ? ' ✅' : ' ❌'}
                          </span>
                          <span className="similarity-score">
                            {comp.confidence.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>

                    {results.stamps.best_match && (
                      <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#edf2f7', borderRadius: '8px' }}>
                        <strong>Best Match:</strong> Stamp {results.stamps.best_match.doc1_stamp_id} ↔
                        Stamp {results.stamps.best_match.doc2_stamp_id}
                        ({results.stamps.best_match.confidence.toFixed(4)})
                      </div>
                    )}
                  </>
                ) : (
                  <p className="warning-text">⚠️ No stamp data available</p>
                )}
              </div>
            </div> {/* END OF DETAILS SECTION (Fixed) */}

            <div className="action-buttons">
              <button className="download-button" onClick={downloadResults}>
                📥 Download Results
              </button>
              <button className="reset-button" onClick={handleReset}>
                🔄 Compare New Documents
              </button>
            </div>
          </div> /* END OF RESULTS SECTION (Fixed) */
        )}

        <footer className="footer">
          <p>RAZEEN SAIYED - Document Verification System</p>
          <p className="footer-small">Powered by FaceNet, Google Cloud Document AI, and Computer Vision</p>
        </footer>
      </div>
    </div>
  );
}

export default App;