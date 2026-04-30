import { useState, useEffect } from 'react'
import axios from 'axios'
import { Upload, History, TrendingUp, Music, BarChart3, Star, Trash2, Cpu } from 'lucide-react'

const MODELS = [
  { id: 'CNN', name: 'CNN (Deep Learning)', genreAcc: '90.3%', emotionAcc: '92.1%' },
  { id: 'CRNN', name: 'CRNN (CNN + LSTM)', genreAcc: '88.2%', emotionAcc: '90.3%' },
  { id: 'LSTM', name: 'LSTM (Sequential)', genreAcc: '88.0%', emotionAcc: '89.0%' },
  { id: 'SVM', name: 'SVM (Support Vector)', genreAcc: '79.6%', emotionAcc: '82.7%' },
  { id: 'GMM', name: 'GMM (Gaussian Mixture)', genreAcc: '71.4%', emotionAcc: '75.3%' },
  { id: 'HMM', name: 'HMM (Hidden Markov)', genreAcc: '70.0%', emotionAcc: '72.0%' }
];

const EMOTION_COLORS = {
  happy: '#f1c40f',
  sad: '#3498db',
  angry: '#e74c3c',
  calm: '#2ecc71'
};

export default function Dashboard() {
  const [file, setFile] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const [selectedModel, setSelectedModel] = useState('CNN')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      const res = await axios.get('http://localhost:3000/api/history')
      setHistory(res.data)
    } catch (err) {
      console.error("Failed to fetch history", err)
    }
  }

  const handleDelete = async (id) => {
    try {
      await axios.delete(`http://localhost:3000/api/history/${id}`)
      fetchHistory()
    } catch (err) {
      console.error("Delete failed", err)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true)
    else if (e.type === "dragleave") setDragActive(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
      setPrediction(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setPrediction(null)

    const formData = new FormData()
    formData.append('audio', file)
    formData.append('model', selectedModel)

    try {
      const res = await axios.post('http://localhost:3000/api/classify', formData)
      setPrediction(res.data)
      fetchHistory()
    } catch (err) {
      setError(err.response?.data?.details || "Classification failed. Check backend logs.")
    } finally {
      setLoading(false)
    }
  }

  // Stats logic
  const totalScans = history.length;
  const topGenre = history.length > 0
    ? history.reduce((acc, curr) => {
      if(curr.genre) acc[curr.genre] = (acc[curr.genre] || 0) + 1;
      return acc;
    }, {})
    : null;
  const favoredGenre = topGenre && Object.keys(topGenre).length > 0
    ? Object.keys(topGenre).reduce((a, b) => topGenre[a] > topGenre[b] ? a : b)
    : "None";

  return (
    <div style={{ width: '100%', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ textAlign: 'left', marginBottom: '2rem' }}>
        <h1 style={{ margin: 0 }}>Music Intelligence</h1>
        <p style={{ opacity: 0.7, marginTop: '5px' }}>Upload an audio file to classify its genre and detect emotion.</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <BarChart3 size={20} color="#4facfe" />
          <div style={{ fontSize: '0.8em', opacity: 0.6, marginTop: '5px' }}>Total Analyzed</div>
          <div className="stat-value">{totalScans}</div>
        </div>
        <div className="stat-card">
          <TrendingUp size={20} color="#4facfe" />
          <div style={{ fontSize: '0.8em', opacity: 0.6, marginTop: '5px' }}>Top Genre</div>
          <div className="stat-value" style={{ textTransform: 'capitalize' }}>{favoredGenre}</div>
        </div>
        <div className="stat-card">
          <Star size={20} color="#4facfe" />
          <div style={{ fontSize: '0.8em', opacity: 0.6, marginTop: '5px' }}>Selected Model</div>
          <div className="stat-value" style={{ fontSize: '1.2em' }}>{selectedModel}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
        <div className="glass-card" style={{ maxWidth: 'none', display: 'flex', flexDirection: 'column', gap: '15px' }}>
          
          {/* Model Selection Dropdown */}
          <div style={{ textAlign: 'left' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontWeight: 'bold' }}>
              <Cpu size={18} /> Select AI Model
            </label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.2)',
                background: 'rgba(0,0,0,0.5)', color: '#fff', fontSize: '1rem', outline: 'none'
              }}
            >
              {MODELS.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} — (Genre: {m.genreAcc} | Emotion: {m.emotionAcc})
                </option>
              ))}
            </select>
          </div>

          <h3 style={{ marginTop: '10px', textAlign: 'left', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Upload size={20} /> New Analysis
          </h3>
          <div
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragOver={handleDrag} onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <input type="file" id="fileInput" style={{ display: 'none' }} onChange={(e) => setFile(e.target.files[0])} accept="audio/*" />
            {file ? (
              <div><Music size={32} /><p>{file.name}</p></div>
            ) : (
              <p>Drop audio file to classify</p>
            )}
          </div>
          {error && <p style={{ color: '#ff6b6b' }}>{error}</p>}
          <button onClick={handleUpload} disabled={!file || loading} style={{ width: '100%', marginTop: '10px' }}>
            {loading ? 'Analyzing Neural Patterns...' : 'Start Classification'}
          </button>

          {prediction && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '20px' }}>
              <h3 style={{ margin: 0, textAlign: 'center' }}>Analysis Results</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                
                {/* Genre Panel */}
                <div className="result-panel" style={{ textAlign: 'center', padding: '25px', background: 'linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(79, 172, 254, 0.1) 100%)', borderRadius: '15px', border: '1px solid rgba(79, 172, 254, 0.2)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
                  <Music size={32} color="#4facfe" style={{ margin: '0 auto 10px auto' }} />
                  <div style={{ opacity: 0.7, fontSize: '0.9em', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '10px' }}>Music Genre</div>
                  <div className="genre-title" style={{ fontSize: '2em', margin: '0 0 15px 0', textTransform: 'capitalize', color: '#fff', textShadow: '0 2px 10px rgba(79,172,254,0.5)' }}>
                    {prediction.genre}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85em', marginBottom: '8px' }}>
                    <span style={{ opacity: 0.6 }}>Confidence Level</span>
                    <span style={{ fontWeight: 'bold' }}>{(prediction.genre_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="confidence-bar" style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div className="confidence-fill" style={{ width: `${prediction.genre_confidence * 100}%`, height: '100%', background: 'linear-gradient(90deg, #00f2fe 0%, #4facfe 100%)', transition: 'width 1s ease-out' }}></div>
                  </div>
                </div>
                
                {/* Emotion Panel */}
                <div className="result-panel" style={{ textAlign: 'center', padding: '25px', background: `linear-gradient(145deg, rgba(255,255,255,0.05) 0%, ${EMOTION_COLORS[prediction.emotion] || '#4facfe'}20 100%)`, borderRadius: '15px', border: `1px solid ${EMOTION_COLORS[prediction.emotion] || '#4facfe'}40`, boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
                  <Star size={32} color={EMOTION_COLORS[prediction.emotion] || '#fff'} style={{ margin: '0 auto 10px auto' }} />
                  <div style={{ opacity: 0.7, fontSize: '0.9em', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '10px' }}>Detected Emotion</div>
                  <div className="genre-title" style={{ fontSize: '2em', margin: '0 0 15px 0', textTransform: 'capitalize', color: EMOTION_COLORS[prediction.emotion] || '#fff', textShadow: `0 2px 10px ${EMOTION_COLORS[prediction.emotion] || '#fff'}80` }}>
                    {prediction.emotion}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85em', marginBottom: '8px' }}>
                    <span style={{ opacity: 0.6 }}>Confidence Level</span>
                    <span style={{ fontWeight: 'bold' }}>{(prediction.emotion_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="confidence-bar" style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div className="confidence-fill" style={{ width: `${prediction.emotion_confidence * 100}%`, height: '100%', background: EMOTION_COLORS[prediction.emotion] || '#4facfe', transition: 'width 1s ease-out' }}></div>
                  </div>
                </div>
                
              </div>
            </div>
          )}
        </div>

        <div>
          <h3 style={{ marginTop: 0, textAlign: 'left', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <History size={20} /> Recent History
          </h3>
          <div style={{ maxHeight: '600px', overflowY: 'auto', paddingRight: '10px' }}>
            {history.map(item => (
              <div key={item.id} className="history-item" style={{ marginBottom: '10px', padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', position: 'relative', textAlign: 'left' }}>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '5px' }}>
                  <span style={{ textTransform: 'capitalize', fontWeight: 'bold', color: '#4facfe' }}>{item.genre}</span>
                  <span style={{ opacity: 0.5 }}>|</span>
                  <span style={{ textTransform: 'capitalize', fontWeight: 'bold', color: EMOTION_COLORS[item.emotion] || '#fff' }}>{item.emotion}</span>
                </div>
                <div style={{ fontSize: '0.8em', opacity: 0.7, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', marginBottom: '5px' }}>
                  {item.filename}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7em', opacity: 0.5 }}>
                  <span>{item.model_used || 'CNN'}</span>
                  <span>{new Date(item.timestamp).toLocaleString()}</span>
                </div>
                <button
                  onClick={() => handleDelete(item.id)}
                  style={{ position: 'absolute', right: '10px', top: '10px', background: 'transparent', padding: '5px', margin: 0 }}
                >
                  <Trash2 size={14} color="#ff6b6b" />
                </button>
              </div>
            ))}
            {history.length === 0 && <p style={{ opacity: 0.5 }}>No records found.</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
