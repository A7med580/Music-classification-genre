import { useState, useEffect } from 'react'
import axios from 'axios'
import { Upload, History, TrendingUp, Music, BarChart3, Star, Trash2 } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Dashboard() {
  const [file, setFile] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const { user } = useAuth()

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      const res = await axios.get('/api/history')
      setHistory(res.data)
    } catch (err) {
      console.error("Failed to fetch history", err)
    }
  }

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/history/${id}`)
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

    try {
      const res = await axios.post('/api/classify', formData)
      setPrediction(res.data)
      fetchHistory()
    } catch (err) {
      setError(err.response?.data?.details || "Classification failed.")
    } finally {
      setLoading(false)
    }
  }

  // Stats logic
  const totalScans = history.length;
  const topGenre = history.length > 0
    ? history.reduce((acc, curr) => {
      acc[curr.genre] = (acc[curr.genre] || 0) + 1;
      return acc;
    }, {})
    : null;
  const favoredGenre = topGenre
    ? Object.keys(topGenre).reduce((a, b) => topGenre[a] > topGenre[b] ? a : b)
    : "None";

  return (
    <div style={{ width: '100%', maxWidth: '1000px' }}>
      <div style={{ textAlign: 'left', marginBottom: '2rem' }}>
        <h1>Music Intelligence</h1>
        <p style={{ opacity: 0.7 }}>Welcome back, <span style={{ color: '#4facfe', fontWeight: 'bold' }}>{user?.username}</span></p>
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
          <div style={{ fontSize: '0.8em', opacity: 0.6, marginTop: '5px' }}>Accuracy Avg</div>
          <div className="stat-value">88%</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
        <div className="glass-card" style={{ maxWidth: 'none' }}>
          <h3 style={{ marginTop: 0, textAlign: 'left', display: 'flex', alignItems: 'center', gap: '10px' }}>
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
          <button onClick={handleUpload} disabled={!file || loading} style={{ width: '100%' }}>
            {loading ? 'Analyzing Neural Patterns...' : 'Start Classification'}
          </button>

          {prediction && (
            <div className="result-container" style={{ textAlign: 'left' }}>
              <div className="genre-title">{prediction.genre}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem' }}>
                <span style={{ opacity: 0.6 }}>Confidence Level</span>
                <span>{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="confidence-bar">
                <div className="confidence-fill" style={{ width: `${prediction.confidence * 100}%` }}></div>
              </div>
            </div>
          )}
        </div>

        <div>
          <h3 style={{ marginTop: 0, textAlign: 'left', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <History size={20} /> Recent History
          </h3>
          <div style={{ maxHeight: '450px', overflowY: 'auto', paddingRight: '10px' }}>
            {history.map(item => (
              <div key={item.id} className="history-item" style={{ marginBottom: '10px' }}>
                <div style={{ textTransform: 'capitalize', fontWeight: 'bold', color: '#4facfe' }}>{item.genre}</div>
                <div style={{ fontSize: '0.8em', opacity: 0.7, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.filename}
                </div>
                <div style={{ fontSize: '0.7em', opacity: 0.4 }}>{new Date(item.timestamp).toLocaleString()}</div>
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
