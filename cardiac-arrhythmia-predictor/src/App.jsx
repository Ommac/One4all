import { useState, useRef } from 'react'

export default function App() {
  const [fileName, setFileName] = useState('')
  const [phase, setPhase] = useState('idle')
  const timeoutRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    setFileName(file ? file.name : '')
    setPhase('idle')
  }

  const handleAnalyze = () => {
    if (phase === 'processing') return
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    setPhase('processing')
    timeoutRef.current = setTimeout(() => {
      setPhase('done')
      timeoutRef.current = null
    }, 2000)
  }

  return (
    <>
      <style>{`
        html, body, #root {
          height: 100%;
          margin: 0;
        }
        body {
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 15px;
          color: #222;
          background: #fff;
        }
        .wrap {
          min-height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
        }
        .card {
          width: 100%;
          max-width: 360px;
          padding: 28px 24px;
          border: 1px solid #ddd;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        h1 {
          margin: 0 0 20px;
          font-size: 1.15rem;
          font-weight: 600;
          text-align: center;
          line-height: 1.3;
        }
        label.file-label {
          display: block;
          margin-bottom: 8px;
          font-size: 0.875rem;
          color: #444;
        }
        input[type="file"] {
          width: 100%;
          font-size: 0.8125rem;
        }
        .file-name {
          margin-top: 10px;
          font-size: 0.8125rem;
          color: #555;
          word-break: break-all;
          min-height: 1.2em;
        }
        button {
          margin-top: 18px;
          width: 100%;
          padding: 10px 14px;
          font-size: 0.9375rem;
          border: 1px solid #bbb;
          border-radius: 6px;
          background: #f7f7f7;
          cursor: pointer;
        }
        button:hover {
          background: #eee;
        }
        button:disabled {
          opacity: 0.65;
          cursor: not-allowed;
        }
        .status {
          margin-top: 16px;
          font-size: 0.875rem;
          text-align: center;
          color: #333;
          min-height: 1.3em;
        }
      `}</style>
      <div className="wrap">
        <div className="card">
          <h1>Cardiac Arrhythmia Predictor</h1>
          <label className="file-label" htmlFor="csv">
            CSV file
          </label>
          <input
            id="csv"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
          />
          <div className="file-name">{fileName}</div>
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={phase === 'processing'}
          >
            Analyze File
          </button>
          <div className="status">
            {phase === 'processing' && 'Processing...'}
            {phase === 'done' && 'Prediction: Atrial Fibrillation (Dummy)'}
          </div>
        </div>
      </div>
    </>
  )
}
