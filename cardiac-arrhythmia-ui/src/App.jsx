import { useState, useRef, useEffect } from 'react'

const ACCENT = '#2563eb'

function isCsvFile(file) {
  if (!file || !file.name) return false
  return file.name.toLowerCase().endsWith('.csv')
}

export default function App() {
  const [fileName, setFileName] = useState('')
  const [phase, setPhase] = useState('idle')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    },
    []
  )

  const assignFile = (file) => {
    if (!isCsvFile(file)) return
    setFileName(file.name)
    setPhase('idle')
  }

  const handleInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) assignFile(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) assignFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const openPicker = () => inputRef.current?.click()

  const handleAnalyze = () => {
    if (!fileName || phase === 'analyzing') return
    if (timerRef.current) clearTimeout(timerRef.current)
    setPhase('analyzing')
    const ms = 1000 + Math.floor(Math.random() * 1000)
    timerRef.current = setTimeout(() => {
      setPhase('complete')
      timerRef.current = null
    }, ms)
  }

  const showResultPanel = phase === 'analyzing' || phase === 'complete'

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; }
        html, body, #root {
          height: 100%;
          margin: 0;
        }
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 15px;
          line-height: 1.5;
          color: #1f2937;
          background: #eef0f3;
          -webkit-font-smoothing: antialiased;
        }
        .page {
          min-height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 32px 20px;
        }
        .card {
          width: 100%;
          max-width: 420px;
          background: #fff;
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.06);
          padding: 36px 32px 32px;
        }
        h1 {
          margin: 0 0 10px;
          font-size: 1.35rem;
          font-weight: 600;
          letter-spacing: -0.02em;
          color: #111827;
          text-align: center;
        }
        .subtitle {
          margin: 0 0 28px;
          font-size: 0.875rem;
          color: #6b7280;
          text-align: center;
          line-height: 1.45;
        }
        .dropzone {
          border: 1px dashed #c5cad3;
          border-radius: 10px;
          padding: 28px 20px;
          text-align: center;
          background: #fafbfc;
          cursor: pointer;
          transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
        }
        .dropzone:hover {
          border-color: #94a3b8;
          background: #f4f6f8;
        }
        .dropzone.drag {
          border-color: ${ACCENT};
          background: #eff6ff;
          box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.15);
        }
        .dropzone p {
          margin: 0 0 12px;
          font-size: 0.875rem;
          color: #4b5563;
        }
        .browse {
          display: inline-block;
          font-size: 0.8125rem;
          font-weight: 500;
          color: ${ACCENT};
        }
        .browse:hover { text-decoration: underline; }
        .file-meta {
          margin-top: 12px;
          font-size: 0.8125rem;
          color: #374151;
          word-break: break-all;
          min-height: 1.25em;
        }
        .file-meta:empty::before {
          content: '\\00a0';
        }
        input[type="file"] {
          position: absolute;
          width: 0;
          height: 0;
          opacity: 0;
          pointer-events: none;
        }
        .btn-primary {
          margin-top: 22px;
          width: 100%;
          padding: 11px 16px;
          font-size: 0.9375rem;
          font-weight: 500;
          color: #fff;
          background: ${ACCENT};
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: background 0.15s, opacity 0.15s;
        }
        .btn-primary:hover:not(:disabled) {
          background: #1d4ed8;
        }
        .btn-primary:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }
        .result-wrap {
          margin-top: 24px;
        }
        .result-box {
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          padding: 18px 20px;
          background: #fafbfc;
        }
        .result-box .label {
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #9ca3af;
          margin-bottom: 10px;
        }
        .analyzing {
          font-size: 0.9375rem;
          color: #4b5563;
        }
        .result-lines {
          margin: 0;
          font-size: 0.9375rem;
          color: #111827;
        }
        .result-lines dt {
          float: left;
          clear: left;
          width: 8.5rem;
          margin: 0;
          padding: 4px 0;
          font-weight: 500;
          color: #6b7280;
        }
        .result-lines dd {
          margin: 0 0 0 8.5rem;
          padding: 4px 0;
          font-weight: 500;
        }
        .result-lines::after {
          content: '';
          display: table;
          clear: both;
        }
      `}</style>

      <div className="page">
        <div className="card">
          <h1>Cardiac Arrhythmia Predictor</h1>
          <p className="subtitle">
            Upload ECG data in CSV format to analyze cardiac rhythm
          </p>

          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            onChange={handleInputChange}
            aria-label="Choose CSV file"
          />

          <div
            role="button"
            tabIndex={0}
            className={`dropzone${dragOver ? ' drag' : ''}`}
            onClick={openPicker}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                openPicker()
              }
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <p>Drag and drop your CSV here, or</p>
            <span className="browse">browse files</span>
            <div className="file-meta">{fileName}</div>
          </div>

          <button
            type="button"
            className="btn-primary"
            onClick={handleAnalyze}
            disabled={!fileName || phase === 'analyzing'}
          >
            Analyze File
          </button>

          {showResultPanel && (
            <div className="result-wrap">
              <div className="result-box">
                <div className="label">Result</div>
                {phase === 'analyzing' && (
                  <div className="analyzing">Analyzing...</div>
                )}
                {phase === 'complete' && (
                  <dl className="result-lines">
                    <dt>Prediction</dt>
                    <dd>Atrial Fibrillation</dd>
                    <dt>Confidence</dt>
                    <dd>92%</dd>
                  </dl>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
