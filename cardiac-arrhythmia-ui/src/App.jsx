import { useEffect, useMemo, useRef, useState } from 'react'
import ResultPanel from './components/ResultPanel'
import SignalChart from './components/SignalChart'
import UploadCard from './components/UploadCard'
import { CARD_SHADOW, COLORS } from './uiTheme'

const API_URL = 'http://localhost:5000'

function isCsvFile(file) {
  if (!file?.name) return false
  return file.name.toLowerCase().endsWith('.csv')
}

function parseECGFromCSV(content) {
  const lines = content.trim().split('\n')
  if (!lines.length) return null

  const firstLine = lines[0]
  const hasHeader =
    firstLine.toLowerCase().includes('label') ||
    firstLine.toLowerCase().includes('signal') ||
    Number.isNaN(parseFloat(firstLine.split(',')[0]))

  const values = []
  const startIndex = hasHeader ? 1 : 0

  for (let index = startIndex; index < lines.length; index += 1) {
    const line = lines[index].trim()
    if (!line) continue

    const arrayMatch = line.match(/\[([^\]]+)\]/)
    if (arrayMatch) {
      const parsed = arrayMatch[1]
        .split(',')
        .map((value) => parseFloat(value.trim()))
        .filter((value) => !Number.isNaN(value))

      if (parsed.length) return parsed
    }

    const columns = line.split(',')
    const startColumn = Number.isNaN(parseFloat(columns[0])) ? 1 : 0

    for (let columnIndex = startColumn; columnIndex < columns.length; columnIndex += 1) {
      const value = parseFloat(columns[columnIndex].trim())
      if (!Number.isNaN(value)) values.push(value)
    }

    if (values.length > 100) break
  }

  return values.length ? values : null
}

function getErrorMessage(message) {
  const text = String(message || '').toLowerCase()

  if (
    text.includes('invalid') ||
    text.includes('parse') ||
    text.includes('csv') ||
    text.includes('signal too short') ||
    text.includes('no file selected')
  ) {
    return 'Invalid file'
  }

  return 'Server error'
}

function shouldFallbackToStandardPredict(status, message) {
  const text = String(message || '').toLowerCase()

  return status >= 500 && text.includes('ensemble model not loaded')
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewSignal, setPreviewSignal] = useState(null)
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    document.title = 'Cardiac Monitoring Dashboard'
  }, [])

  const displayedSignal = useMemo(() => {
    if (Array.isArray(response?.signal) && response.signal.length) return response.signal
    return previewSignal
  }, [previewSignal, response])

  const handleBrowse = () => {
    inputRef.current?.click()
  }

  const clearSelection = () => {
    setSelectedFile(null)
    setPreviewSignal(null)
    setResponse(null)
    setError('')
    setLoading(false)

    if (inputRef.current) inputRef.current.value = ''
  }

  const processFile = async (file) => {
    if (!isCsvFile(file)) {
      setError('Invalid file')
      return
    }

    setSelectedFile(file)
    setResponse(null)
    setError('')

    try {
      const content = await file.text()
      const parsedSignal = parseECGFromCSV(content)
      setPreviewSignal(parsedSignal || null)
    } catch {
      setPreviewSignal(null)
    }
  }

  const handleInputChange = (event) => {
    const file = event.target.files?.[0]
    if (file) processFile(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    setDragActive(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)

    const file = event.dataTransfer.files?.[0]
    if (file) processFile(file)
  }

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return

    setLoading(true)
    setError('')
    setResponse(null)

    try {
      const requestPrediction = async (endpoint) => {
        const formData = new FormData()
        formData.append('ecg', selectedFile)

        const apiResponse = await fetch(`${API_URL}${endpoint}`, {
          method: 'POST',
          body: formData,
        })

        const data = await apiResponse.json()

        return { apiResponse, data }
      }

      let { apiResponse, data } = await requestPrediction('/predict/ensemble')

      if (shouldFallbackToStandardPredict(apiResponse.status, data.error)) {
        ;({ apiResponse, data } = await requestPrediction('/predict'))
      }

      if (!apiResponse.ok) {
        throw new Error(data.error || 'Server error')
      }

      setResponse(data)
    } catch (requestError) {
      setError(getErrorMessage(requestError.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <style>{`
        * { box-sizing: border-box; }
        html, body, #root { min-height: 100%; margin: 0; }
        body {
          font-family: "Segoe UI", Arial, sans-serif;
          background: ${COLORS.slate100};
          color: ${COLORS.slate900};
        }
        .page-shell {
          min-height: 100vh;
          padding: 28px;
        }
        .page-width {
          max-width: 1380px;
          margin: 0 auto;
        }
        .topbar {
          background: ${COLORS.white};
          border: 1px solid ${COLORS.slate200};
          border-radius: 18px;
          box-shadow: ${CARD_SHADOW};
          padding: 24px 28px;
          margin-bottom: 24px;
        }
        .topbar h1 {
          margin: 0 0 6px;
          font-size: 1.9rem;
          font-weight: 700;
          color: ${COLORS.slate900};
        }
        .topbar p {
          margin: 0;
          color: ${COLORS.slate500};
          font-size: 0.98rem;
        }
        .layout {
          display: grid;
          grid-template-columns: 360px minmax(0, 1fr);
          gap: 24px;
          align-items: start;
        }
        .stack {
          display: grid;
          gap: 24px;
        }
        .card {
          background: ${COLORS.white};
          border: 1px solid ${COLORS.slate200};
          border-radius: 18px;
          box-shadow: ${CARD_SHADOW};
          padding: 24px;
        }
        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 20px;
        }
        .section-header h2 {
          margin: 4px 0 0;
          font-size: 1.12rem;
          font-weight: 700;
          color: ${COLORS.slate900};
        }
        .eyebrow {
          margin: 0;
          color: ${COLORS.primary};
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .upload-surface {
          border: 1.5px dashed ${COLORS.slate300};
          border-radius: 16px;
          background: ${COLORS.slate50};
          padding: 28px 22px;
          display: flex;
          gap: 16px;
          cursor: pointer;
          transition: border-color 0.15s ease, background 0.15s ease;
        }
        .upload-surface.active,
        .upload-surface:hover {
          border-color: ${COLORS.primary};
          background: ${COLORS.primarySoft};
        }
        .upload-icon {
          width: 48px;
          height: 48px;
          border-radius: 14px;
          background: ${COLORS.white};
          border: 1px solid ${COLORS.slate200};
          color: ${COLORS.primary};
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        .upload-icon svg {
          width: 22px;
          height: 22px;
        }
        .upload-copy {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .upload-copy strong {
          font-size: 0.98rem;
        }
        .upload-copy span {
          color: ${COLORS.slate500};
          font-size: 0.9rem;
        }
        .file-row,
        .meta-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          padding: 16px 0;
          border-bottom: 1px solid ${COLORS.slate200};
        }
        .meta-item:last-child {
          border-bottom: 0;
          padding-bottom: 0;
        }
        .field-label {
          display: block;
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: ${COLORS.slate500};
          margin-bottom: 6px;
        }
        .field-value {
          color: ${COLORS.slate900};
          font-size: 0.94rem;
          word-break: break-word;
        }
        .text-button {
          border: 0;
          background: transparent;
          color: ${COLORS.primary};
          font-weight: 600;
          cursor: pointer;
          padding: 0;
        }
        .primary-button {
          margin-top: 18px;
          width: 100%;
          border: 0;
          border-radius: 14px;
          background: ${COLORS.primary};
          color: ${COLORS.white};
          font-size: 0.98rem;
          font-weight: 700;
          padding: 14px 18px;
          cursor: pointer;
          display: inline-flex;
          justify-content: center;
          align-items: center;
          gap: 10px;
        }
        .primary-button:disabled {
          background: ${COLORS.slate300};
          cursor: not-allowed;
        }
        .helper-note {
          margin-top: 14px;
          color: ${COLORS.slate500};
          font-size: 0.82rem;
          line-height: 1.5;
        }
        .helper-note code {
          background: ${COLORS.slate100};
          border-radius: 6px;
          padding: 2px 6px;
          color: ${COLORS.slate700};
        }
        .status-banner {
          margin-top: 16px;
          border-radius: 14px;
          padding: 14px 16px;
          font-size: 0.92rem;
          border: 1px solid ${COLORS.slate200};
        }
        .status-banner.error {
          background: #fef2f2;
          color: ${COLORS.danger};
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .status-banner.warning {
          background: #fff8eb;
          color: ${COLORS.warning};
          margin-bottom: 18px;
        }
        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .summary-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
          margin-bottom: 18px;
        }
        .summary-item {
          border: 1px solid ${COLORS.slate200};
          border-radius: 16px;
          background: ${COLORS.slate50};
          padding: 16px;
        }
        .summary-label {
          display: block;
          color: ${COLORS.slate500};
          font-size: 0.8rem;
          margin-bottom: 8px;
        }
        .summary-value {
          font-size: 1.1rem;
          font-weight: 700;
        }
        .probability-section {
          border-top: 1px solid ${COLORS.slate200};
          padding-top: 18px;
          margin-top: 8px;
        }
        .probability-list {
          display: grid;
          gap: 14px;
          margin-top: 10px;
        }
        .probability-head {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 8px;
          font-size: 0.9rem;
        }
        .probability-track {
          height: 10px;
          border-radius: 999px;
          background: ${COLORS.slate100};
          overflow: hidden;
        }
        .probability-fill {
          height: 100%;
          border-radius: 999px;
          background: ${COLORS.primary};
        }
        .meta-grid {
          margin-top: 20px;
          border-top: 1px solid ${COLORS.slate200};
          padding-top: 4px;
        }
        .chart-card {
          min-height: 440px;
        }
        .empty-state {
          min-height: 220px;
          border: 1px dashed ${COLORS.slate300};
          border-radius: 16px;
          background: ${COLORS.slate50};
          color: ${COLORS.slate500};
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 28px;
        }
        .signal-note {
          margin-top: 16px;
          color: ${COLORS.slate500};
          font-size: 0.84rem;
        }
        .spinner {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          border: 2px solid rgba(255, 255, 255, 0.35);
          border-top-color: ${COLORS.white};
          animation: spin 0.8s linear infinite;
        }
        .sr-only-input {
          position: absolute;
          width: 1px;
          height: 1px;
          opacity: 0;
          pointer-events: none;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @media (max-width: 1080px) {
          .layout {
            grid-template-columns: 1fr;
          }
        }
        @media (max-width: 760px) {
          .page-shell {
            padding: 16px;
          }
          .topbar,
          .card {
            padding: 18px;
            border-radius: 16px;
          }
          .summary-grid {
            grid-template-columns: 1fr;
          }
          .upload-surface {
            flex-direction: column;
          }
        }
      `}</style>

      <div className="page-shell">
        <div className="page-width">
          <header className="topbar">
            <h1>Cardiac Monitoring Dashboard</h1>
            <p>
              Upload ECG waveform data in CSV format to review the ensemble prediction and waveform
              trace in a clinical dashboard.
            </p>
          </header>

          <div className="layout">
            <UploadCard
              dragActive={dragActive}
              error={error}
              fileName={selectedFile?.name || ''}
              inputRef={inputRef}
              loading={loading}
              onAnalyze={handleAnalyze}
              onBrowse={handleBrowse}
              onChange={handleInputChange}
              onClear={clearSelection}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            />

            <div className="stack">
              {response?.low_confidence === true && (
                <div
                  style={{
                    background: '#fff8eb',
                    border: '1px solid #f59e0b',
                    color: '#92400e',
                    padding: '12px',
                    borderRadius: '8px',
                    marginBottom: '16px',
                  }}
                >
                  {'⚠️ Signal Quality Notice: Your signal had '}
                  {response.original_length}
                  {' points. Model is optimized for 2500 points (360Hz). Prediction may be less accurate.'}
                </div>
              )}

              <ResultPanel result={response} loading={loading} />

              {response?.was_normalized === true && response?.low_confidence === false && (
                <div
                  style={{
                    color: '#6b7280',
                    fontSize: '0.82rem',
                    marginTop: '8px',
                  }}
                >
                  {'ℹ️ Signal normalized from '}
                  {response.original_length}
                  {' → 2500 points'}
                </div>
              )}

              <section className="card chart-card">
                <div className="section-header">
                  <div>
                    <p className="eyebrow">Waveform</p>
                    <h2>ECG Signal View</h2>
                  </div>
                </div>

                <SignalChart signal={displayedSignal} />

                <div className="signal-note">
                  The waveform view uses the signal returned by the backend when available; otherwise
                  it previews the uploaded CSV content.
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
