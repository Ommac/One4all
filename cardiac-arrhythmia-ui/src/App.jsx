<<<<<<< HEAD
import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceArea, Tooltip } from 'recharts'

// API Configuration
const API_URL = 'http://localhost:5000'

// Color palette - medical-grade
const COLORS = {
  primary: '#0066CC',
  primaryDark: '#004C99',
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#DC2626',
  gray50: '#F9FAFB',
  gray100: '#F3F4F6',
  gray200: '#E5E7EB',
  gray300: '#D1D5DB',
  gray400: '#9CA3AF',
  gray500: '#6B7280',
  gray600: '#4B5563',
  gray700: '#374151',
  gray800: '#1F2937',
  gray900: '#111827',
  ecgLine: '#0066CC',
  ecgAbnormal: '#DC2626',
  gridLine: '#E5E7EB',
}

// Risk level mapping
const getRiskLevel = (prediction, confidence) => {
  if (prediction === 'VFib') return { level: 'CRITICAL', color: COLORS.danger }
  if (prediction === 'AFib') {
    if (confidence >= 0.9) return { level: 'HIGH', color: COLORS.danger }
    if (confidence >= 0.7) return { level: 'MEDIUM', color: COLORS.warning }
    return { level: 'LOW', color: COLORS.success }
  }
  return { level: 'LOW', color: COLORS.success }
}

// Utility to validate CSV files
function isCsvFile(file) {
  if (!file || !file.name) return false
  return file.name.toLowerCase().endsWith('.csv')
}

// Parse ECG signal from CSV content
function parseECGFromCSV(content) {
  const lines = content.trim().split('\n')
  if (lines.length === 0) return null

  // Check for header
  const firstLine = lines[0]
  const hasHeader = firstLine.toLowerCase().includes('label') || 
                    firstLine.toLowerCase().includes('signal') ||
                    isNaN(parseFloat(firstLine.split(',')[0]))

  let signalData = []
  const startIdx = hasHeader ? 1 : 0

  for (let i = startIdx; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue

    // Check if line contains a stringified array like "[0.031, 0.038, ...]"
    const arrayMatch = line.match(/\[([^\]]+)\]/)
    if (arrayMatch) {
      const values = arrayMatch[1].split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      if (values.length > 0) {
        signalData = values
        break
      }
    }

    // Otherwise parse as comma-separated values
    const parts = line.split(',')
    // Skip first column if it looks like a label
    const startCol = isNaN(parseFloat(parts[0])) ? 1 : 0
    for (let j = startCol; j < parts.length; j++) {
      const val = parseFloat(parts[j].trim())
      if (!isNaN(val)) signalData.push(val)
    }
    if (signalData.length > 100) break  // We have enough data from first row
  }

  return signalData.length > 0 ? signalData : null
}

// ECG Chart Component
function ECGChart({ data, abnormalRegion, isAbnormal }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []
    // Downsample for performance if needed
    const step = data.length > 1000 ? Math.ceil(data.length / 500) : 1
    return data
      .filter((_, i) => i % step === 0)
      .map((value, index) => ({
        time: (index * step / 360).toFixed(2),
        voltage: value,
        index: index * step
      }))
  }, [data])

  if (!chartData.length) return null

  const minVoltage = Math.min(...chartData.map(d => d.voltage))
  const maxVoltage = Math.max(...chartData.map(d => d.voltage))
  const padding = (maxVoltage - minVoltage) * 0.1

  return (
    <div style={{ width: '100%', height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gridLine} />
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 11, fill: COLORS.gray500 }}
            tickLine={{ stroke: COLORS.gray300 }}
            axisLine={{ stroke: COLORS.gray300 }}
            label={{ value: 'Time (s)', position: 'bottom', offset: -5, fontSize: 11, fill: COLORS.gray500 }}
          />
          <YAxis 
            domain={[minVoltage - padding, maxVoltage + padding]}
            tick={{ fontSize: 11, fill: COLORS.gray500 }}
            tickLine={{ stroke: COLORS.gray300 }}
            axisLine={{ stroke: COLORS.gray300 }}
            label={{ value: 'mV', angle: -90, position: 'insideLeft', fontSize: 11, fill: COLORS.gray500 }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip 
            contentStyle={{ 
              background: 'white', 
              border: `1px solid ${COLORS.gray200}`,
              borderRadius: 6,
              fontSize: 12
            }}
            formatter={(value) => [value.toFixed(4) + ' mV', 'Voltage']}
            labelFormatter={(label) => `Time: ${label}s`}
          />
          {abnormalRegion && (
            <ReferenceArea
              x1={abnormalRegion.start}
              x2={abnormalRegion.end}
              fill={COLORS.danger}
              fillOpacity={0.15}
              stroke={COLORS.danger}
              strokeOpacity={0.3}
            />
          )}
          <Line
            type="monotone"
            dataKey="voltage"
            stroke={isAbnormal ? COLORS.ecgAbnormal : COLORS.ecgLine}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Loading Spinner Component
function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <div className="spinner" />
      <span style={{ color: COLORS.gray600, fontSize: 14 }}>Analyzing ECG signal...</span>
    </div>
  )
}

// Result Card Component
function ResultCard({ title, children, variant = 'default' }) {
  const borderColor = variant === 'danger' ? COLORS.danger : 
                      variant === 'warning' ? COLORS.warning : 
                      variant === 'success' ? COLORS.success : COLORS.gray200
  
  return (
    <div style={{
      background: 'white',
      borderRadius: 8,
      border: `1px solid ${borderColor}`,
      borderLeft: `4px solid ${borderColor}`,
      padding: '16px 20px',
      marginBottom: 16
    }}>
      {title && (
        <div style={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: COLORS.gray500,
          marginBottom: 12
        }}>
          {title}
        </div>
      )}
      {children}
    </div>
  )
}

// Model Badge Component
function ModelBadge({ name, prediction, isActive }) {
  const bgColor = prediction === 'Normal' ? '#DCFCE7' :
                  prediction === 'AFib' ? '#FEF3C7' :
                  prediction === 'VFib' ? '#FEE2E2' : COLORS.gray100
  const textColor = prediction === 'Normal' ? '#166534' :
                    prediction === 'AFib' ? '#92400E' :
                    prediction === 'VFib' ? '#991B1B' : COLORS.gray600

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '10px 14px',
      background: isActive ? COLORS.gray50 : 'white',
      borderRadius: 6,
      border: `1px solid ${COLORS.gray200}`
    }}>
      <span style={{ fontSize: 13, color: COLORS.gray700, fontWeight: 500 }}>{name}</span>
      <span style={{
        fontSize: 12,
        fontWeight: 600,
        padding: '4px 10px',
        borderRadius: 12,
        background: bgColor,
        color: textColor
      }}>
        {prediction}
      </span>
    </div>
  )
}

// Main App Component
export default function App() {
  const [file, setFile] = useState(null)
  const [fileName, setFileName] = useState('')
  const [ecgData, setEcgData] = useState(null)
  const [phase, setPhase] = useState('idle')  // idle, analyzing, complete, error
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  // Parse CSV file on upload
  const processFile = useCallback(async (uploadedFile) => {
    if (!isCsvFile(uploadedFile)) {
      setError('Please upload a valid CSV file')
      return
    }
    
    setFile(uploadedFile)
    setFileName(uploadedFile.name)
    setError('')
    setResult(null)
    setPhase('idle')

    // Read and parse the file for preview
    try {
      const content = await uploadedFile.text()
      const signal = parseECGFromCSV(content)
      if (signal && signal.length > 0) {
        setEcgData(signal)
      } else {
        setError('Could not parse ECG signal from file')
      }
    } catch (err) {
      setError('Error reading file: ' + err.message)
    }
  }, [])

  const handleInputChange = (e) => {
    const f = e.target.files?.[0]
    if (f) processFile(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) processFile(f)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const openPicker = () => inputRef.current?.click()

  // Analyze ECG via API
  const handleAnalyze = async () => {
    if (!file || phase === 'analyzing') return
    
    setPhase('analyzing')
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('ecg', file)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      
      // Update ECG data with filtered signal from response
      if (data.signal && data.signal.length > 0) {
        setEcgData(data.signal)
      }
      
      setPhase('complete')
    } catch (err) {
      setError(err.message || 'Failed to analyze ECG')
      setPhase('error')
    }
  }

  // Reset handler
  const handleReset = () => {
    setFile(null)
    setFileName('')
    setEcgData(null)
    setPhase('idle')
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  // Calculate risk level
  const riskInfo = result ? getRiskLevel(result.prediction, result.confidence) : null
  const isAbnormal = result && result.prediction !== 'Normal'

  // Abnormal region for highlighting (simulated based on prediction)
  const abnormalRegion = useMemo(() => {
    if (!isAbnormal || !ecgData) return null
    const totalTime = (ecgData.length / 360).toFixed(2)
    // Highlight middle portion as abnormal region
    return {
      start: (totalTime * 0.3).toFixed(2),
      end: (totalTime * 0.7).toFixed(2)
    }
  }, [isAbnormal, ecgData])
=======
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
      const formData = new FormData()
      formData.append('ecg', selectedFile)

      const apiResponse = await fetch(`${API_URL}/predict/ensemble`, {
        method: 'POST',
        body: formData,
      })

      const data = await apiResponse.json()

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
>>>>>>> 2a54c5d6894d190445be40b9332b4f75bdcccb7f

  return (
    <>
      <style>{`
<<<<<<< HEAD
        *, *::before, *::after { box-sizing: border-box; }
        html, body, #root {
          height: 100%;
          margin: 0;
        }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
          font-size: 14px;
          line-height: 1.5;
          color: ${COLORS.gray800};
          background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f8 100%);
          -webkit-font-smoothing: antialiased;
        }
        .dashboard {
          min-height: 100vh;
          padding: 24px;
        }
        .header {
          text-align: center;
          margin-bottom: 32px;
          padding: 24px;
          background: white;
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .header h1 {
          margin: 0 0 8px;
          font-size: 1.75rem;
          font-weight: 700;
          color: ${COLORS.primary};
          letter-spacing: -0.02em;
        }
        .header .subtitle {
          margin: 0;
          font-size: 0.9375rem;
          color: ${COLORS.gray500};
          font-weight: 400;
        }
        .main-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 24px;
          max-width: 1400px;
          margin: 0 auto;
        }
        @media (min-width: 1024px) {
          .main-grid {
            grid-template-columns: 380px 1fr;
          }
        }
        .panel {
          background: white;
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03);
          padding: 24px;
        }
        .panel-title {
          font-size: 0.875rem;
          font-weight: 600;
          color: ${COLORS.gray800};
          margin: 0 0 20px;
          padding-bottom: 12px;
          border-bottom: 1px solid ${COLORS.gray200};
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .panel-title svg {
          width: 18px;
          height: 18px;
          color: ${COLORS.primary};
        }
        .dropzone {
          border: 2px dashed ${COLORS.gray300};
          border-radius: 10px;
          padding: 32px 24px;
          text-align: center;
          background: ${COLORS.gray50};
          cursor: pointer;
          transition: all 0.15s ease;
        }
        .dropzone:hover {
          border-color: ${COLORS.primary};
          background: #f0f7ff;
        }
        .dropzone.drag {
          border-color: ${COLORS.primary};
          background: #e0f0ff;
          box-shadow: inset 0 0 0 2px rgba(0, 102, 204, 0.1);
        }
        .dropzone-icon {
          width: 48px;
          height: 48px;
          margin: 0 auto 16px;
          color: ${COLORS.gray400};
        }
        .dropzone p {
          margin: 0 0 8px;
          font-size: 0.9375rem;
          color: ${COLORS.gray600};
        }
        .browse-link {
          font-size: 0.875rem;
          font-weight: 500;
          color: ${COLORS.primary};
          text-decoration: none;
        }
        .browse-link:hover { text-decoration: underline; }
        .file-info {
          margin-top: 16px;
          padding: 12px 16px;
          background: ${COLORS.gray100};
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }
        .file-name {
          font-size: 0.8125rem;
          color: ${COLORS.gray700};
          font-weight: 500;
          word-break: break-all;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .file-icon {
          width: 16px;
          height: 16px;
          color: ${COLORS.success};
        }
        .btn-clear {
          background: none;
          border: none;
          color: ${COLORS.gray400};
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
        }
        .btn-clear:hover {
          background: ${COLORS.gray200};
          color: ${COLORS.gray600};
        }
        input[type="file"] {
          position: absolute;
          width: 0;
          height: 0;
          opacity: 0;
          pointer-events: none;
        }
        .btn-primary {
          margin-top: 20px;
          width: 100%;
          padding: 14px 20px;
          font-size: 0.9375rem;
          font-weight: 600;
          color: #fff;
          background: ${COLORS.primary};
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.15s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }
        .btn-primary:hover:not(:disabled) {
          background: ${COLORS.primaryDark};
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 102, 204, 0.25);
        }
        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        .btn-secondary {
          margin-top: 12px;
          width: 100%;
          padding: 12px 16px;
          font-size: 0.875rem;
          font-weight: 500;
          color: ${COLORS.gray600};
          background: ${COLORS.gray100};
          border: 1px solid ${COLORS.gray200};
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.15s ease;
        }
        .btn-secondary:hover {
          background: ${COLORS.gray200};
          color: ${COLORS.gray700};
        }
        .error-box {
          margin-top: 16px;
          padding: 12px 16px;
          background: #FEF2F2;
          border: 1px solid #FECACA;
          border-radius: 8px;
          color: ${COLORS.danger};
          font-size: 0.8125rem;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .warning-banner {
          padding: 16px 20px;
          background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
          border: 1px solid #F59E0B;
          border-radius: 10px;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .warning-banner.critical {
          background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
          border-color: ${COLORS.danger};
        }
        .warning-icon {
          width: 24px;
          height: 24px;
          flex-shrink: 0;
        }
        .warning-text {
          font-size: 0.9375rem;
          font-weight: 600;
          color: ${COLORS.gray800};
        }
        .result-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
          margin-bottom: 20px;
        }
        .stat-card {
          text-align: center;
          padding: 20px 16px;
          background: ${COLORS.gray50};
          border-radius: 10px;
          border: 1px solid ${COLORS.gray200};
        }
        .stat-label {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: ${COLORS.gray500};
          margin-bottom: 8px;
        }
        .stat-value {
          font-size: 1.5rem;
          font-weight: 700;
          color: ${COLORS.gray800};
        }
        .stat-value.success { color: ${COLORS.success}; }
        .stat-value.warning { color: ${COLORS.warning}; }
        .stat-value.danger { color: ${COLORS.danger}; }
        .models-grid {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .ecg-placeholder {
          height: 280px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: ${COLORS.gray50};
          border: 1px dashed ${COLORS.gray300};
          border-radius: 8px;
          color: ${COLORS.gray400};
          font-size: 0.875rem;
        }
        .performance-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
        }
        .perf-item {
          text-align: center;
          padding: 16px 12px;
          background: ${COLORS.gray50};
          border-radius: 8px;
        }
        .perf-label {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: ${COLORS.gray500};
          margin-bottom: 6px;
        }
        .perf-value {
          font-size: 1.125rem;
          font-weight: 700;
          color: ${COLORS.primary};
        }
        .perf-value.green { color: ${COLORS.success}; }
        .spinner {
          width: 20px;
          height: 20px;
          border: 2px solid ${COLORS.gray200};
          border-top-color: ${COLORS.primary};
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .footer {
          text-align: center;
          margin-top: 32px;
          font-size: 0.75rem;
          color: ${COLORS.gray400};
        }
      `}</style>

      <div className="dashboard">
        {/* Header */}
        <header className="header">
          <h1>🫀 AI Cardiac Arrhythmia Predictor</h1>
          <p className="subtitle">Clinical Decision Support System • AI-Powered ECG Analysis</p>
        </header>

        {/* Main Grid */}
        <div className="main-grid">
          {/* Left Panel - Upload & Controls */}
          <div className="panel">
            <h2 className="panel-title">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
              </svg>
              ECG Data Upload
            </h2>

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
              <svg className="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 13h6m-3-3v6m5 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5.586a1 1 0 0 1 .707.293l5.414 5.414a1 1 0 0 1 .293.707V19a2 2 0 0 1-2 2z"/>
              </svg>
              <p>Drag & drop ECG file here</p>
              <span className="browse-link">or click to browse</span>
            </div>

            {fileName && (
              <div className="file-info">
                <span className="file-name">
                  <svg className="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 13l4 4L19 7"/>
                  </svg>
                  {fileName}
                </span>
                <button className="btn-clear" onClick={handleReset} title="Remove file">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12"/>
                  </svg>
                </button>
              </div>
            )}

            {error && (
              <div className="error-box">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M12 8v4m0 4h.01"/>
                </svg>
                {error}
              </div>
            )}

            <button
              type="button"
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={!file || phase === 'analyzing'}
            >
              {phase === 'analyzing' ? (
                <>
                  <div className="spinner" style={{ borderTopColor: 'white', borderColor: 'rgba(255,255,255,0.3)' }} />
                  Analyzing...
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                  </svg>
                  Analyze ECG
                </>
              )}
            </button>

            {result && (
              <button type="button" className="btn-secondary" onClick={handleReset}>
                Upload New File
              </button>
            )}

            {/* Performance Panel */}
            <div style={{ marginTop: 28 }}>
              <h3 style={{ fontSize: '0.8125rem', fontWeight: 600, color: COLORS.gray700, marginBottom: 14 }}>
                Model Performance
              </h3>
              <div className="performance-grid">
                <div className="perf-item">
                  <div className="perf-label">Model</div>
                  <div className="perf-value">CNN</div>
                </div>
                <div className="perf-item">
                  <div className="perf-label">Clean Acc</div>
                  <div className="perf-value green">99.9%</div>
                </div>
                <div className="perf-item">
                  <div className="perf-label">Noisy Acc</div>
                  <div className="perf-value">85.3%</div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - ECG Chart & Results */}
          <div className="panel">
            <h2 className="panel-title">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
              ECG Waveform Analysis
            </h2>

            {/* Warning Banner */}
            {result && result.prediction !== 'Normal' && (
              <div className={`warning-banner ${result.prediction === 'VFib' ? 'critical' : ''}`}>
                <svg className="warning-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                  <line x1="12" y1="9" x2="12" y2="13"/>
                  <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                <span className="warning-text">
                  ⚠️ Warning: {Math.round(result.confidence * 100)}% probability of {result.prediction === 'AFib' ? 'Atrial Fibrillation' : 'Ventricular Fibrillation'} detected
                </span>
              </div>
            )}

            {/* ECG Chart */}
            {ecgData ? (
              <ECGChart data={ecgData} abnormalRegion={abnormalRegion} isAbnormal={isAbnormal} />
            ) : (
              <div className="ecg-placeholder">
                <span>Upload an ECG file to view waveform</span>
              </div>
            )}

            {/* Results Section */}
            {phase === 'analyzing' && (
              <div style={{ marginTop: 24, textAlign: 'center', padding: 20 }}>
                <LoadingSpinner />
              </div>
            )}

            {result && (
              <div style={{ marginTop: 24 }}>
                {/* Main Result Stats */}
                <div className="result-grid">
                  <div className="stat-card">
                    <div className="stat-label">Prediction</div>
                    <div className={`stat-value ${result.prediction === 'Normal' ? 'success' : result.prediction === 'VFib' ? 'danger' : 'warning'}`}>
                      {result.prediction === 'AFib' ? 'AFib' : result.prediction}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Confidence</div>
                    <div className="stat-value">
                      {Math.round(result.confidence * 100)}%
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Risk Level</div>
                    <div className="stat-value" style={{ color: riskInfo?.color }}>
                      {riskInfo?.level}
                    </div>
                  </div>
                </div>

                {/* Probability Distribution */}
                <ResultCard title="Class Probabilities">
                  <div style={{ display: 'flex', gap: 16 }}>
                    {Object.entries(result.probabilities || {}).map(([label, prob]) => (
                      <div key={label} style={{ flex: 1, textAlign: 'center' }}>
                        <div style={{
                          height: 8,
                          background: COLORS.gray200,
                          borderRadius: 4,
                          overflow: 'hidden',
                          marginBottom: 8
                        }}>
                          <div style={{
                            height: '100%',
                            width: `${prob * 100}%`,
                            background: label === 'Normal' ? COLORS.success : label === 'VFib' ? COLORS.danger : COLORS.warning,
                            borderRadius: 4
                          }} />
                        </div>
                        <div style={{ fontSize: 11, color: COLORS.gray500 }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.gray700 }}>
                          {(prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </ResultCard>

                {/* Individual Model Results */}
                <ResultCard title="Model Consensus">
                  <div className="models-grid">
                    <ModelBadge name="XGBoost / RandomForest" prediction={result.xgb_result} />
                    <ModelBadge name="CNN (Convolutional)" prediction={result.cnn_result} isActive />
                    <ModelBadge name="LSTM (Sequential)" prediction={result.lstm_result} />
                  </div>
                </ResultCard>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="footer">
          AI Cardiac Arrhythmia Predictor • For Clinical Decision Support Only • Not a Diagnostic Tool
        </footer>
=======
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
              <ResultPanel result={response} loading={loading} />

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
>>>>>>> 2a54c5d6894d190445be40b9332b4f75bdcccb7f
      </div>
    </>
  )
}
