import { COLORS } from '../uiTheme'

export default function UploadCard({
  dragActive,
  fileName,
  inputRef,
  loading,
  error,
  onAnalyze,
  onBrowse,
  onChange,
  onClear,
  onDragLeave,
  onDragOver,
  onDrop,
}) {
  return (
    <section className="card">
      <div className="section-header">
        <div>
          <p className="eyebrow">Data Intake</p>
          <h2>ECG CSV Upload</h2>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        onChange={onChange}
        aria-label="Upload ECG CSV file"
        className="sr-only-input"
      />

      <div
        className={`upload-surface${dragActive ? ' active' : ''}`}
        onClick={onBrowse}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onBrowse()
          }
        }}
      >
        <div className="upload-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M12 16V4" />
            <path d="M7 9l5-5 5 5" />
            <path d="M4 16.5V19a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2.5" />
          </svg>
        </div>
        <div className="upload-copy">
          <strong>Select a .csv file</strong>
          <span>Drag and drop or click to browse local ECG data.</span>
        </div>
      </div>

      <div className="file-row">
        <div>
          <div className="field-label">Selected file</div>
          <div className="field-value">{fileName || 'No file selected'}</div>
        </div>
        {fileName && (
          <button type="button" className="text-button" onClick={onClear}>
            Clear
          </button>
        )}
      </div>

      {error && (
        <div className="status-banner error">
          <span className="status-dot" style={{ background: COLORS.danger }} />
          <span>{error}</span>
        </div>
      )}

      <button
        type="button"
        className="primary-button"
        onClick={onAnalyze}
        disabled={!fileName || loading}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Processing ECG...
          </>
        ) : (
          'Upload and Analyze'
        )}
      </button>

      <div className="helper-note">
        Files are sent to <code>http://localhost:5000/predict/ensemble</code> using
        multipart upload with the <code>ecg</code> key.
      </div>
    </section>
  )
}
