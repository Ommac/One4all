import { COLORS } from '../uiTheme'

function normalizeProbabilityValue(value) {
  if (typeof value === 'number') {
    return Math.max(0, Math.min(100, value * 100))
  }

  if (typeof value === 'string') {
    const parsed = parseFloat(value.replace('%', '').trim())
    if (!Number.isNaN(parsed)) return Math.max(0, Math.min(100, parsed))
  }

  return 0
}

export default function ResultPanel({ result, loading }) {
  if (loading) {
    return (
      <section className="card result-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Analysis</p>
            <h2>Prediction Result</h2>
          </div>
        </div>
        <div className="empty-state">The ECG is being analyzed. Results will appear here shortly.</div>
      </section>
    )
  }

  if (!result) {
    return (
      <section className="card result-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Analysis</p>
            <h2>Prediction Result</h2>
          </div>
        </div>
        <div className="empty-state">Upload a valid ECG CSV file to review the classification output.</div>
      </section>
    )
  }

  const rawPrediction = result.prediction || '--'
  const classification =
    rawPrediction === 'Normal' || rawPrediction === 'Normal Sinus Rhythm' ? 'Normal' : 'Abnormal'
  const confidence = result.confidence ?? '--'
  const probabilities = Object.entries(result.probabilities || {})

  return (
    <section className="card result-card">
      <div className="section-header">
        <div>
          <p className="eyebrow">Analysis</p>
          <h2>Prediction Result</h2>
        </div>
      </div>

      <div className="summary-grid">
        <div className="summary-item">
          <span className="summary-label">Classification</span>
          <strong
            className="summary-value"
            style={{ color: classification === 'Normal' ? COLORS.success : COLORS.danger }}
          >
            {classification}
          </strong>
        </div>
        <div className="summary-item">
          <span className="summary-label">Detected Rhythm</span>
          <strong className="summary-value">{rawPrediction}</strong>
        </div>
        <div className="summary-item">
          <span className="summary-label">Confidence</span>
          <strong className="summary-value">{confidence}</strong>
        </div>
      </div>

      {result.warning && <div className="status-banner warning">{result.warning}</div>}

      {probabilities.length > 0 && (
        <div className="probability-section">
          <div className="field-label">Class probabilities</div>
          <div className="probability-list">
            {probabilities.map(([label, value]) => {
              const width = normalizeProbabilityValue(value)

              return (
                <div key={label} className="probability-item">
                  <div className="probability-head">
                    <span>{label}</span>
                    <strong>{typeof value === 'number' ? `${width.toFixed(1)}%` : value}</strong>
                  </div>
                  <div className="probability-track">
                    <div className="probability-fill" style={{ width: `${width}%` }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <div className="meta-grid">
        <div className="meta-item">
          <span className="field-label">Model</span>
          <span className="field-value">{result.model_used || 'Backend prediction service'}</span>
        </div>
        <div className="meta-item">
          <span className="field-label">Reported F1</span>
          <span className="field-value">{result.model_f1_score || 'Not provided'}</span>
        </div>
      </div>
    </section>
  )
}
