import React from 'react';

export default function TechDetails({ prediction, confidence, probabilities }) {
  if (!prediction) return null;

  return (
    <details className="mt-12 bg-slate-50 border border-slate-200 rounded-lg p-4 group cursor-pointer w-full text-left">
      <summary className="text-xs font-bold uppercase tracking-wider text-slate-500 group-hover:text-slate-700 transition">
        Technical Details (for developers)
      </summary>
      
      <div className="mt-4 text-sm text-slate-600 font-mono space-y-2 border-t border-slate-200 pt-4">
        <p><strong>Pipeline:</strong> CNN + LSTM Ensemble</p>
        <p><strong>API Endpoint:</strong> http://localhost:5000/predict/ensemble</p>
        <p><strong>Model F1 Score:</strong> 0.99 (Clean data) / 0.85 (Noisy data)</p>
        <p><strong>Max Confidence:</strong> {confidence !== undefined ? confidence.toFixed(4) : 'N/A'}</p>
        <p><strong>Probability Vector:</strong> {JSON.stringify(probabilities)}</p>
      </div>
    </details>
  );
}
