import React from 'react';
import { ChevronDown } from 'lucide-react';

export default function TechDetails({ prediction, confidence, probabilities }) {
  if (!prediction) return null;

  return (
    <details 
      className="animate-float-in animate-float-in-5 glass-card p-5 group cursor-pointer w-full text-left mt-8"
      id="tech-details"
    >
      <summary className="text-[11px] font-bold uppercase tracking-[0.15em] text-slate-600 group-hover:text-slate-400 transition-colors flex items-center gap-2 list-none">
        <ChevronDown size={14} className="text-slate-600 group-open:rotate-180 transition-transform duration-300" />
        Technical Details (for developers)
      </summary>
      
      <div className="mt-5 text-sm text-slate-400 font-mono space-y-3 border-t border-slate-700/40 pt-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block mb-1">Pipeline</span>
            <span className="text-slate-300 text-xs font-semibold">CNN + LSTM Ensemble</span>
          </div>
          <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block mb-1">API Endpoint</span>
            <span className="text-cyan-400/80 text-xs font-semibold">localhost:5000/predict/ensemble</span>
          </div>
          <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block mb-1">Model F1 Score</span>
            <span className="text-slate-300 text-xs font-semibold">0.99 (Clean) / 0.85 (Noisy)</span>
          </div>
          <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block mb-1">Max Confidence</span>
            <span className="text-slate-300 text-xs font-semibold">
              {confidence !== undefined ? confidence.toFixed(4) : 'N/A'}
            </span>
          </div>
        </div>
        <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
          <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block mb-1">Probability Vector</span>
          <code className="text-cyan-400/70 text-xs break-all">
            {JSON.stringify(probabilities, null, 2)}
          </code>
        </div>
      </div>
    </details>
  );
}
