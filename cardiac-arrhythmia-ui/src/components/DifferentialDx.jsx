import React from 'react';
import { AlertCircle } from 'lucide-react';

const BAR_CONFIG = {
  VFib: { color: 'bg-red-500', label: 'Ventricular Fibrillation', glow: 'shadow-red-500/30' },
  AFib: { color: 'bg-amber-500', label: 'Atrial Fibrillation', glow: 'shadow-amber-500/30' },
  Normal: { color: 'bg-emerald-500', label: 'Normal Sinus Rhythm', glow: 'shadow-emerald-500/30' },
};

function parseProb(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = parseFloat(value.replace('%', '').trim());
    if (!Number.isNaN(parsed)) return parsed / 100;
  }
  return 0;
}

export default function DifferentialDx({ probabilities }) {
  if (!probabilities) return null;

  const vfib = parseProb(probabilities.VFib);
  const afib = parseProb(probabilities.AFib);
  const normal = parseProb(probabilities.Normal);

  const sorted = [
    { name: 'VFib', prob: vfib, ...BAR_CONFIG.VFib },
    { name: 'AFib', prob: afib, ...BAR_CONFIG.AFib },
    { name: 'Normal', prob: normal, ...BAR_CONFIG.Normal },
  ].sort((a, b) => b.prob - a.prob);

  const top1 = sorted[0];
  const top2 = sorted[1];
  const isClose = (top1.prob - top2.prob) < 0.1;

  return (
    <div className="glass-card p-6 h-full flex flex-col" id="differential-dx">
      <h3 className="text-white font-bold text-lg tracking-tight mb-5">
        Differential Diagnosis
      </h3>
      
      {isClose && (
        <div className="bg-slate-800/60 border border-slate-600/30 p-4 mb-5 flex items-start gap-3 rounded-xl">
          <AlertCircle className="text-amber-400 flex-shrink-0 mt-0.5" size={18} />
          <p className="text-slate-300 font-semibold text-sm leading-relaxed">
            Two rhythms are similarly likely — verify clinically
          </p>
        </div>
      )}

      <div className="space-y-5 flex-1">
        {sorted.map((item, idx) => {
          const pct = Math.round(item.prob * 100);
          return (
            <div key={item.name} className="group">
              <div className="flex justify-between text-sm font-bold mb-2">
                <span className="text-slate-300 group-hover:text-white transition-colors">
                  {item.label}
                </span>
                <span className="text-white tabular-nums">{pct}%</span>
              </div>
              <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/50">
                <div 
                  className={`h-full ${item.color} rounded-full transition-all duration-700 ease-out`}
                  style={{ 
                    width: `${Math.max(pct, 1)}%`,
                    boxShadow: idx === 0 ? `0 0 12px ${item.glow}` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-slate-700/40">
        <div className="flex items-center gap-4 flex-wrap">
          {Object.entries(BAR_CONFIG).map(([key, config]) => (
            <div key={key} className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${config.color}`} />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                {key}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
