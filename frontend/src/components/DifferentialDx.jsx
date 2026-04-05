import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function DifferentialDx({ probabilities }) {
  if (!probabilities) return null;

  // Expected probabilities: { VFib: 0.1, AFib: 0.8, Normal: 0.1 }
  const vfib = probabilities.VFib || 0;
  const afib = probabilities.AFib || 0;
  const normal = probabilities.Normal || 0;

  // Sort to find top two
  const sorted = [
    { name: 'VFib', prob: vfib, color: 'bg-red-500', label: 'Ventricular Fibrillation' },
    { name: 'AFib', prob: afib, color: 'bg-amber-500', label: 'Atrial Fibrillation' },
    { name: 'Normal', prob: normal, color: 'bg-emerald-500', label: 'Normal Sinus Rhythm' }
  ].sort((a, b) => b.prob - a.prob);

  const top1 = sorted[0];
  const top2 = sorted[1];
  const isClose = (top1.prob - top2.prob) < 0.1; // Within 10%

  return (
    <div className="bg-white border-2 border-slate-200 rounded-xl p-6 shadow-sm mb-6">
      <h3 className="text-slate-900 font-bold text-lg mb-4">Differential Diagnosis</h3>
      
      {isClose && (
        <div className="bg-slate-100 border-l-4 border-slate-400 p-4 mb-5 flex items-start gap-3 rounded-r-lg">
          <AlertCircle className="text-slate-500 flex-shrink-0 mt-0.5" size={20} />
          <p className="text-slate-700 font-semibold text-sm">
            Two rhythms are similarly likely — verify clinically
          </p>
        </div>
      )}

      <div className="space-y-5">
        {sorted.map((item) => (
          <div key={item.name}>
            <div className="flex justify-between text-sm font-bold mb-1.5">
              <span className="text-slate-700">{item.label}</span>
              <span className="text-slate-900">{Math.round(item.prob * 100)}%</span>
            </div>
            <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
              <div 
                className={`h-full ${item.color} rounded-full transition-all duration-500`}
                style={{ width: `${Math.max(item.prob * 100, 1)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
