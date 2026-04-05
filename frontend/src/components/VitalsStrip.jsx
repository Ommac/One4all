import React from 'react';

function VitalCard({ title, value, unit, status, subtitle }) {
  let colorClass = 'text-gray-900';
  let borderClass = 'border-gray-200 bg-white';
  
  if (status === 'critical') {
    colorClass = 'text-red-700';
    borderClass = 'border-red-400 bg-red-50';
  } else if (status === 'warning') {
    colorClass = 'text-amber-700';
    borderClass = 'border-amber-400 bg-amber-50';
  } else if (status === 'normal') {
    colorClass = 'text-emerald-700';
    borderClass = 'border-emerald-200 bg-emerald-50';
  }

  return (
    <div className={`rounded-xl border-2 p-5 flex flex-col shadow-sm ${borderClass}`}>
      <span className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">{title}</span>
      <div className="flex items-baseline gap-1 mt-auto">
        <span className={`text-3xl font-extrabold tracking-tight ${colorClass}`}>{value}</span>
        {unit && <span className={`text-sm font-bold ${colorClass} opacity-80`}>{unit}</span>}
      </div>
      <span className={`text-xs font-semibold mt-2 ${colorClass} opacity-90`}>{subtitle}</span>
    </div>
  );
}

export default function VitalsStrip({ hr, pr, qrs, qtc, prediction, status }) {
  if (!prediction) return null;

  const isVFib = prediction === 'VFib';
  const isAFib = prediction === 'AFib';
  const isNormal = prediction === 'Normal';

  // Rhythm Classification Card setup
  let rhythmLabel = 'Normal Sinus Rhythm';
  let rhythmSub = 'Normal';
  let rhythmStatus = 'normal';
  
  if (isVFib) {
    rhythmLabel = 'VFib';
    rhythmSub = 'Life-threatening';
    rhythmStatus = 'critical';
  } else if (isAFib) {
    rhythmLabel = 'AFib';
    rhythmSub = 'Abnormal';
    rhythmStatus = 'warning';
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
      <VitalCard 
        title="Heart Rate" 
        value={isVFib || !hr ? '—' : hr} 
        unit={isVFib || !hr ? '' : 'bpm'} 
        status={status.hr}
        subtitle={isVFib || !hr ? 'Undetectable' : 'Normal: 60-100'}
      />
      <VitalCard 
        title="PR Interval" 
        value={isVFib || !pr ? '—' : pr} 
        unit={isVFib || !pr ? '' : 'ms'} 
        status={status.pr}
        subtitle={isVFib || !pr ? 'Indeterminate' : 'Normal: 120-200'}
      />
      <VitalCard 
        title="QRS Duration" 
        value={isVFib || !qrs ? '—' : qrs} 
        unit={isVFib || !qrs ? '' : 'ms'} 
        status={status.qrs}
        subtitle={isVFib || !qrs ? 'No QRS found' : 'Normal: <120'}
      />
      <VitalCard 
        title="QT / QTc" 
        value={isVFib || !qtc ? '—' : qtc} 
        unit={isVFib || !qtc ? '' : 'ms'} 
        status={status.qtc}
        subtitle={isVFib || !qtc ? 'Indeterminate' : 'Normal: <440'}
      />
      <VitalCard 
        title="Rhythm" 
        value={rhythmLabel} 
        status={rhythmStatus}
        subtitle={rhythmSub}
      />
    </div>
  );
}
