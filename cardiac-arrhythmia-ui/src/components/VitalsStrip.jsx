import React from 'react';
import { Heart, Activity, Timer, Waves, Radio } from 'lucide-react';

const STATUS_STYLES = {
  critical: {
    border: 'border-red-500/40',
    bg: 'bg-red-500/8',
    value: 'text-red-400',
    glow: 'shadow-red-500/10',
    dot: 'bg-red-500',
    iconBg: 'from-red-500/20 to-red-600/10',
  },
  warning: {
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/8',
    value: 'text-amber-400',
    glow: 'shadow-amber-500/10',
    dot: 'bg-amber-500',
    iconBg: 'from-amber-500/20 to-amber-600/10',
  },
  normal: {
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/5',
    value: 'text-emerald-400',
    glow: 'shadow-emerald-500/10',
    dot: 'bg-emerald-500',
    iconBg: 'from-emerald-500/20 to-emerald-600/10',
  },
  indeterminate: {
    border: 'border-slate-600/40',
    bg: 'bg-slate-800/30',
    value: 'text-slate-400',
    glow: '',
    dot: 'bg-slate-500',
    iconBg: 'from-slate-500/20 to-slate-600/10',
  },
};

function VitalCard({ title, value, unit, status, subtitle, icon: Icon, delay }) {
  const style = STATUS_STYLES[status] || STATUS_STYLES.indeterminate;

  return (
    <div 
      className={`
        animate-float-in animate-float-in-${delay}
        glass-card ${style.border} ${style.bg}
        rounded-2xl p-5 flex flex-col relative overflow-hidden
        hover:shadow-xl ${style.glow}
        transition-all duration-300
      `}
    >
      {/* Status dot */}
      <div className="flex items-center justify-between mb-3">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${style.iconBg} flex items-center justify-center`}>
          <Icon className={style.value} size={22} strokeWidth={2} />
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${style.dot} ${status === 'critical' ? 'animate-pulse' : ''}`} />
          <span className={`text-[10px] font-bold uppercase tracking-widest ${style.value}`}>
            {status === 'indeterminate' ? 'N/A' : status}
          </span>
        </div>
      </div>

      {/* Label */}
      <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-slate-500 mb-1.5">
        {title}
      </span>

      {/* Value */}
      <div className="flex items-baseline gap-1.5">
        <span className={`text-3xl font-extrabold tracking-tight ${style.value}`}>
          {value}
        </span>
        {unit && (
          <span className={`text-sm font-bold ${style.value} opacity-60`}>
            {unit}
          </span>
        )}
      </div>

      {/* Subtitle */}
      <span className="text-xs font-medium text-slate-500 mt-2 leading-tight">
        {subtitle}
      </span>
    </div>
  );
}

export default function VitalsStrip({ hr, pr, qrs, qtc, prediction, status }) {
  if (!prediction) return null;

  const isVFib = prediction === 'VFib';
  const isAFib = prediction === 'AFib';

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
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4" id="vitals-strip">
      <VitalCard 
        title="Heart Rate" 
        icon={Heart}
        value={isVFib || !hr ? '—' : hr} 
        unit={isVFib || !hr ? '' : 'bpm'} 
        status={isVFib ? 'critical' : (status.hr || 'indeterminate')}
        subtitle={isVFib || !hr ? 'Undetectable' : 'Normal: 60–100'}
        delay={1}
      />
      <VitalCard 
        title="PR Interval" 
        icon={Timer}
        value={isVFib || !pr ? '—' : pr} 
        unit={isVFib || !pr ? '' : 'ms'} 
        status={isVFib ? 'indeterminate' : (status.pr || 'indeterminate')}
        subtitle={isVFib || !pr ? 'Indeterminate' : 'Normal: 120–200'}
        delay={2}
      />
      <VitalCard 
        title="QRS Duration" 
        icon={Activity}
        value={isVFib || !qrs ? '—' : qrs} 
        unit={isVFib || !qrs ? '' : 'ms'} 
        status={isVFib ? 'indeterminate' : (status.qrs || 'indeterminate')}
        subtitle={isVFib || !qrs ? 'No QRS found' : 'Normal: <120'}
        delay={3}
      />
      <VitalCard 
        title="QT / QTc" 
        icon={Waves}
        value={isVFib || !qtc ? '—' : qtc} 
        unit={isVFib || !qtc ? '' : 'ms'} 
        status={isVFib ? 'indeterminate' : (status.qtc || 'indeterminate')}
        subtitle={isVFib || !qtc ? 'Indeterminate' : 'Normal: <440'}
        delay={4}
      />
      <VitalCard 
        title="Rhythm" 
        icon={Radio}
        value={rhythmLabel} 
        status={rhythmStatus}
        subtitle={rhythmSub}
        delay={5}
      />
    </div>
  );
}
