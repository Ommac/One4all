import React from 'react';

/* SVG Heart with ECG trace inside */
function HeartECGIcon({ size = 56, className = '' }) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 56 56" 
      fill="none" 
      className={className}
    >
      {/* Heart Shape */}
      <path 
        d="M28 50 C28 50, 6 36, 6 20 C6 12, 12 6, 20 6 C24 6, 27 8, 28 11 C29 8, 32 6, 36 6 C44 6, 50 12, 50 20 C50 36, 28 50, 28 50Z"
        fill="rgba(255,255,255,0.15)" 
        stroke="rgba(255,255,255,0.6)" 
        strokeWidth="1.5"
      />
      {/* ECG Trace inside heart */}
      <polyline 
        points="12,28 18,28 21,28 23,20 25,36 27,16 29,38 31,22 33,28 36,28 38,24 40,28 44,28"
        fill="none" 
        stroke="white" 
        strokeWidth="2" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
    </svg>
  );
}

const CONFIG = {
  VFib: {
    title: 'Ventricular Fibrillation',
    subtitle: 'Chaotic, life-threatening rhythm. Heart is not pumping effectively.',
    gradient: 'from-red-600 via-red-700 to-red-900',
    borderColor: 'border-red-500/50',
    glowColor: 'shadow-red-500/30',
    dotColor: 'bg-red-300',
    severity: 'CRITICAL',
    severityBg: 'bg-red-500/30 text-red-200',
  },
  AFib: {
    title: 'Atrial Fibrillation',
    subtitle: 'Irregular, rapid heart rate. Increased risk of stroke and heart failure.',
    gradient: 'from-amber-500 via-amber-600 to-amber-800',
    borderColor: 'border-amber-500/50',
    glowColor: 'shadow-amber-500/20',
    dotColor: 'bg-amber-300',
    severity: 'WARNING',
    severityBg: 'bg-amber-500/30 text-amber-200',
  },
};

export default function AlertBanner({ prediction }) {
  if (!prediction || prediction === 'Normal') return null;

  const config = CONFIG[prediction] || CONFIG.AFib;
  const isCritical = prediction === 'VFib';

  return (
    <div 
      className={`
        animate-float-in animate-float-in-1 w-full rounded-2xl border
        ${config.borderColor}
        bg-gradient-to-r ${config.gradient}
        shadow-2xl ${config.glowColor}
        p-6 md:p-8 flex items-center gap-6 relative overflow-hidden
      `}
      style={isCritical ? { animation: 'pulse-danger 2s ease-in-out infinite' } : {}}
      role="alert"
      id="critical-alert-banner"
    >
      {/* Scan line animation */}
      {isCritical && (
        <div 
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent)',
            animation: 'scan-line 3s linear infinite',
          }}
        />
      )}

      {/* Heart Icon with pulsing glow */}
      <div className="relative flex-shrink-0">
        <div 
          className={`absolute inset-0 rounded-full ${isCritical ? 'bg-white/15' : 'bg-white/10'} blur-xl`}
          style={isCritical ? { animation: 'pulse-dot 1.5s ease-in-out infinite' } : {}}
        />
        <div className={`relative z-10 ${isCritical ? 'animate-heartbeat' : ''}`}>
          <HeartECGIcon size={56} />
        </div>
      </div>

      {/* Text Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight leading-tight">
            {config.title}
          </h2>
          {isCritical && (
            <span 
              className={`w-3 h-3 rounded-full ${config.dotColor}`}
              style={{ animation: 'pulse-dot 1s ease-in-out infinite' }}
            />
          )}
        </div>
        <p className="text-white/80 text-base md:text-lg font-medium leading-relaxed">
          {config.subtitle}
        </p>
      </div>

      {/* Severity badge */}
      <div className="hidden md:flex flex-shrink-0">
        <span className={`px-4 py-2 rounded-xl text-xs font-black tracking-widest uppercase ${config.severityBg} backdrop-blur-sm`}>
          {config.severity}
        </span>
      </div>
    </div>
  );
}
