import React from 'react';
import { Phone, HeartPulse, Zap } from 'lucide-react';

const ACTIONS = [
  {
    icon: Phone,
    title: 'Call Resuscitation Team',
    desc: 'Activate code blue immediately',
    step: '1',
  },
  {
    icon: HeartPulse,
    title: 'Begin CPR',
    desc: 'If patient is unresponsive',
    step: '2',
  },
  {
    icon: Zap,
    title: 'Prepare Defibrillator',
    desc: 'Immediate shock required',
    step: '3',
  },
];

export default function ActionPanel({ prediction }) {
  if (prediction !== 'VFib') return null;

  return (
    <div 
      className="animate-float-in animate-float-in-2 glass-card border-red-500/30 bg-red-950/30 p-6 md:p-8"
      id="action-panel"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="w-7 h-7 rounded-lg bg-red-500/20 flex items-center justify-center">
          <span className="text-red-400 font-black text-sm">!</span>
        </div>
        <h3 className="text-red-300 font-extrabold text-sm uppercase tracking-[0.15em]">
          Immediate Action Required
        </h3>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {ACTIONS.map((action) => (
          <div 
            key={action.step}
            className="
              relative group rounded-xl border border-red-500/20 
              bg-red-950/40 hover:bg-red-900/40
              p-5 flex flex-col items-center text-center
              transition-all duration-300
              hover:border-red-400/40 hover:shadow-lg hover:shadow-red-500/10
            "
          >
            {/* Step number */}
            <span className="absolute top-3 left-3 w-6 h-6 rounded-lg bg-red-500/15 text-red-400 text-xs font-black flex items-center justify-center">
              {action.step}
            </span>

            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-red-500/20 to-red-600/10 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-300">
              <action.icon className="text-red-400" size={28} />
            </div>
            <h4 className="font-bold text-base text-white leading-tight mb-1">
              {action.title}
            </h4>
            <p className="text-sm font-medium text-red-300/70">
              {action.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
