import React from 'react';
import { Phone, HeartPulse, Zap } from 'lucide-react';

export default function ActionPanel({ prediction }) {
  if (prediction !== 'VFib') return null;

  return (
    <div className="bg-red-50 border-2 border-red-500 rounded-xl p-6 mb-6 shadow-sm">
      <h3 className="text-red-800 font-bold text-lg mb-4 uppercase tracking-wider flex items-center gap-2">
        <span className="bg-red-500 text-white w-6 h-6 flex items-center justify-center rounded-full text-sm">!</span>
        Immediate Action Required
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border text-red-900 border-red-200 rounded-lg p-5 shadow-sm flex flex-col items-center text-center">
          <Phone className="text-red-600 mb-3" size={32} />
          <h4 className="font-bold text-[1.1rem] leading-tight mb-1">Call Resuscitation Team</h4>
          <p className="text-sm font-medium text-red-700/80">Activate code blue immediately</p>
        </div>
        
        <div className="bg-white border text-red-900 border-red-200 rounded-lg p-5 shadow-sm flex flex-col items-center text-center">
          <HeartPulse className="text-red-600 mb-3" size={32} />
          <h4 className="font-bold text-[1.1rem] leading-tight mb-1">Begin CPR</h4>
          <p className="text-sm font-medium text-red-700/80">If patient is unresponsive</p>
        </div>
        
        <div className="bg-white border text-red-900 border-red-200 rounded-lg p-5 shadow-sm flex flex-col items-center text-center">
          <Zap className="text-red-600 mb-3" size={32} />
          <h4 className="font-bold text-[1.1rem] leading-tight mb-1">Prepare Defibrillator</h4>
          <p className="text-sm font-medium text-red-700/80">Immediate shock required</p>
        </div>
      </div>
    </div>
  );
}
