import React from 'react';
import { Activity } from 'lucide-react';

export default function AlertBanner({ prediction }) {
  if (prediction === 'Normal' || !prediction) return null;

  const isCritical = prediction === 'VFib';
  const bgColor = isCritical ? 'bg-red-600' : 'bg-amber-500';
  const Title = isCritical ? 'Ventricular Fibrillation' : 'Atrial Fibrillation';
  const Subtitle = isCritical 
    ? 'Chaotic, life-threatening rhythm. Heart is not pumping effectively.'
    : 'Irregular, rapid heart rate. Increased risk of stroke and heart failure.';

  return (
    <div className={`w-full ${bgColor} text-white rounded-xl shadow-lg p-6 mb-6 flex items-center gap-6`}>
      <div className="relative flex-shrink-0">
        <div className="absolute inset-0 bg-white opacity-20 rounded-full animate-ping"></div>
        <div className="bg-white text-red-600 rounded-full p-4 relative z-10">
          <Activity size={40} strokeWidth={2.5} />
        </div>
      </div>
      
      <div>
        <h2 className="text-3xl font-extrabold tracking-tight mb-1 flex items-center gap-3">
          {Title}
          {isCritical && (
            <span className="inline-block w-4 h-4 rounded-full bg-red-200 animate-pulse border border-white"></span>
          )}
        </h2>
        <p className="text-red-50 text-xl font-medium tracking-wide">
          {Subtitle}
        </p>
      </div>
    </div>
  );
}
