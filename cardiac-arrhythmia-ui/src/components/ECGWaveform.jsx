import React, { useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  ResponsiveContainer, ReferenceArea, Tooltip 
} from 'recharts';

export default function ECGWaveform({ data, anomalyRegion }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    const step = Math.max(1, Math.floor(data.length / 800));
    const sampled = [];
    for (let i = 0; i < data.length; i += step) {
      sampled.push({
        time: parseFloat((i / 360).toFixed(3)),
        voltage: data[i],
        rawIndex: i
      });
    }
    return sampled;
  }, [data]);

  if (!chartData.length) return null;

  const voltages = chartData.map(d => d.voltage);
  const minVoltage = Math.min(...voltages);
  const maxVoltage = Math.max(...voltages);
  const padding = (maxVoltage - minVoltage) * 0.15;

  let highlightStart = null;
  let highlightEnd = null;

  if (anomalyRegion) {
    highlightStart = parseFloat((anomalyRegion.startIndex / 360).toFixed(3));
    highlightEnd = parseFloat((anomalyRegion.endIndex / 360).toFixed(3));
  }

  return (
    <div className="glass-card p-6 h-full" id="ecg-waveform">
      {/* Header */}
      <div className="flex justify-between items-end mb-5">
        <div>
          <h3 className="text-white font-bold text-lg tracking-tight">ECG Trace</h3>
          <p className="text-slate-500 text-xs font-medium tracking-wide mt-0.5">
            Lead II • 25 mm/s • 10 mm/mV
          </p>
        </div>
        {anomalyRegion && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-xs font-bold text-amber-400 tracking-wide uppercase">
              Irregular activity detected
            </span>
          </div>
        )}
      </div>

      {/* Chart Container */}
      <div 
        className="w-full rounded-xl overflow-hidden border border-slate-700/50"
        style={{ 
          height: 280,
          background: 'linear-gradient(180deg, rgba(15,23,42,0.8) 0%, rgba(15,23,42,0.95) 100%)'
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="rgba(148,163,184,0.08)" 
              vertical={true} 
              horizontal={true} 
            />
            <XAxis 
              dataKey="time" 
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fontSize: 11, fill: '#64748b', fontWeight: 500 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155', strokeWidth: 1 }}
              label={{ 
                value: 'Time (seconds)', 
                position: 'bottom', 
                offset: 0, 
                fill: '#64748b', 
                fontSize: 11, 
                fontWeight: 600 
              }}
            />
            <YAxis 
              domain={[minVoltage - padding, maxVoltage + padding]}
              tick={{ fontSize: 11, fill: '#64748b', fontWeight: 500 }}
              tickLine={{ stroke: '#334155' }}
              axisLine={{ stroke: '#334155', strokeWidth: 1 }}
              label={{ 
                value: 'mV', 
                angle: -90, 
                position: 'insideLeft', 
                fill: '#64748b', 
                fontSize: 11, 
                fontWeight: 600 
              }}
              tickFormatter={(v) => v.toFixed(1)}
            />

            <Tooltip
              contentStyle={{
                background: 'rgba(15,23,42,0.95)',
                border: '1px solid rgba(148,163,184,0.2)',
                borderRadius: 12,
                boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
                color: '#f1f5f9',
                fontSize: 12,
                fontWeight: 600,
              }}
              formatter={(value) => [Number(value).toFixed(4) + ' mV', 'Amplitude']}
              labelFormatter={(value) => `Time: ${Number(value).toFixed(2)}s`}
            />
            
            {/* Anomaly highlight region */}
            {highlightStart !== null && highlightEnd !== null && (
              <ReferenceArea
                x1={highlightStart}
                x2={highlightEnd}
                fill="#f59e0b"
                fillOpacity={0.08}
                stroke="#f59e0b"
                strokeOpacity={0.3}
                strokeDasharray="4 4"
              />
            )}
            
            {/* ECG Line */}
            <Line
              type="monotone"
              dataKey="voltage"
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
