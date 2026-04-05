import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceArea } from 'recharts';

export default function ECGWaveform({ data, anomalyRegion }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Downsample for performance (standard technique for large raw arrays)
    const step = Math.max(1, Math.floor(data.length / 800));
    const sampled = [];
    for (let i = 0; i < data.length; i += step) {
      sampled.push({
        time: (i / 360).toFixed(2), // 360 Hz is typical
        voltage: data[i],
        rawIndex: i
      });
    }
    return sampled;
  }, [data]);

  if (!chartData.length) return null;

  const minVoltage = Math.min(...chartData.map(d => d.voltage));
  const maxVoltage = Math.max(...chartData.map(d => d.voltage));
  const padding = (maxVoltage - minVoltage) * 0.15;

  let highlightStart = null;
  let highlightEnd = null;

  if (anomalyRegion) {
    highlightStart = (anomalyRegion.startIndex / 360).toFixed(2);
    highlightEnd = (anomalyRegion.endIndex / 360).toFixed(2);
  }

  return (
    <div className="bg-white border-2 border-slate-200 rounded-xl p-6 shadow-sm mb-6">
      <div className="flex justify-between items-end mb-4">
        <div>
          <h3 className="text-slate-900 font-bold text-lg">ECG Trace</h3>
          <p className="text-slate-500 text-sm font-medium">Lead II • 25 mm/s • 10 mm/mV</p>
        </div>
        {anomalyRegion && (
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-400"></span>
            <span className="text-sm font-bold text-amber-600 tracking-wide uppercase">Irregular activity detected here</span>
          </div>
        )}
      </div>

      <div className="w-full h-72 bg-rose-50/30 rounded-lg overflow-hidden border border-rose-100">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={true} horizontal={true} />
            <XAxis 
              dataKey="time" 
              type="number"
              domain={['dataMin', 'dataMax']}
              tick={{ fontSize: 12, fill: '#64748b', fontWeight: 600 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1', strokeWidth: 2 }}
              label={{ value: 'Time (seconds)', position: 'bottom', offset: 0, fill: '#64748b', fontSize: 13, fontWeight: 600 }}
            />
            <YAxis 
              domain={[minVoltage - padding, maxVoltage + padding]}
              tick={{ fontSize: 12, fill: '#64748b', fontWeight: 600 }}
              tickLine={{ stroke: '#cbd5e1' }}
              axisLine={{ stroke: '#cbd5e1', strokeWidth: 2 }}
              label={{ value: 'Amplitude (mV)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 13, fontWeight: 600 }}
              tickFormatter={(v) => v.toFixed(1)}
            />
            
            {highlightStart && highlightEnd && (
              <ReferenceArea
                x1={Number(highlightStart)}
                x2={Number(highlightEnd)}
                fill="#f59e0b"
                fillOpacity={0.15}
                stroke="#f59e0b"
                strokeOpacity={0.4}
                strokeDasharray="3 3"
              />
            )}
            
            <Line
              type="monotone"
              dataKey="voltage"
              stroke="#0f172a"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
