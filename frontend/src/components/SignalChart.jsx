import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { COLORS } from '../uiTheme'

export default function SignalChart({ signal }) {
  const chartData = useMemo(() => {
    if (!signal?.length) return []

    const step = signal.length > 1200 ? Math.ceil(signal.length / 700) : 1

    return signal
      .filter((_, index) => index % step === 0)
      .map((value, index) => ({
        index: index * step,
        time: ((index * step) / 360).toFixed(2),
        amplitude: Number(value),
      }))
  }, [signal])

  if (!chartData.length) {
    return (
      <div className="empty-state">
        ECG waveform will appear here after a valid file is uploaded.
      </div>
    )
  }

  const voltages = chartData.map((point) => point.amplitude)
  const minVoltage = Math.min(...voltages)
  const maxVoltage = Math.max(...voltages)
  const padding = Math.max((maxVoltage - minVoltage) * 0.1, 0.05)

  return (
    <div style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={COLORS.slate200} strokeDasharray="3 3" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 11, fill: COLORS.slate500 }}
            tickLine={{ stroke: COLORS.slate300 }}
            axisLine={{ stroke: COLORS.slate300 }}
            label={{ value: 'Time (s)', position: 'insideBottom', offset: -4, fill: COLORS.slate500, fontSize: 11 }}
          />
          <YAxis
            domain={[minVoltage - padding, maxVoltage + padding]}
            tick={{ fontSize: 11, fill: COLORS.slate500 }}
            tickLine={{ stroke: COLORS.slate300 }}
            axisLine={{ stroke: COLORS.slate300 }}
            tickFormatter={(value) => value.toFixed(2)}
            label={{ value: 'mV', angle: -90, position: 'insideLeft', fill: COLORS.slate500, fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 10,
              border: `1px solid ${COLORS.slate200}`,
              boxShadow: '0 12px 24px rgba(15, 23, 42, 0.08)',
            }}
            formatter={(value) => [Number(value).toFixed(4), 'Voltage']}
            labelFormatter={(value) => `Time ${value}s`}
          />
          <Line
            type="monotone"
            dataKey="amplitude"
            stroke={COLORS.primary}
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
