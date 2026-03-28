import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Dot,
} from 'recharts'
import type { ScorePoint } from '../lib/types'
import { SENSOR_LABELS, SENSOR_COLORS, SENSOR_ORDER } from '../lib/constants'

const ANOMALY_COLOR = '#DC2626'
const DEFAULT_COLOR = '#6B7280'
const THRESHOLD = 0.55

const ACTIVE_DOT_STYLE = { r: 6, stroke: '#FFFFFF', strokeWidth: 2 }
const TOOLTIP_CONTENT_STYLE = {
  backgroundColor: '#FFFFFF',
  border: '1px solid #E5E7EB',
  borderRadius: '6px',
  color: '#374151',
  fontSize: '0.75rem',
}
const TOOLTIP_LABEL_STYLE = { color: '#6B7280', marginBottom: '4px', fontWeight: 600 }

interface SoundPulseProps {
  scores: ScorePoint[]
  threshold?: number
}

interface SensorRow {
  index: number
  label: string
  value: number
  isLatest: boolean
}

function formatRelativeLabel(timestamp: string): string {
  const now = Date.now()
  const t = new Date(timestamp).getTime()
  const diffSec = Math.round((now - t) / 1000)
  if (diffSec < 5) return '今'
  if (diffSec < 60) return `${diffSec}秒前`
  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin}分前`
  return `${Math.round(diffMin / 60)}時間前`
}

function buildChartData(scores: ScorePoint[]): Map<string, SensorRow[]> {
  const result = new Map<string, SensorRow[]>()
  const sorted = [...scores].sort((a, b) => a.timestamp.localeCompare(b.timestamp))

  // Group by sensor
  const bySensor = new Map<string, ScorePoint[]>()
  for (const point of sorted) {
    if (!bySensor.has(point.sensor_id)) {
      bySensor.set(point.sensor_id, [])
    }
    bySensor.get(point.sensor_id)!.push(point)
  }

  for (const [sensorId, points] of bySensor) {
    const rows: SensorRow[] = points.map((point, i) => ({
      index: i + 1,
      label: formatRelativeLabel(point.timestamp),
      value: point.distance,
      isLatest: i === points.length - 1,
    }))
    result.set(sensorId, rows)
  }

  return result
}

interface AnomalyDotProps {
  cx?: number
  cy?: number
  value?: number
  stroke?: string
  payload?: SensorRow
}

function createAnomalyDot(threshold: number, sensorColor: string) {
  return function renderAnomalyDot(props: AnomalyDotProps) {
    const { cx, cy, value, payload } = props
    if (cx == null || cy == null || value == null) return null
    const isAnomaly = value >= threshold
    const isLatest = payload?.isLatest ?? false

    if (isLatest) {
      // Pulsing dot for latest point
      return (
        <g key={`latest-${cx}-${cy}`}>
          <circle
            cx={cx}
            cy={cy}
            r={8}
            fill={isAnomaly ? ANOMALY_COLOR : sensorColor}
            opacity={0.2}
          >
            <animate attributeName="r" values="6;12;6" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0.05;0.3" dur="2s" repeatCount="indefinite" />
          </circle>
          <Dot
            cx={cx}
            cy={cy}
            r={5}
            fill={isAnomaly ? ANOMALY_COLOR : sensorColor}
            stroke="#FFFFFF"
            strokeWidth={2}
          />
        </g>
      )
    }

    if (isAnomaly) {
      return (
        <Dot
          key={`${cx}-${cy}`}
          cx={cx}
          cy={cy}
          r={5}
          fill={ANOMALY_COLOR}
          stroke="#FFFFFF"
          strokeWidth={2}
        />
      )
    }

    return (
      <Dot
        key={`${cx}-${cy}`}
        cx={cx}
        cy={cy}
        r={3}
        fill={sensorColor}
        stroke="none"
        strokeWidth={0}
      />
    )
  }
}

function SoundPulse({ scores, threshold = THRESHOLD }: SoundPulseProps) {
  const sensorDataMap = useMemo(() => buildChartData(scores), [scores])
  const sensorIds = useMemo(() => {
    const keys = [...sensorDataMap.keys()]
    const ordered = SENSOR_ORDER.filter(id => keys.includes(id)) as string[]
    const rest = keys.filter(id => !(SENSOR_ORDER as readonly string[]).includes(id))
    return ordered.concat(rest)
  }, [sensorDataMap])
  const totalDetections = scores.length

  if (scores.length === 0) {
    return (
      <div className="sound-pulse">
        <div className="sound-pulse-header">
          <h2 className="section-title">音</h2>
        </div>
        <p className="sound-pulse-empty">
          右のデモ操作パネルからボタンを押すとデータが表示されます
        </p>
      </div>
    )
  }

  return (
    <div className="sound-pulse">
      <div className="sound-pulse-header">
        <h2 className="section-title">音</h2>
        <span className="sound-pulse-subtitle">
          {totalDetections}回の検知
        </span>
      </div>
      <div role="img" aria-label="時系列の音響距離スコアチャート">
        {sensorIds.map(id => {
          const rows = sensorDataMap.get(id)!
          const latest = rows[rows.length - 1]
          const sensorColor = SENSOR_COLORS[id] ?? DEFAULT_COLOR
          const dotRenderer = createAnomalyDot(threshold, sensorColor)
          return (
            <div key={id} className="sound-pulse-sensor-chart">
              <div className="sound-pulse-sensor-header">
                <span className="sound-pulse-sensor-label">
                  {SENSOR_LABELS[id] ?? id}
                </span>
                {latest && (
                  <span className={`sound-pulse-latest ${latest.value >= threshold ? 'sound-pulse-latest--anomaly' : ''}`}>
                    {latest.value.toFixed(2)}
                    <span className="sound-pulse-latest-label">
                      {latest.value >= threshold ? ' 異常' : ' 正常'}
                    </span>
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={rows} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                  <XAxis
                    dataKey="label"
                    stroke="#6B7280"
                    fontSize={10}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    stroke="#6B7280"
                    fontSize={10}
                    domain={[0, 1.1]}
                    tickCount={5}
                    tickFormatter={(v: number) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_CONTENT_STYLE}
                    labelStyle={TOOLTIP_LABEL_STYLE}
                    formatter={(value: unknown) => {
                      if (typeof value === 'number') {
                        const label = value >= threshold ? ' (異常)' : ' (正常)'
                        return [`${value.toFixed(2)}${label}`, SENSOR_LABELS[id] ?? id]
                      }
                      return [String(value), SENSOR_LABELS[id] ?? id]
                    }}
                    labelFormatter={(label: unknown) => String(label ?? '')}
                  />
                  <ReferenceLine
                    y={threshold}
                    stroke="#D97706"
                    strokeDasharray="6 3"
                    strokeWidth={1}
                    label={{ value: '注意', fill: '#D97706', position: 'right', fontSize: 9 }}
                  />
                  <ReferenceLine
                    y={1.0}
                    stroke={ANOMALY_COLOR}
                    strokeWidth={2}
                    label={{ value: '異常検知', fill: ANOMALY_COLOR, position: 'right', fontSize: 10 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={sensorColor}
                    strokeWidth={2}
                    dot={dotRenderer}
                    activeDot={ACTIVE_DOT_STYLE}
                    connectNulls
                    animationDuration={300}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default SoundPulse
