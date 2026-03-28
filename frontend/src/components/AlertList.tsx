import { useMemo } from 'react'
import type { Alert, ScorePoint } from '../lib/types'
import { formatDistance, distanceClass } from '../lib/formatDistance'
import { formatRelativeTime } from '../lib/formatRelativeTime'
import { SENSOR_LABELS } from '../lib/constants'

interface AlertListProps {
  alerts: Alert[]
  scores: ScorePoint[]
  threshold?: number
}

const URGENCY_LABEL: Record<string, string> = {
  low: '低',
  medium: '中',
  high: '高',
  critical: '緊急',
}

interface LogEntry {
  key: string
  sensorId: string
  timestamp: string
  distance: number
  isAnomaly: boolean
  alert: Alert | null
}

interface LogResult {
  latestPerSensor: LogEntry[]
  anomalies: LogEntry[]
}

function buildLog(alerts: Alert[], scores: ScorePoint[], threshold: number): LogResult {
  const alertMap = new Map<string, Alert>()
  for (const a of alerts) {
    alertMap.set(`${a.sensor_id}:${a.timestamp}`, a)
  }

  const entries: LogEntry[] = scores.map((s, i) => {
    const isAnomaly = s.distance >= threshold
    const alert = alertMap.get(`${s.sensor_id}:${s.timestamp}`) ?? null
    return {
      key: `${s.sensor_id}-${s.timestamp}-${i}`,
      sensorId: s.sensor_id,
      timestamp: s.timestamp,
      distance: s.distance,
      isAnomaly,
      alert,
    }
  })

  // Latest per sensor
  const latestMap = new Map<string, LogEntry>()
  const sorted = [...entries].sort((a, b) => a.timestamp.localeCompare(b.timestamp))
  for (const entry of sorted) {
    latestMap.set(entry.sensorId, entry)
  }
  const latestPerSensor = [...latestMap.values()]

  // Anomalies only, deduplicated by sensor+timestamp, newest first
  const seen = new Set<string>()
  const anomalies: LogEntry[] = []
  for (const e of [...entries].reverse()) {
    if (!e.isAnomaly) continue
    const dedupKey = `${e.sensorId}:${e.timestamp}`
    if (seen.has(dedupKey)) continue
    seen.add(dedupKey)
    anomalies.push(e)
    if (anomalies.length >= 50) break
  }

  return { latestPerSensor, anomalies }
}

function AlertList({ alerts, scores, threshold = 0.55 }: AlertListProps) {
  const { latestPerSensor, anomalies } = useMemo(
    () => buildLog(alerts, scores, threshold),
    [alerts, scores, threshold]
  )

  return (
    <div className="alert-list">
      <h2 className="section-title">検知ログ</h2>

      {/* Status bar */}
      {latestPerSensor.length > 0 && (
        <div className="alert-status-bar">
          {latestPerSensor.map(entry => (
            <div
              key={entry.sensorId}
              className={`alert-status-chip ${entry.isAnomaly ? 'alert-status-chip--anomaly' : ''}`}
            >
              <span
                className={`alert-dot ${entry.isAnomaly ? 'alert-dot--anomaly' : 'alert-dot--normal'}`}
                role="img"
                aria-label={entry.isAnomaly ? "異常" : "正常"}
              />
              <span className="alert-status-sensor">
                {SENSOR_LABELS[entry.sensorId] ?? entry.sensorId}
              </span>
              <span className="alert-status-score">
                {entry.distance.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Anomaly timeline */}
      {anomalies.length === 0 ? (
        <p className="alert-empty">異常は検知されていません</p>
      ) : (
        <ul className="alert-items">
          {anomalies.map(entry => (
            <li key={entry.key} className="alert-item alert-item--anomaly">
              <div className="alert-header">
                <span className="alert-sensor">
                  <span className="alert-dot alert-dot--anomaly" role="img" aria-label="異常" />
                  {SENSOR_LABELS[entry.sensorId] ?? entry.sensorId}
                </span>
                <span className="alert-time">{formatRelativeTime(entry.timestamp)}</span>
              </div>
              <div className="alert-body">
                <span className={`alert-distance ${distanceClass(entry.distance)}`}>
                  {formatDistance(entry.distance)}
                </span>
                <span className="alert-tag alert-tag--anomaly">異常</span>
                {entry.alert?.intent && (
                  <span className={`alert-urgency alert-urgency--${entry.alert.intent.urgency}`}>
                    {URGENCY_LABEL[entry.alert.intent.urgency] ?? entry.alert.intent.urgency}
                  </span>
                )}
              </div>
              {entry.alert?.intent && (
                <div className="alert-judgment-box">
                  {entry.alert.intent.judgment}
                </div>
              )}
              {(() => {
                const missing = entry.alert?.anomaly
                  ? entry.alert.anomaly.baseline_categories.filter(c => !entry.alert!.anomaly.matched_labels.includes(c))
                  : []
                return missing.length > 0 ? (
                  <div className="alert-missing">
                    消えた音: {missing.join(', ')}
                  </div>
                ) : null
              })()}
              {(entry.alert?.anomaly?.matched_labels?.length ?? 0) > 0 && (
                <div className="alert-matched">
                  検出音: {entry.alert!.anomaly.matched_labels.join(', ')}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default AlertList
