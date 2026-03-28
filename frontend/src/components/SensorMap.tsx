import type { SensorStatus } from '../lib/types'
import { formatDistance, distanceClass } from '../lib/formatDistance'
import { formatRelativeTime } from '../lib/formatRelativeTime'

interface SensorMapProps {
  statuses: SensorStatus[]
}

function getStatusColor(status: SensorStatus): string {
  if (!status.is_active) return '#D1D5DB'
  if (status.is_anomaly) return '#DC2626'
  return '#722F37'
}

function getStatusLabel(status: SensorStatus): string {
  if (!status.is_active) return '停止中'
  if (status.is_anomaly) return '異常'
  return '正常'
}

function SensorMap({ statuses }: SensorMapProps) {
  return (
    <div className="sensor-map">
      <h2 className="section-title">センサー一覧</h2>
      {statuses.length === 0 ? (
        <p className="sensor-map-empty">センサーが登録されていません</p>
      ) : (
        <div className="sensor-map-grid">
          {statuses.map(status => (
            <div key={status.sensor_id} className="sensor-map-item">
              <span
                className="status-dot"
                style={{ backgroundColor: getStatusColor(status) }}
                role="img"
                aria-label={getStatusLabel(status)}
              />
              <span className="status-label">{getStatusLabel(status)}</span>
              <div className="sensor-info">
                <span className="sensor-name">{status.name}</span>
                <span className="sensor-location">{status.location}</span>
              </div>
              <div className="sensor-score">
                <span className={`sensor-score-value ${distanceClass(status.current_distance)}`}>
                  {formatDistance(status.current_distance)}
                </span>
                <span className="sensor-score-time">
                  {formatRelativeTime(status.last_checked)}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default SensorMap
