export interface Sensor {
  sensor_id: string
  name: string
  location: string
}

export interface AnomalyResult {
  sensor_id: string
  timestamp: string
  distance: number
  is_anomaly: boolean
  threshold: number
  matched_labels: string[]
  baseline_categories: string[]
}

export interface Intent {
  sensor_id: string
  timestamp: string
  judgment: string
  recommendation: string
  urgency: 'low' | 'medium' | 'high' | 'critical'
  supplement: string
}

export interface Alert {
  alert_id: string
  sensor_id: string
  timestamp: string
  anomaly: AnomalyResult
  intent: Intent | null
}

export interface SensorStatus {
  sensor_id: string
  name: string
  location: string
  is_active: boolean
  last_checked: string | null
  current_distance: number
  is_anomaly: boolean
}

export interface ScorePoint {
  sensor_id: string
  timestamp: string
  distance: number
}
