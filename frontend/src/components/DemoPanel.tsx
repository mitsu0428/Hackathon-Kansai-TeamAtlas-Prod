import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../lib/api'

type Scenario = 'normal' | 'hvac_failure' | 'unusual_activity'
type SoundType = 'normal' | 'silence'
type SensorId = 'urban' | 'indoor' | 'park'

const SCENARIOS: { label: string; scenario: Scenario }[] = [
  { label: '正常', scenario: 'normal' },
  { label: '空調停止', scenario: 'hvac_failure' },
  { label: '深夜異常', scenario: 'unusual_activity' },
]

const SENSORS: { id: SensorId; label: string }[] = [
  { id: 'indoor', label: '室内' },
  { id: 'park', label: '公園' },
  { id: 'urban', label: '都市' },
]

const SOUND_TYPES: { label: string; soundType: SoundType; danger?: boolean }[] = [
  { label: '環境音', soundType: 'normal' },
  { label: '無音', soundType: 'silence', danger: true },
]

function DemoPanel() {
  const queryClient = useQueryClient()
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [selectedSensor, setSelectedSensor] = useState<SensorId>('urban')

  const invalidateAll = () => {
    setErrorMsg(null)
    void queryClient.invalidateQueries({ queryKey: ['status'] })
    void queryClient.invalidateQueries({ queryKey: ['alerts'] })
    void queryClient.invalidateQueries({ queryKey: ['scores'] })
  }

  const handleError = (err: Error) => {
    if (axios.isAxiosError(err) && err.response?.status) {
      setErrorMsg(`エラー (${String(err.response.status)})`)
    } else {
      setErrorMsg(err.message)
    }
  }

  const simulateMutation = useMutation({
    mutationFn: (scenario: Scenario) =>
      api.post('/api/demo/simulate', { scenario, duration_points: 3 }),
    onSuccess: invalidateAll,
    onError: handleError,
  })

  const generateMutation = useMutation({
    mutationFn: (soundType: SoundType) =>
      api.post('/api/demo/generate', {
        sensor_id: selectedSensor,
        sound_type: soundType,
        auto_detect: true,
      }),
    onSuccess: invalidateAll,
    onError: handleError,
  })

  const resetMutation = useMutation({
    mutationFn: () => api.post('/api/demo/reset'),
    onSuccess: invalidateAll,
    onError: handleError,
  })

  const detectMutation = useMutation({
    mutationFn: () => api.post('/api/demo/detect', {}),
    onSuccess: invalidateAll,
    onError: handleError,
  })

  const isLoading =
    simulateMutation.isPending
    || generateMutation.isPending
    || resetMutation.isPending
    || detectMutation.isPending

  return (
    <div className="demo-panel">
      <h2 className="demo-panel-title">デモ操作</h2>

      {errorMsg && (
        <div className="demo-panel-error" role="alert">{errorMsg}</div>
      )}

      {/* シナリオ */}
      <div className="demo-section">
        <span className="demo-section-label">シナリオ</span>
        <div className="demo-btn-row">
          {SCENARIOS.map(({ label, scenario }) => (
            <button
              key={scenario}
              className="demo-card"
              disabled={isLoading}
              onClick={() => simulateMutation.mutate(scenario)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* 音声生成 */}
      <div className="demo-section">
        <span className="demo-section-label">音声生成</span>
        <div className="demo-sensor-select">
          {SENSORS.map(({ id, label }) => (
            <button
              key={id}
              className={`demo-sensor-btn ${selectedSensor === id ? 'demo-sensor-btn--active' : ''}`}
              onClick={() => setSelectedSensor(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="demo-btn-row">
          {SOUND_TYPES.map(({ label, soundType, danger }) => (
            <button
              key={soundType}
              className={`demo-card ${danger ? 'demo-card--danger' : ''}`}
              disabled={isLoading}
              onClick={() => generateMutation.mutate(soundType)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* アクション */}
      <div className="demo-actions">
        <button
          className="demo-btn demo-btn--primary"
          disabled={isLoading}
          onClick={() => detectMutation.mutate()}
        >
          {isLoading ? '実行中...' : '即時スキャン'}
        </button>
        <button
          className="demo-btn demo-btn--danger"
          disabled={isLoading}
          onClick={() => resetMutation.mutate()}
        >
          クリア
        </button>
      </div>
    </div>
  )
}

export default DemoPanel
