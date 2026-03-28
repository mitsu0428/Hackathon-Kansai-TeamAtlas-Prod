import { useStatus } from '../hooks/useStatus'
import { useScores } from '../hooks/useScores'
import DemoPanel from './DemoPanel'
import SensorMap from './SensorMap'
import SoundPulse from './SoundPulse'

function Dashboard() {
  const { data: statuses, isLoading: statusLoading, error: statusError } = useStatus()
  const { data: scores, isLoading: scoresLoading, error: scoresError } = useScores()

  if (statusLoading || scoresLoading) {
    return (
      <div className="dashboard">
        <h1 className="dashboard-title">空間音 異常検知システム</h1>
        <div className="skeleton-grid">
          <div className="skeleton-card" />
          <div className="skeleton-card" />
          <div className="skeleton-card" />
        </div>
        <div className="skeleton-chart" />
        <div className="skeleton-list" />
      </div>
    )
  }

  if (statusError || scoresError) {
    return (
      <div className="dashboard-error-container">
        <h2>データの取得に失敗しました</h2>
        <p>APIサーバーが起動していますか？ ({window.location.origin})</p>
        <button
          className="demo-btn demo-btn--primary"
          onClick={() => window.location.reload()}
        >
          再読み込み
        </button>
      </div>
    )
  }

  return (
    <main className="dashboard">
      <div className="dashboard-header">
        <h1 className="dashboard-title">空間音 異常検知システム</h1>
      </div>

      <div className="dashboard-top">
        <section className="dashboard-chart" aria-label="Sound Pulse Chart">
          <SoundPulse scores={scores ?? []} />
        </section>
        <section className="dashboard-controls" aria-label="Demo Controls">
          <DemoPanel />
          <SensorMap statuses={statuses ?? []} />
        </section>
      </div>

    </main>
  )
}

export default Dashboard
