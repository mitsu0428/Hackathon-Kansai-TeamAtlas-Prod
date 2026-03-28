/**
 * 距離スコアを人が読める形式に変換する。
 * 0に近いほど「ふだん通り」、閾値(0.5)を超えると「異常」。
 */
export function formatDistance(distance: number, threshold: number = 0.6): string {
  const ratio = distance / threshold
  if (ratio < 0.3) return `${distance.toFixed(2)} (とても安定)`
  if (ratio < 0.6) return `${distance.toFixed(2)} (安定)`
  if (ratio < 0.9) return `${distance.toFixed(2)} (やや変化あり)`
  if (ratio < 1.0) return `${distance.toFixed(2)} (注意)`
  if (ratio < 1.5) return `${distance.toFixed(2)} (異常)`
  return `${distance.toFixed(2)} (重大な異常)`
}

/**
 * 距離スコアに応じたCSSクラスを返す。
 */
export function distanceClass(distance: number, threshold: number = 0.6): string {
  if (distance >= threshold) return 'distance--anomaly'
  if (distance >= threshold * 0.8) return 'distance--warning'
  return 'distance--normal'
}
