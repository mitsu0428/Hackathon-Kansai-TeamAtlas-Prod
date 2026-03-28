// ポーリング間隔 (ms)
// VITE_POLL_INTERVAL で上書き可能 (GPU: 2000, CPU: 5000)
export const POLL_INTERVAL = Number(import.meta.env.VITE_POLL_INTERVAL) || 5000
