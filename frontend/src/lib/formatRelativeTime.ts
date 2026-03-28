export function formatRelativeTime(timestamp: string | null): string {
  if (!timestamp) return '未取得'
  const now = Date.now()
  const then = new Date(timestamp).getTime()
  const diffSeconds = Math.floor((now - then) / 1000)

  if (diffSeconds < 5) return 'たった今'
  if (diffSeconds < 60) return `${diffSeconds}秒前`
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}分前`
  if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}時間前`
  return `${Math.floor(diffSeconds / 86400)}日前`
}
