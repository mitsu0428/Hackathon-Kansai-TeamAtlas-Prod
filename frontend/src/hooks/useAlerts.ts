import { useQuery } from '@tanstack/react-query'
import api from '../lib/api'
import { POLL_INTERVAL } from '../lib/config'
import type { Alert } from '../lib/types'

export function useAlerts() {
  return useQuery<Alert[]>({
    queryKey: ['alerts'],
    queryFn: () => api.get('/api/alerts').then(res => res.data),
    refetchInterval: POLL_INTERVAL,
  })
}
