import { useQuery } from '@tanstack/react-query'
import api from '../lib/api'
import { POLL_INTERVAL } from '../lib/config'
import type { SensorStatus } from '../lib/types'

export function useStatus() {
  return useQuery<SensorStatus[]>({
    queryKey: ['status'],
    queryFn: () => api.get('/api/sensors/status').then(res => res.data),
    refetchInterval: POLL_INTERVAL,
  })
}
