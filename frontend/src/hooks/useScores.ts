import { useQuery } from '@tanstack/react-query'
import api from '../lib/api'
import { POLL_INTERVAL } from '../lib/config'
import type { ScorePoint } from '../lib/types'

export function useScores(limit: number = 30) {
  return useQuery<ScorePoint[]>({
    queryKey: ['scores', limit],
    queryFn: () => api.get(`/api/sensors/scores?limit=${limit}`).then(res => res.data),
    refetchInterval: POLL_INTERVAL,
  })
}
