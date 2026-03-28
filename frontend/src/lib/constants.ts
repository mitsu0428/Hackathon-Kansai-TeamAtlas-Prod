export const SENSOR_ORDER = ['indoor', 'park', 'urban'] as const

export const SENSOR_LABELS: Record<string, string> = {
  indoor: '室内',
  park: '公園',
  urban: '都市',
}

export const SENSOR_COLORS: Record<string, string> = {
  urban: '#374151',
  indoor: '#722F37',
  park: '#9CA3AF',
}
