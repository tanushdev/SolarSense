import type {
  HealthStatus,
  ForecastResponse,
  NowcastEvent,
  LightCurvePoint,
  ModelMetrics,
  MetricsResponse,
  ModelsResponse,
  AlertsResponse,
  NoaaData,
  DatasetsResponse,
  ForecastTimeseries,
  LiveLightCurveResponse,
} from '@/types'

const BASE_URL = '/api'

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`)
  }
  return res.json()
}

export const api = {
  health: () => fetchJson<HealthStatus>('/health'),

  predict: (data: { timestamps: number[]; soft_flux: number[]; hard_flux: number[] }) =>
    fetchJson<ForecastResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  nowcast: (data: { timestamps: number[]; soft_flux: number[]; hard_flux: number[] }) =>
    fetchJson<{ is_flare: boolean; flare_class?: string; confidence: string }>('/nowcast', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  history: () => fetchJson<NowcastEvent[]>('/history'),

  lightcurve: (hours = 1) =>
    fetchJson<LightCurvePoint[]>(`/lightcurve?hours=${hours}`),

  forecast: () =>
    fetchJson<ForecastResponse>('/forecast'),

  noaa: () =>
    fetchJson<NoaaData>('/noaa'),

  metrics: () =>
    fetchJson<MetricsResponse>('/metrics'),

  models: () =>
    fetchJson<ModelsResponse>('/models'),

  alerts: () =>
    fetchJson<AlertsResponse>('/alerts'),

  datasets: () =>
    fetchJson<DatasetsResponse>('/datasets'),

  forecastTimeseries: (hours = 72) =>
    fetchJson<ForecastTimeseries>(`/forecast/timeseries?hours=${hours}`),

  lightcurveLive: () =>
    fetchJson<LiveLightCurveResponse>('/lightcurve/live'),
}
