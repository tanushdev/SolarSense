import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '@/services/api'
import type { ForecastResponse, ValidationReport, NoaaData, MetricsResponse, ModelsResponse, AlertsResponse, DatasetsResponse } from '@/types'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  })
}

export function useForecast() {
  return useQuery({
    queryKey: ['forecast'],
    queryFn: api.forecast,
    refetchInterval: 30_000,
  })
}

export function useHistory() {
  return useQuery({
    queryKey: ['history'],
    queryFn: api.history,
    refetchInterval: 60_000,
  })
}

export function useLightCurve(hours = 1) {
  return useQuery({
    queryKey: ['lightcurve', hours],
    queryFn: () => api.lightcurve(hours),
    refetchInterval: 15_000,
  })
}

export function useNowcastMutation() {
  return useMutation({
    mutationFn: api.nowcast,
  })
}

export function usePredictMutation() {
  return useMutation({
    mutationFn: api.predict,
  })
}

export function useValidation(start?: string, end?: string) {
  return useQuery({
    queryKey: ['validation', start, end],
    queryFn: async (): Promise<ValidationReport> => {
      const res = await fetch(`/validation/report?start_date=${start ?? '2024-07-01'}&end_date=${end ?? '2024-12-31'}`)
      if (!res.ok) throw new Error('Validation API failed')
      return res.json()
    },
    refetchInterval: 120_000,
  })
}

export function useNoaa() {
  return useQuery<NoaaData>({
    queryKey: ['noaa'],
    queryFn: api.noaa,
    refetchInterval: 60_000,
  })
}

export function useModels() {
  return useQuery<ModelsResponse>({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: 120_000,
  })
}

export function useAlerts() {
  return useQuery<AlertsResponse>({
    queryKey: ['alerts'],
    queryFn: api.alerts,
    refetchInterval: 30_000,
  })
}

export function useDatasets() {
  return useQuery<DatasetsResponse>({
    queryKey: ['datasets'],
    queryFn: api.datasets,
    refetchInterval: 120_000,
  })
}

export function useMetrics() {
  return useQuery<MetricsResponse>({
    queryKey: ['metrics'],
    queryFn: api.metrics,
    refetchInterval: 60_000,
  })
}

