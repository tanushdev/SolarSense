export interface HealthStatus {
  status: string
  model_version: string
  gpu_available: boolean
  device: string
}

export interface SimilarEvent {
  date: string
  flare_class: string
  similarity: number
  lead_time: number
  description: string
}

export interface ForecastResponse {
  timestamp: string
  flare_probability: number
  uncertainty: number
  lower_bound: number
  upper_bound: number
  predicted_class: string
  class_probs: Record<string, number>
  lead_time_minutes: number
  alert_level: 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED'
  model: string
  data_timestamp: string
  similar_events: SimilarEvent[]
  physics_reason: string
}

export interface NowcastEvent {
  start_time: string
  peak_time: string
  end_time: string
  peak_soft_flux: number
  peak_hard_flux: number
  flare_class: string
  flare_subclass: number
  confirmation: string
  quality: number
}

export interface LightCurvePoint {
  timestamp: string
  soft_flux: number
  hard_flux: number
}

export interface ModelMetrics {
  tss: number
  hss: number
  brier: number
  auc: number
  ece: number
  avg_lead_time: number
  false_alarm_rate: number
  precision: number
  recall: number
  f1: number
}

export interface PerformanceMetrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  pr_auc: number
  tpr: number
  far: number
  tss: number
  hss: number
}

export interface SystemStatus {
  api: 'healthy' | 'degraded' | 'down'
  model: 'loaded' | 'loading' | 'unavailable'
  last_update: string
  uptime_hours: number
}

export interface MetricsResponse extends ModelMetrics {
  roc_auc: number
  accuracy: number
  model: string
  total_predictions: number
  correct_predictions: number
  prediction_accuracy: number
}

export interface ModelsResponse {
  active_model: string
  available_models: { name: string; tag: string; path: string }[]
}

export interface AlertsResponse {
  alert: boolean
  alert_level: string
  flare_class: string
  probability: number
  lead_time_minutes: number
  timestamp: string
}

export interface NoaaData {
  timestamp: string
  flux_w_m2: number | null
  flux_class: string
  flare_class: string | null
  flare_peak_time: string | null
  data_time: string | null
  status: 'live' | 'cached' | 'stale' | 'initializing'
  error: string | null
}

export interface ValidationMatch {
  our_peak: string
  our_class: string
  goes_peak: string
  goes_class: string
  delta_min: number
  our_hard_flux: number
}

export interface DatasetInfo {
  path: string; exists: boolean; files: number; size_mb: number
}

export interface DatasetsResponse {
  solexs: DatasetInfo; hel1os: DatasetInfo; processed: DatasetInfo
  catalogs: DatasetInfo; models: DatasetInfo
}

export interface ValidationReport {
  period: { start: string; end: string }
  our_events: number
  goes_events: number
  matched: number
  missed: number
  false_alarms: number
  detection_rate: number
  precision: number
  recall: number
  seed_source: string
  matches: ValidationMatch[]
}
