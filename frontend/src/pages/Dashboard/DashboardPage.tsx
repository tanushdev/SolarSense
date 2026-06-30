import { useHealth, useForecast, useNoaa, useMetrics, useLightCurveLive } from "@/hooks/useApi"
import { MetricCard } from "@/components/domain/MetricCard"
import { AlertBanner } from "@/components/domain/AlertBanner"
import { PredictionCard } from "@/components/domain/PredictionCard"
import { NOAAStatus } from "@/components/domain/NOAAStatus"
import { LightCurveChart } from "@/components/charts/LightCurveChart"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export default function DashboardPage() {
  const { data: health } = useHealth()
  const { data: forecast } = useForecast()
  const { data: live } = useLightCurveLive()
  const { data: noaa } = useNoaa()
  const { data: metrics } = useMetrics()
  const lightcurve = live?.points?.filter(p => p.soft_flux != null).map(p => ({
    timestamp: p.timestamp, soft_flux: p.soft_flux!, hard_flux: p.hard_flux
  }))
  const forecastPoints = live?.points?.filter(p => p.probability != null).map(p => ({
    timestamp: p.timestamp,
    probability: p.probability!,
    soft_flux: p.forecast_soft,
    hard_flux: p.forecast_hard,
  }))

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="display-lg text-bug-text">MISSION CONTROL</h1>
            <p className="mono-sm text-bug-muted mt-1">Solar flare nowcasting & prediction system</p>
          </div>
          <div className="flex items-center gap-3 mono-sm">
            <span className={cn(health?.status === "healthy" ? "text-bug-success" : "text-bug-muted-soft")}>
              {health?.status?.toUpperCase() ?? "OFFLINE"}
            </span>
            <span className={cn(
              forecast?.alert_level === "RED" ? "text-bug-warning" : forecast?.alert_level === "ORANGE" ? "text-bug-body-strong" : "text-bug-muted"
            )}>
              {forecast?.alert_level ?? "INIT"} ALERT
            </span>
          </div>
        </div>
      </div>

      <AlertBanner
        show={(forecast?.flare_probability ?? 0) > 0.5}
        level={(forecast?.flare_probability ?? 0) > 0.7 ? "high" : "low_confidence"}
        message={`${forecast?.predicted_class} PROBABILITY ${((forecast?.flare_probability ?? 0) * 100).toFixed(0)}% · LEAD ~${forecast?.lead_time_minutes.toFixed(0)} MIN`}
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="FLARE PROB" value={forecast ? `${(forecast.flare_probability * 100).toFixed(0)}%` : "—"} />
        <MetricCard label="PREDICTED CLASS" value={forecast?.predicted_class ?? "—"} />
        <MetricCard label="LEAD TIME" value={forecast ? `${forecast.lead_time_minutes.toFixed(0)}` : "—"} unit="min" />
        <MetricCard label="MODEL TSS" value={metrics?.tss != null ? metrics.tss.toFixed(3) : "—"} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <LightCurveChart data={lightcurve} forecast={forecastPoints} probability={forecast?.flare_probability} />
        </div>
        <div className="space-y-6">
          {forecast && <PredictionCard forecast={forecast} />}
          <NOAAStatus data={noaa} isLoading={!noaa} />
          <Card>
            <CardContent className="p-4 space-y-2">
              <div className="mono-label text-bug-muted">SYSTEM</div>
              <div className="grid grid-cols-2 gap-2 mono-sm">
                <div><span className="text-bug-muted">API</span><br /><span className={cn(health?.status === "healthy" ? "text-bug-success" : "text-bug-warning")}>{health?.status ?? "—"}</span></div>
                <div><span className="text-bug-muted">GPU</span><br /><span className="text-bug-text">{health?.gpu_available ? "AVAILABLE" : "N/A"}</span></div>
                <div><span className="text-bug-muted">MODEL</span><br /><span className="text-bug-text font-mono">{forecast?.model ?? "—"}</span></div>
                <div><span className="text-bug-muted">DEVICE</span><br /><span className="text-bug-text">{health?.device ?? "—"}</span></div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
