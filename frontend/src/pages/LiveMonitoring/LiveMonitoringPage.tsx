import { useForecast, useNoaa, useLightCurveLive } from "@/hooks/useApi"
import { MetricCard } from "@/components/domain/MetricCard"
import { NOAAStatus } from "@/components/domain/NOAAStatus"
import { LightCurveChart } from "@/components/charts/LightCurveChart"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export default function LiveMonitoringPage() {
  const { data: live } = useLightCurveLive()
  const { data: forecast } = useForecast()
  const { data: noaa } = useNoaa()
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
            <h1 className="display-lg text-bug-text">LIVE MONITOR</h1>
            <p className="mono-sm text-bug-muted mt-1">Real-time X-ray flux & flare detection</p>
          </div>
          <span className={cn("mono-sm", forecast?.alert_level === "RED" ? "text-bug-warning" : forecast?.alert_level === "ORANGE" ? "text-bug-body-strong" : "text-bug-muted")}>
            {forecast?.alert_level ?? "INIT"} ALERT
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="FLUX (SOFT)" value={lightcurve?.length ? (lightcurve[lightcurve.length - 1].soft_flux?.toExponential(2) ?? "—") : "—"} />
        <MetricCard label="FLUX (HARD)" value={lightcurve?.length ? (lightcurve[lightcurve.length - 1].hard_flux?.toExponential(2) ?? "—") : "—"} />
        <MetricCard label="FLARE PROB" value={forecast ? `${(forecast.flare_probability * 100).toFixed(0)}%` : "—"} />
        <MetricCard label="LEAD TIME" value={forecast ? `${forecast.lead_time_minutes.toFixed(0)}` : "—"} unit="min" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <LightCurveChart data={lightcurve} forecast={forecastPoints} probability={forecast?.flare_probability} />
        </div>
        <div className="space-y-6">
          <NOAAStatus data={noaa} isLoading={!noaa} />
          {forecast && (
            <Card>
              <CardContent className="p-4 space-y-3">
                <div className="mono-label text-bug-muted">FORECAST</div>
                <div className="flex justify-between mono-sm"><span className="text-bug-muted">PROBABILITY</span><span className="text-bug-text">{(forecast.flare_probability * 100).toFixed(1)}%</span></div>
                <div className="flex justify-between mono-sm"><span className="text-bug-muted">CLASS</span><span className="text-bug-text">{forecast.predicted_class}</span></div>
                <div className="flex justify-between mono-sm"><span className="text-bug-muted">LEAD</span><span className="text-bug-text">{forecast.lead_time_minutes.toFixed(0)} min</span></div>
                {forecast.physics_reason && <p className="body-serif-sm text-bug-body mt-2">{forecast.physics_reason}</p>}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
