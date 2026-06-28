import type { ForecastResponse } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function PredictionCard({ forecast, className }: { forecast: ForecastResponse; className?: string }) {
  const pct = Math.round(forecast.flare_probability * 100)

  return (
    <Card className={cn("flex flex-col items-center p-6", className)}>
      <span className={cn("text-[40px] font-light tracking-tight text-bug-text font-mono")}>{pct}%</span>
      <span className="mono-label text-bug-muted mt-1">PROBABILITY</span>

      <div className="mt-4 text-center">
        <span className="text-[28px] font-light tracking-tight text-bug-text font-mono">{forecast.predicted_class}</span>
        <div className="mono-sm text-bug-muted mt-0.5">PREDICTED CLASS</div>
      </div>

      <div className="flex gap-3 mt-4 mono-sm text-bug-muted">
        {Object.entries(forecast.class_probs).map(([cls, prob]) => (
          <span key={cls}>{cls}: {(prob * 100).toFixed(0)}%</span>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 w-full mt-5 pt-4 border-t border-bug-hairline mono-sm">
        <div className="text-bug-muted">LEAD TIME</div><div className="text-right text-bug-text font-mono">{forecast.lead_time_minutes.toFixed(0)} min</div>
        <div className="text-bug-muted">UNCERTAINTY</div><div className="text-right text-bug-text font-mono">±{(forecast.uncertainty * 100).toFixed(0)}%</div>
        <div className="text-bug-muted">ALERT</div><div className="text-right text-bug-text font-mono">{forecast.alert_level}</div>
        <div className="text-bug-muted">MODEL</div><div className="text-right text-bug-text font-mono">{forecast.model}</div>
      </div>
    </Card>
  )
}
