import { useForecast } from "@/hooks/useApi"
import { AlertBanner } from "@/components/domain/AlertBanner"
import { Card, CardContent } from "@/components/ui/card"
import { cn, classColor } from "@/lib/utils"

export default function ForecastsPage() {
  const { data: forecast, isLoading } = useForecast()

  if (isLoading || !forecast) return (
    <div className="flex items-center justify-center h-[60vh]">
      <span className="mono-label text-bug-muted">LOADING FORECAST DATA...</span>
    </div>
  )

  const fc = forecast

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="display-lg text-bug-text">FLARE FORECAST</h1>
            <p className="mono-sm text-bug-muted mt-1">Multi-class probabilistic prediction</p>
          </div>
          <span className={cn("mono-sm", fc.alert_level === "RED" ? "text-bug-warning" : fc.alert_level === "ORANGE" ? "text-bug-body-strong" : "text-bug-muted")}>
            {fc.alert_level} ALERT
          </span>
        </div>
      </div>

      <AlertBanner
        show={fc.flare_probability > 0.5}
        level={fc.flare_probability > 0.7 ? "high" : "low_confidence"}
        message={`${fc.predicted_class} · ${(fc.flare_probability * 100).toFixed(0)}% PROBABILITY · LEAD ~${fc.lead_time_minutes.toFixed(0)} MIN`}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Probability display */}
        <div className="space-y-6">
          <Card className="p-6 flex flex-col items-center">
            <div className="text-[56px] font-light tracking-tight text-bug-text font-mono">
              {(fc.flare_probability * 100).toFixed(0)}%
            </div>
            <span className="mono-label text-bug-muted mt-1">PROBABILITY</span>
            <div className="mt-3 text-center">
              <span className={cn("text-[32px] font-light tracking-tight font-mono", classColor(fc.predicted_class))}>{fc.predicted_class}</span>
              <div className="mono-sm text-bug-muted mt-0.5">PREDICTED CLASS</div>
            </div>
          </Card>

          <Card>
            <div className="px-4 py-2.5 border-b border-bug-hairline">
              <span className="mono-label text-bug-muted-soft">DETAILS</span>
            </div>
            <CardContent className="p-4 space-y-2.5 mono-sm">
              {[
                ["PROBABILITY", `${(fc.flare_probability * 100).toFixed(1)}%`],
                ["UNCERTAINTY", `±${(fc.uncertainty * 100).toFixed(1)}%`],
                ["CONFIDENCE", `${((1 - fc.uncertainty) * 100).toFixed(0)}%`],
                ["LEAD TIME", `${fc.lead_time_minutes.toFixed(0)} MIN`],
                ["PREDICTED CLASS", fc.predicted_class],
                ["MODEL", fc.model],
              ].map(([l, v]) => (
                <div key={l} className="flex justify-between">
                  <span className="text-bug-muted">{l}</span>
                  <span className="text-bug-text">{v}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {fc.physics_reason && (
            <Card>
              <div className="px-4 py-2.5 border-b border-bug-hairline">
                <span className="mono-label text-bug-muted-soft">PHYSICS</span>
              </div>
              <CardContent className="p-4">
                <p className="body-serif-sm text-bug-body leading-relaxed">{fc.physics_reason}</p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Class probabilities */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <div className="px-5 py-2.5 border-b border-bug-hairline">
              <span className="mono-label text-bug-muted-soft">CLASS PROBABILITIES</span>
            </div>
            <CardContent className="p-5">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(fc.class_probs).map(([cls, prob]) => (
                  <div key={cls} className="border border-bug-hairline p-4 text-center">
                    <div className={cn("text-[28px] font-light tracking-tight font-mono", classColor(cls))}>{cls}</div>
                    <div className="mono-sm text-bug-muted mt-1">{(prob * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {fc.similar_events.length > 0 && (
            <Card>
              <div className="px-5 py-2.5 border-b border-bug-hairline">
                <span className="mono-label text-bug-muted-soft">SIMILAR EVENTS</span>
              </div>
              <CardContent className="p-0 divide-y divide-bug-hairline">
                {fc.similar_events.map((e, i) => (
                  <div key={i} className="p-4 space-y-1">
                    <div className="flex justify-between">
                      <span className={cn("text-[16px] font-light tracking-tight font-mono", classColor(e.flare_class))}>{e.flare_class}</span>
                      <span className="mono-sm text-bug-muted">{e.date}</span>
                    </div>
                    <div className="mono-sm text-bug-muted-soft">
                      SIM: {(e.similarity * 100).toFixed(0)}% · LEAD: {e.lead_time.toFixed(0)} MIN
                    </div>
                    <p className="body-serif-sm text-bug-body">{e.description}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
