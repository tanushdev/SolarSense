import { useMemo } from "react"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine, ReferenceArea } from "recharts"
import type { LightCurvePoint, ForecastPoint } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const axisStyle = { fontSize: 12, fill: "#666666" }
const tooltipStyle = { background: "#000000", border: "1px solid #262626", borderRadius: 0, fontSize: 12 }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload
    const softVal = dataPoint.soft_flux ?? dataPoint.precursor_soft ?? dataPoint.soft_forecast
    const hardVal = dataPoint.hard_flux ?? dataPoint.precursor_hard ?? dataPoint.hard_forecast
    const probVal = dataPoint.prob_value

    return (
      <div style={{ background: "#000000", border: "1px solid #262626", padding: "8px", fontSize: "12px", fontFamily: "monospace" }}>
        <p style={{ color: "#999999", marginBottom: "4px" }}>{label}</p>
        {softVal !== null && softVal !== undefined && (
          <p style={{ color: "#3b82f6" }}>Soft X-ray: {softVal.toExponential(3)} W/m²</p>
        )}
        {hardVal !== null && hardVal !== undefined && (
          <p style={{ color: "#ff9800" }}>Hard X-ray: {hardVal.toExponential(3)} W/m²</p>
        )}
        {probVal !== null && probVal !== undefined && (
          <p style={{ color: "#ef4444" }}>Probability: {(probVal * 100).toFixed(1)}%</p>
        )}
      </div>
    )
  }
  return null
}

export function LightCurveChart({ data, forecast, probability, className }: {
  data?: LightCurvePoint[]
  forecast?: ForecastPoint[]
  probability?: number
  className?: string
}) {
  const { chartData, lastObservedTimestamp, fifteenMinForecastTimestamp } = useMemo(() => {
    if (!data) return { chartData: undefined }
    const obs = data.slice(-200).map(d => ({
      timestamp: d.timestamp,
      soft_flux: d.soft_flux,
      hard_flux: d.hard_flux != null && d.hard_flux < 1e-5 ? d.hard_flux : null,
      soft_forecast: null as number | null,
      hard_forecast: null as number | null,
      precursor_soft: null as number | null,
      precursor_hard: null as number | null,
      prob_value: null as number | null,
    }))

    // Find the last non-null observed values and their indices to connect the forecast properly
    let lastSoftObsVal = null as number | null
    let lastHardObsVal = null as number | null
    let lastSoftIdx = -1
    let lastHardIdx = -1
    for (let idx = obs.length - 1; idx >= 0; idx--) {
      if (lastSoftObsVal === null && obs[idx].soft_flux !== null) {
        lastSoftObsVal = obs[idx].soft_flux
        lastSoftIdx = idx
      }
      if (lastHardObsVal === null && obs[idx].hard_flux !== null) {
        lastHardObsVal = obs[idx].hard_flux
        lastHardIdx = idx
      }
      if (lastSoftObsVal !== null && lastHardObsVal !== null) break
    }

    // Precursor heating: last 15 observed points colored RED when probability > 0.5 (impending flare)
    const obsLength = obs.length
    const precursorActive = probability != null && probability > 0.5
    const precursorPointsCount = 15

    const processedObs = obs.map((d, idx) => {
      const isPrecursor = precursorActive && (obsLength - idx <= precursorPointsCount)
      const isTransitionPoint = precursorActive && (obsLength - idx === precursorPointsCount + 1)
      return {
        ...d,
        soft_flux: isPrecursor ? null : d.soft_flux,
        hard_flux: isPrecursor ? null : d.hard_flux,
        precursor_soft: isPrecursor || isTransitionPoint ? d.soft_flux : null,
        precursor_hard: isPrecursor || isTransitionPoint ? d.hard_flux : null,
        // Bridge gap: start forecast lines from the last non-null observed point index
        soft_forecast: (lastSoftIdx !== -1 && idx >= lastSoftIdx) ? lastSoftObsVal : null,
        hard_forecast: (lastHardIdx !== -1 && idx >= lastHardIdx) ? lastHardObsVal : null,
      }
    })

    if (!forecast || forecast.length === 0) return { chartData: processedObs }

    const last = obs[obs.length - 1]
    const lastObservedTimestamp = last.timestamp
    const fifteenMinForecastTimestamp = forecast[Math.min(15, forecast.length - 1)]?.timestamp

    const forecastPoints = forecast.map((f, i) => ({
      timestamp: f.timestamp,
      soft_flux: null as number | null,
      hard_flux: null as number | null,
      soft_forecast: f.soft_flux,
      hard_forecast: f.hard_flux,
      precursor_soft: null as number | null,
      precursor_hard: null as number | null,
      prob_value: f.probability,
    }))

    return {
      chartData: [...processedObs, ...forecastPoints],
      lastObservedTimestamp,
      fifteenMinForecastTimestamp
    }
  }, [data, forecast, probability])

  return (
    <Card className={className}>
      <CardHeader><CardTitle>LIGHT CURVES</CardTitle></CardHeader>
      <CardContent>
        {!data || data.length === 0 ? (
          <div className="h-48 flex items-center justify-center mono-label text-bug-muted">NO DATA</div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData}>
              <XAxis dataKey="timestamp" tick={axisStyle} tickLine={false} axisLine={{ stroke: "#262626" }}
                tickFormatter={(v) => v?.slice(11, 16) ?? v?.slice(5, 10) ?? ""} />
              <YAxis yAxisId="flux" scale="log" domain={[1e-9, 1e-4]} tick={axisStyle} tickLine={false} axisLine={false}
                width={55} tickFormatter={(v) => v.toExponential(0)} />
              <YAxis yAxisId="prob" orientation="right" tick={axisStyle} tickLine={false} axisLine={false}
                domain={[0, 1]} width={0} hide />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#999999" }} />
              
              {/* Observed Telemetry */}
              <Line yAxisId="flux" type="monotone" dataKey="soft_flux" stroke="#3b82f6" strokeWidth={1.5}
                strokeDasharray="4 4" dot={false} name="Soft X-ray" connectNulls />
              <Line yAxisId="flux" type="monotone" dataKey="hard_flux" stroke="#ff9800" strokeWidth={1.5}
                dot={false} name="Hard X-ray" connectNulls />

              {/* Precursor Heating Highlight */}
              <Line yAxisId="flux" type="monotone" dataKey="precursor_soft" stroke="#ef4444" strokeWidth={2.5}
                strokeDasharray="4 4" dot={false} name="Precursor Heating (Soft)" connectNulls />
              <Line yAxisId="flux" type="monotone" dataKey="precursor_hard" stroke="#dc2626" strokeWidth={2.5}
                dot={false} name="Precursor Heating (Hard)" connectNulls />

              {/* Forecast Predictions */}
              <Line yAxisId="flux" type="monotone" dataKey="soft_forecast" stroke="#666666" strokeWidth={1.5}
                strokeDasharray="6 3" dot={false} name="Soft Forecast" connectNulls />
              <Line yAxisId="flux" type="monotone" dataKey="hard_forecast" stroke="#666666" strokeWidth={1.5}
                dot={false} name="Hard Forecast" connectNulls />

              {/* Current Time T = 0 Vertical Dashed Line */}
              {lastObservedTimestamp && (
                <ReferenceLine x={lastObservedTimestamp} stroke="#aaaaaa" strokeDasharray="4 4" strokeWidth={1.5}
                  label={{ value: "Current Time (T = 0)", fill: "#aaaaaa", position: "insideTopLeft", fontSize: 11 }} />
              )}

              {/* 15-Minute Shaded Forecast Window */}
              {lastObservedTimestamp && fifteenMinForecastTimestamp && (
                <ReferenceArea x1={lastObservedTimestamp} x2={fifteenMinForecastTimestamp}
                  fill="#ef4444" fillOpacity={0.08} label={{ value: "15-Min Window", fill: "#ef4444", position: "insideTop", fontSize: 10 }} />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}
