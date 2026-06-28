import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts"
import type { LightCurvePoint } from "@/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const axisStyle = { fontSize: 12, fill: "#666666" }
const tooltipStyle = { background: "#000000", border: "1px solid #262626", borderRadius: 0, fontSize: 12 }

export function LightCurveChart({ data, className }: { data?: LightCurvePoint[]; className?: string }) {
  return (
    <Card className={className}>
      <CardHeader><CardTitle>LIGHT CURVES</CardTitle></CardHeader>
      <CardContent>
        {!data || data.length === 0 ? (
          <div className="h-48 flex items-center justify-center mono-label text-bug-muted">NO DATA</div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data.slice(-200)}>
              <XAxis dataKey="timestamp" tick={axisStyle} tickLine={false} axisLine={{ stroke: "#262626" }}
                tickFormatter={(v) => v?.slice(11, 16) ?? ""} />
              <YAxis yAxisId="soft" tick={axisStyle} tickLine={false} axisLine={false} domain={["auto", "auto"]} width={55} />
              <YAxis yAxisId="hard" orientation="right" tick={axisStyle} tickLine={false} axisLine={false} domain={["auto", "auto"]} width={55} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#999999" }} itemStyle={{ color: "#ffffff" }} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#999999" }} />
              <Line yAxisId="soft" type="monotone" dataKey="soft_flux" stroke="#cccccc" strokeWidth={1.5} dot={false} name="Soft X-ray" />
              <Line yAxisId="hard" type="monotone" dataKey="hard_flux" stroke="#cccccc" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Hard X-ray" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}
