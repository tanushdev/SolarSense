import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const axisStyle = { fontSize: 12, fill: "#666666" }
const data = Array.from({ length: 30 }, (_, i) => ({
  t: `T+${(i + 1) * 2}`, prob: +(Math.random() * 0.3 + 0.5 * Math.exp(-i / 20) + 0.1 * Math.sin(i / 3)).toFixed(3),
}))

export function ForecastTimeline({ className }: { className?: string }) {
  return (
    <Card className={className}>
      <CardHeader><CardTitle>FORECAST HORIZON</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={data}>
            <XAxis dataKey="t" tick={axisStyle} tickLine={false} axisLine={{ stroke: "#262626" }} />
            <YAxis domain={[0, 1]} tick={axisStyle} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tickLine={false} axisLine={false} width={45} />
            <Tooltip contentStyle={{ background: "#000000", border: "1px solid #262626", borderRadius: 0, fontSize: 12 }} />
            <Area type="monotone" dataKey="prob" stroke="#cccccc" strokeWidth={1.5} fill="#ffffff" fillOpacity={0.03} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
