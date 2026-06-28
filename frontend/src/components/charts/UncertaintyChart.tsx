import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const axisStyle = { fontSize: 12, fill: "#666666" }
const data = Array.from({ length: 20 }, (_, i) => ({
  bin: `${(i * 5).toFixed(0)}`, count: Math.round(Math.exp(-((i - 10) ** 2) / 20) * 50 + Math.random() * 10),
}))

export function UncertaintyChart({ className }: { className?: string }) {
  return (
    <Card className={className}>
      <CardHeader><CardTitle>CONFIDENCE DISTRIBUTION</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <XAxis dataKey="bin" tick={axisStyle} tickLine={false} axisLine={{ stroke: "#262626" }} />
            <YAxis hide />
            <Bar dataKey="count" fill="#ffffff" fillOpacity={0.08} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
