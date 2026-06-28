import { cn } from "@/lib/utils"
import { Card, CardContent } from "@/components/ui/card"

export function MetricCard({
  label, value, unit
}: {
  label: string; value: string | number; unit?: string
}) {
  return (
    <Card>
      <CardContent className="p-4 space-y-1">
        <div className="mono-label text-bug-muted">{label}</div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[22px] font-light tracking-tight text-bug-text font-mono">
            {value}
          </span>
          {unit && <span className="mono-sm text-bug-muted-soft">{unit}</span>}
        </div>
      </CardContent>
    </Card>
  )
}
