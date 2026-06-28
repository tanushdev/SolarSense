import type { NoaaData } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function NOAAStatus({ data, isLoading }: { data?: NoaaData; isLoading: boolean }) {
  if (isLoading || !data) {
    return (
      <Card>
        <CardContent className="p-4">
          <div className="animate-pulse space-y-2">
            <div className="h-3 bg-bug-card rounded w-24" />
            <div className="h-6 bg-bug-card rounded w-32" />
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="mono-label text-bug-muted">NOAA GOES XRS</div>
        <div className="flex items-baseline gap-2">
          <span className="text-[26px] font-light tracking-tight text-bug-text font-mono">
            {data.flux_w_m2 != null ? data.flux_w_m2.toExponential(2) : "—"}
          </span>
          <span className="mono-sm text-bug-muted-soft">W/m²</span>
        </div>
        <div className="flex items-center gap-3 mono-sm">
          <span className={cn(data.flux_class === "X" ? "text-bug-warning" : data.flux_class === "M" ? "text-bug-body-strong" : "text-bug-link")}>
            {data.flux_class}
          </span>
          <span className="text-bug-muted">{data.status.toUpperCase()}</span>
        </div>
        {data.flare_class && (
          <div className="mono-sm text-bug-muted">
            LATEST: {data.flare_class} @ {data.flare_peak_time?.slice(11, 19) ?? "—"}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
