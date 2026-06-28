import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function ProbabilityBreakdown({ data, predicted, className }: { data: Record<string, number>; predicted: string; className?: string }) {
  return (
    <Card className={className}>
      <CardHeader><CardTitle>CLASS PROBABILITIES</CardTitle></CardHeader>
      <CardContent>
        <div className="flex justify-center gap-2 flex-wrap">
          {Object.entries(data).map(([cls, prob]) => (
            <div key={cls} className={cn("border px-3 py-2 text-center", cls === predicted ? "border-bug-text" : "border-bug-hairline")}>
              <div className="text-[18px] font-light tracking-tight font-mono text-bug-text">{cls}</div>
              <div className="mono-sm text-bug-muted mt-0.5">{(prob * 100).toFixed(0)}%</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
