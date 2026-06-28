import { cn } from "@/lib/utils"

const styles = {
  high: "border-bug-hairline-strong text-bug-warning",
  low_confidence: "border-bug-hairline text-bug-muted",
  info: "border-bug-hairline text-bug-link",
}

export function AlertBanner({
  show, level = "info", message
}: {
  show: boolean; level?: 'high' | 'low_confidence' | 'info'; message: string
}) {
  if (!show) return null
  return (
    <div className={cn("flex items-center gap-2.5 border px-4 py-2.5 mono-sm", styles[level])}>
      <span>{message}</span>
    </div>
  )
}
